"""
Latent task encoder with temporal-aware Q-Pooler.

Changes vs. the previous version:
  * Accepts both 4D `(B, num_layers, L, D)` and 5D `(B, T, num_layers, L, D)`
    inputs. In the 5D case, the T frames are flattened into the memory
    sequence with a learned temporal embedding so the queries can tell
    "now" from "a few steps ago".
  * RMSNorm everywhere (matching what Qwen3-VL uses internally).
  * Query init std reduced to 0.02 (BERT-style) to avoid saturating
    cross-attention softmax at init.
  * No VICReg call inside the encoder. `vcreg_loss` is kept as a free
    function for ablation, but the new training loop doesn't use it.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype) * self.weight


# --------------------------------------------------------------------------- #
class QPoolerBlock(nn.Module):
    """
    Pre-norm block: queries do self-attn, then cross-attn into a memory
    sequence, then an FFN. RMSNorm everywhere.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = RMSNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.norm3 = RMSNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, queries, memory, memory_key_padding_mask=None):
        q_norm = self.norm1(queries)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        queries = queries + self_out

        c_norm = self.norm2(queries)
        cross_out, attn_weights = self.cross_attn(
            query=c_norm,
            key=memory,
            value=memory,
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        queries = queries + cross_out

        queries = queries + self.ffn(self.norm3(queries))
        return queries, attn_weights


# --------------------------------------------------------------------------- #
class QPooler(nn.Module):
    """
    Q-Former-style pooler with temporal awareness.

    Memory construction:
      input  : either (B, num_layers, L, D_vlm)
                  or (B, T, num_layers, L, D_vlm)
      per layer: RMSNorm + linear projection to hidden_dim
      per (layer, time) slot: add a learned temporal embedding (only if T > 1)
      concat across layers (and time): memory shape (B, M, hidden_dim)

    Key padding mask follows the same flattening so the queries
    correctly ignore padded text positions.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        num_queries: int = 64,
        num_heads: int = 8,
        num_layers: int = 4,
        num_pooler_blocks: int = 3,
        max_obs_horizon: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers
        self.max_obs_horizon = max_obs_horizon
        self.hidden_dim = hidden_dim

        self.layer_norms = nn.ModuleList([RMSNorm(input_dim) for _ in range(num_layers)])
        self.layer_projs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Temporal embedding: one vector per obs-horizon slot.
        # Added to every token coming from a given slot.
        if self.max_obs_horizon > 1:
            self.temporal_embed = nn.Parameter(
                torch.zeros(max_obs_horizon, hidden_dim)
            )
            nn.init.trunc_normal_(self.temporal_embed, std=0.02)
        else:
            self.temporal_embed = None

        # Learnable queries.
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.blocks = nn.ModuleList(
            [
                QPoolerBlock(hidden_dim, num_heads, dropout)
                for _ in range(num_pooler_blocks)
            ]
        )

    # ------------------------------------------------------------------ #
    def _build_memory(self, hidden_states, key_padding_mask):
        """
        hidden_states: 4D (B, num_layers, L, D) or 5D (B, T, num_layers, L, D)
        key_padding_mask: matching shape minus the D axis, or None.

        Returns:
            memory          : (B, M, hidden_dim)
            attn_key_padding: (B, M) bool — True = ignore, or None
        """
        is_temporal = hidden_states.dim() == 5

        if is_temporal:
            B, T, NL, L, D = hidden_states.shape
            assert NL == self.num_layers, (
                f"hidden_states has {NL} layers but encoder expects {self.num_layers}"
            )
            assert T <= self.max_obs_horizon, (
                f"obs_horizon={T} exceeds max_obs_horizon={self.max_obs_horizon}"
            )
        else:
            B, NL, L, D = hidden_states.shape
            T = 1
            hidden_states = hidden_states.unsqueeze(1)  # (B, 1, NL, L, D)
            assert NL == self.num_layers

        layer_tokens = []  # list of (B, T*L, hidden_dim)
        for i in range(self.num_layers):
            h = hidden_states[:, :, i]                  # (B, T, L, D)
            h = self.layer_norms[i](h)
            h = self.layer_projs[i](h)                  # (B, T, L, hidden_dim)

            # Add temporal embedding per slot.
            if self.temporal_embed is not None:
                t_emb = self.temporal_embed[:T].view(1, T, 1, self.hidden_dim)
                h = h + t_emb # broadcast over L

            h = h.reshape(B, T * L, self.hidden_dim)
            layer_tokens.append(h)

        # Concat over layers in the sequence dimension.
        memory = torch.cat(layer_tokens, dim=1)         # (B, num_layers * T * L, D)

        attn_key_padding = None
        if key_padding_mask is not None:
            # Accept (B, L), (B, T, L), or (B, num_layers, L), etc.
            mask = key_padding_mask.bool()
            if mask.dim() == 2:
                # (B, L) — repeat for T then for layers
                mask = mask.unsqueeze(1).expand(-1, T, -1)  # (B, T, L)
            if mask.dim() == 3:
                # (B, T, L)
                mask = mask.reshape(B, T * L)
            else:
                raise ValueError(f"Unsupported mask shape {key_padding_mask.shape}")
            # repeat across layers
            mask = torch.cat([mask] * self.num_layers, dim=1)  # (B, num_layers*T*L)
            # MultiheadAttention's key_padding_mask: True = position is *ignored*.
            attn_key_padding = ~mask
        return memory, attn_key_padding

    # ------------------------------------------------------------------ #
    def forward(self, hidden_states, key_padding_mask=None):
        memory, attn_key_padding = self._build_memory(hidden_states, key_padding_mask)
        B = memory.size(0)

        x = self.queries.unsqueeze(0).expand(B, -1, -1).contiguous()
        last_attn = None
        for block in self.blocks:
            x, last_attn = block(x, memory, memory_key_padding_mask=attn_key_padding)
        return x, last_attn


# --------------------------------------------------------------------------- #
class LatentTaskEncoder(nn.Module):
    """
    Wraps the Q-Pooler with a small MLP head and produces two outputs:
      - latent_seq: (B, num_queries, latent_dim) — for cross-attn conditioning
      - latent    : (B, latent_dim)              — for the aux action head
    """

    def __init__(
        self,
        vlm_hidden_dim: int = 2560,
        num_layers: int = 4,
        q_hidden_dim: int = 768,
        latent_dim: int = 512,
        num_pooling_queries: int = 64,
        num_attention_heads: int = 8,
        max_obs_horizon: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_layers = num_layers

        self.q_pooler = QPooler(
            input_dim=vlm_hidden_dim,
            hidden_dim=q_hidden_dim,
            num_queries=num_pooling_queries,
            num_heads=num_attention_heads,
            num_layers=num_layers,
            num_pooler_blocks=3,
            max_obs_horizon=max_obs_horizon,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(
            nn.Linear(q_hidden_dim, 1024),
            RMSNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim),
            RMSNorm(latent_dim),
        )

    def forward(self, vlm_hidden_states, key_padding_mask=None,
                return_attention_weights=False):
        pooled, attn_weights = self.q_pooler(
            vlm_hidden_states, key_padding_mask=key_padding_mask
        )                                               # (B, Q, q_hidden_dim)
        latent_seq = self.encoder(pooled)               # (B, Q, latent_dim)
        # Pooled global vector for the aux head. Mean over queries —
        # since queries are order-invariant slots this is the natural choice.
        latent_vec = latent_seq.mean(dim=1)             # (B, latent_dim)
        latent_normed = F.normalize(latent_vec, dim=-1, eps=1e-8)

        out = {
            "latent_seq": latent_seq,
            "latent": latent_vec,
            "latent_normed": latent_normed,
            "pooled": pooled,
        }
        if return_attention_weights:
            out["attention_weights"] = attn_weights
        return out
