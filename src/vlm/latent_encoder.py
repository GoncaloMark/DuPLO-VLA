import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# RMSNorm helper (standalone so we don't depend on torch version)
# --------------------------------------------------------------------------- #
class RMSNorm(nn.Module):
    """
    Standard RMSNorm with a learnable gain (gamma). No bias.
    Compatible with torch < 2.4 which doesn't have nn.RMSNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in fp32 for numerical stability, cast back at the end.
        dtype = x.dtype
        x = x.float()
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * rms).to(dtype) * self.weight


# --------------------------------------------------------------------------- #
# Q-Pooler: per-layer norm + projection, cross-attention into learned queries
# --------------------------------------------------------------------------- #
class QPooler(nn.Module):
    """
    DeepStack-style pyramid pooler: samples several VLM decoder layers,
    normalizes and projects each one independently, concatenates them into
    one long memory sequence, and uses learned queries to cross-attend into
    that memory.

    Args:
        input_dim:     width of the raw VLM hidden states (e.g. 3072 for
                       Qwen3-VL-4B — pass self.vlm.config.hidden_size).
        hidden_dim:    pooler internal dim.
        num_queries:   number of learned query vectors (order-invariant set).
        num_heads:     attention heads in cross-attention.
        num_layers:    number of VLM layers being sampled. Must match the
                       length of the first dim of `all_hidden_states`.
        dropout:       dropout in attention and FFN.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        num_queries: int = 16,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers

        # Per-layer RMSNorm + linear projection. Each sampled VLM layer
        # gets its own parameters so the pooler can handle different
        # magnitude statistics at different depths without the final-norm
        # trick.
        self.layer_norms = nn.ModuleList(
            [RMSNorm(input_dim) for _ in range(num_layers)]
        )
        self.layer_projs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)]
        )

        # Learned queries (order-invariant set)
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.2)

        # Cross-attention: queries attend into concatenated memory
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        # FFN refinement
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        all_hidden_states: list[torch.Tensor] | torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ):
        """
        Args:
            all_hidden_states: either
              - a list/tuple of length num_layers, each (B, L_text, input_dim),
              - or a stacked tensor (B, num_layers, L_text, input_dim) as
                produced by the feature extractor.
            key_padding_mask: optional (B, L_text) bool — True positions are
                VALID. This module internally converts to the MultiheadAttention
                convention (True = mask out).

        Returns:
            pooled:       (B, num_queries, hidden_dim)
            attn_weights: (B, num_queries, L_total) — attention weights over
                          concatenated memory. Useful for debugging / viz.
        """
        # Normalize input format to a list of per-layer tensors
        if isinstance(all_hidden_states, torch.Tensor):
            assert all_hidden_states.dim() == 4, (
                "Expected (B, num_layers, L, D) if passing a stacked tensor; "
                f"got {tuple(all_hidden_states.shape)}"
            )
            layer_list = [all_hidden_states[:, i] for i in range(self.num_layers)]
        else:
            layer_list = list(all_hidden_states)
            assert len(layer_list) == self.num_layers, (
                f"Expected {self.num_layers} layers, got {len(layer_list)}"
            )

        B = layer_list[0].size(0)

        # Per-layer norm + projection
        projected = []
        for i, h in enumerate(layer_list):
            h = self.layer_norms[i](h)             # (B, L, input_dim)
            h = self.layer_projs[i](h)             # (B, L, hidden_dim)
            projected.append(h)

        # Concatenate into one memory sequence: (B, num_layers * L, hidden_dim)
        memory = torch.cat(projected, dim=1)

        # Build the attention mask over the concatenated memory.
        # MultiheadAttention wants True = "ignore this key". Our incoming
        # convention (from the extractor) is True = "valid", so invert.
        attn_key_padding = None
        if key_padding_mask is not None:
            # Replicate the per-layer mask across layers (each layer sees
            # the same text token validity).
            mask_valid = torch.cat([key_padding_mask] * self.num_layers, dim=1)
            attn_key_padding = ~mask_valid.bool()

        # Cross-attention: queries attend into memory
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)   # (B, Q, hidden)
        q_input = self.query_norm(queries)
        attended, attn_weights = self.cross_attn(
            query=q_input,
            key=memory,
            value=memory,
            key_padding_mask=attn_key_padding,
            need_weights=True,
            average_attn_weights=True,
        )
        x = queries + attended

        # FFN refinement
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights


# --------------------------------------------------------------------------- #
# Full latent encoder: Q-Pooler -> latent sequence + pooled vector + gate
# --------------------------------------------------------------------------- #
class LatentTaskEncoder(nn.Module):
    """
    Wraps the Q-Pooler and produces all three outputs downstream code needs:

        latent_seq:    (B, num_queries, latent_dim)  — policy conditioning
        latent:        (B, latent_dim)               — VICReg
        latent_normed: (B, latent_dim)               — contrastive

    A zero-init learnable gate scales latent_seq so the planner's output
    starts at zero. The diffusion policy will effectively ignore the latent
    at init and the gate will open during joint training. This is the
    stability trick from ThinkJEPA / AdaLN-zero applied at the planner boundary.

    Args:
        vlm_hidden_dim: width of raw VLM layer outputs (e.g. 3072).
        num_layers:     how many VLM layers the Q-Pooler consumes.
        q_hidden_dim:   Q-Pooler internal width.
        latent_dim:     downstream latent width (what the policy sees).
        num_pooling_queries: number of learned queries.
        num_attention_heads: attention heads.
        dropout:        dropout.
        gate_output:    if True, apply zero-init tanh gate to latent_seq.
                        Turn OFF for ablations that want to measure raw
                        planner behavior without the gate.
    """

    def __init__(
        self,
        vlm_hidden_dim: int = 3072,
        num_layers: int = 4,
        q_hidden_dim: int = 768,
        latent_dim: int = 512,
        num_pooling_queries: int = 16,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        gate_output: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.q_pooler = QPooler(
            input_dim=vlm_hidden_dim,
            hidden_dim=q_hidden_dim,
            num_queries=num_pooling_queries,
            num_heads=num_attention_heads,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Per-query MLP head: Q-Pooler hidden width -> latent_dim.
        # Applied independently per query (as an nn.Linear over the last dim),
        # so structural per-query information is preserved.
        self.encoder = nn.Sequential(
            nn.Linear(q_hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        # Zero-init gate. At init: tanh(0) = 0 -> latent_seq is exactly zero.
        # During training the scalar moves off zero and the gate opens.
        self.gate_output = gate_output
        if gate_output:
            self.output_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        vlm_hidden_states: list[torch.Tensor] | torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        return_attention_weights: bool = False,
    ) -> dict:
        """
        Args:
            vlm_hidden_states: see QPooler.forward.
            key_padding_mask:  optional (B, L_text) bool; True = valid.
            return_attention_weights: include Q-Pooler attention in output.

        Returns a dict with:
            latent_seq:    (B, num_queries, latent_dim) for policy
            latent:        (B, latent_dim) for VICReg
            latent_normed: (B, latent_dim) for contrastive
            pooled:        (B, num_queries, q_hidden_dim) — pre-encoder
            gate_value:    scalar, current value of the output gate (for logging)
            attention_weights (optional)
        """
        pooled, attn_weights = self.q_pooler(
            vlm_hidden_states, key_padding_mask=key_padding_mask
        )

        # Per-query latent: apply the encoder MLP over the last dim for every
        # query independently. This preserves per-query specialization.
        latent_seq = self.encoder(pooled)                     # (B, Q, latent_dim)

        # Zero-init gate on the sequence consumed by the policy
        if self.gate_output:
            gate = torch.tanh(self.output_gate)
            latent_seq = latent_seq * gate
        else:
            gate = torch.tensor(1.0, device=latent_seq.device)

        # Pooled vector for SSL losses. Note: we mean-pool the POST-gate
        # sequence so VICReg also sees the real scale the policy sees;
        # at init this is zero, and VICReg's variance term will immediately
        # start pushing the gate open. That's by design, we want the SSL
        # losses to actively work to open the gate.
        latent_vec = latent_seq.mean(dim=1)                   # (B, latent_dim)
        latent_normed = F.normalize(latent_vec, dim=-1, eps=1e-8)

        out = {
            "latent_seq":    latent_seq,
            "latent":        latent_vec,
            "latent_normed": latent_normed,
            "pooled":        pooled,
            "gate_value":    gate.detach(),
        }
        if return_attention_weights:
            out["attention_weights"] = attn_weights
        return out


# --------------------------------------------------------------------------- #
# Losses (unchanged from your version, included for completeness so you
# can drop this file in and keep working)
# --------------------------------------------------------------------------- #
class TemporalContrastiveLoss(nn.Module):
    """
    Temporal InfoNCE: frames from the same episode are positives,
    frames from different episodes are negatives.

    Inputs:
        latents:     (B, D) — typically latent_normed
        episode_ids: list or tensor of length B, int ids
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        latents: torch.Tensor,
        episode_ids,
    ) -> torch.Tensor:
        B = latents.shape[0]
        if B < 2:
            return latents.sum() * 0.0

        device = latents.device
        if not isinstance(episode_ids, torch.Tensor):
            ep_ids = torch.tensor(list(episode_ids), device=device)
        else:
            ep_ids = episode_ids.to(device)

        same_ep = ep_ids.unsqueeze(0) == ep_ids.unsqueeze(1)
        diag = torch.eye(B, dtype=torch.bool, device=device)
        positives = same_ep & ~diag

        valid_rows = positives.any(dim=1)
        if not valid_rows.any():
            return latents.sum() * 0.0

        sim = (latents @ latents.T) / self.temperature
        sim = sim.masked_fill(diag, float("-inf"))

        log_probs = F.log_softmax(sim, dim=-1)
        pos_counts = positives.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        target = positives.float() / pos_counts

        loss = -(target[valid_rows] * log_probs[valid_rows]).sum(dim=-1)
        return loss.mean()


def vicreg_loss(latent: torch.Tensor, eps: float = 1e-4):
    if latent.ndim == 3:
        # B = Batch, Q = Queries, D = Latent Dim (512)
        B, Q, D = latent.shape
        latent = latent.reshape(B * Q, D) 
    z = latent - latent.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0) + eps)
    var_loss = F.relu(1.0 - std).mean()

    D_dim = z.shape[1]
    cov = (z.T @ z) / max(z.shape[0] - 1, 1)
    off_diag = cov - torch.diag(torch.diagonal(cov))
    cov_loss = off_diag.pow(2).sum() / D_dim
    
    return var_loss, cov_loss
