import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# RMSNorm helper
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
# Q-Pooler — unchanged from previous version
# --------------------------------------------------------------------------- #
class QPooler(nn.Module):
    """
    DeepStack-style pyramid pooler: samples several VLM decoder layers,
    normalizes and projects each one independently, concatenates them into
    one long memory sequence, and uses learned queries to cross-attend into
    that memory.
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

        self.layer_norms = nn.ModuleList(
            [RMSNorm(input_dim) for _ in range(num_layers)]
        )
        self.layer_projs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)]
        )

        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.2)

        self.query_norm = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout,
        )

        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(
        self,
        all_hidden_states,
        key_padding_mask=None,
    ):
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

        projected = []
        for i, h in enumerate(layer_list):
            h = self.layer_norms[i](h)
            h = self.layer_projs[i](h)
            projected.append(h)
        memory = torch.cat(projected, dim=1)

        attn_key_padding = None
        if key_padding_mask is not None:
            mask_valid = torch.cat([key_padding_mask] * self.num_layers, dim=1)
            attn_key_padding = ~mask_valid.bool()

        queries = self.queries.unsqueeze(0).expand(B, -1, -1)
        q_input = self.query_norm(queries)
        attended, attn_weights = self.cross_attn(
            query=q_input, key=memory, value=memory,
            key_padding_mask=attn_key_padding,
            need_weights=True, average_attn_weights=True,
        )
        x = queries + attended
        x = x + self.ffn(self.ffn_norm(x))

        return x, attn_weights


# --------------------------------------------------------------------------- #
# LatentTaskEncoder — gate init 0.1, pre-gate output exposed
# --------------------------------------------------------------------------- #
class LatentTaskEncoder(nn.Module):
    """
    Wraps the Q-Pooler and produces all the outputs downstream code needs:

        latent_seq:           (B, num_queries, latent_dim) post-gate, for policy
        latent_seq_pre_gate:  (B, num_queries, latent_dim) pre-gate, for SSL
        latent:               (B, latent_dim) mean of post-gate
        latent_normed:        (B, latent_dim) F.normalize of latent

    Args:
        gate_init: initial value for output_gate. Default 0.1.
                   tanh(0.1) ~= 0.0997, so the post-gate latent has ~10%
                   magnitude at start — enough for VICReg/policy gradients
                   to push the gate without being at a zero-gradient
                   fixed point.
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
        gate_init: float = 0.1,            # NEW: initial value for output_gate
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

        self.encoder = nn.Sequential(
            nn.Linear(q_hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.gate_output = gate_output
        if gate_output:
            self.output_gate = nn.Parameter(torch.tensor([float(gate_init)]))

    def forward(
        self,
        vlm_hidden_states,
        key_padding_mask=None,
        return_attention_weights: bool = False,
    ) -> dict:
        """
        Returns a dict with:
            latent_seq:           (B, num_queries, latent_dim) post-gate, for policy
            latent_seq_pre_gate:  (B, num_queries, latent_dim) pre-gate, for SSL
            latent:               (B, latent_dim) mean of post-gate
            latent_normed:        (B, latent_dim) F.normalize of latent
            pooled:               (B, num_queries, q_hidden_dim) pre-encoder
            gate_value:           detached scalar of tanh(output_gate)
            attention_weights:    (optional)
        """
        pooled, attn_weights = self.q_pooler(
            vlm_hidden_states, key_padding_mask=key_padding_mask
        )

        # Pre-gate output of the encoder MLP. SSL operates on this.
        latent_seq_pre_gate = self.encoder(pooled)            # (B, Q, latent_dim)

        # Gate scales the policy-facing output without affecting the SSL path.
        if self.gate_output:
            gate = torch.tanh(self.output_gate)
            latent_seq = latent_seq_pre_gate * gate
        else:
            gate = torch.tensor(1.0, device=latent_seq_pre_gate.device)
            latent_seq = latent_seq_pre_gate

        # Post-gate aggregates kept for downstream code that wants to see
        # exactly what the policy consumes.
        latent_vec = latent_seq.mean(dim=1)                   # (B, latent_dim)
        latent_normed = F.normalize(latent_vec, dim=-1, eps=1e-8)

        out = {
            "latent_seq":          latent_seq,
            "latent_seq_pre_gate": latent_seq_pre_gate,
            "latent":              latent_vec,
            "latent_normed":       latent_normed,
            "pooled":              pooled,
            "gate_value":          gate.detach(),
        }
        if return_attention_weights:
            out["attention_weights"] = attn_weights
        return out


# --------------------------------------------------------------------------- #
# Losses (unchanged)
# --------------------------------------------------------------------------- #
class TemporalContrastiveLoss(nn.Module):
    """
    Temporal InfoNCE: frames from the same episode are positives,
    frames from different episodes are negatives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, latents, episode_ids):
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
    """
    Variance + covariance terms of VICReg, applied to a flattened latent.
    Accepts either (B, D) or (B, Q, D) tensors. For (B, Q, D), reshapes
    to (B*Q, D) so each query position counts as a sample for the
    variance/covariance statistics.
    """
    if latent.ndim == 3:
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
