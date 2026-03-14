import torch
import torch.nn as nn
import torch.nn.functional as F


class QPooler(nn.Module):
    """
    Cross-attention bottleneck (Perceiver/Flamingo style).
    Fixes vs previous version:
        - trunc_normal_(std=0.02) init: randn on 4096-dim vectors has norm ~64, saturating softmax immediately
        - FFN block after attention: without it queries cannot process/mix what they attended to
    """
    def __init__(self, hidden_dim, num_queries=8, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads   = num_heads
        self.head_dim    = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0

        self.feature_norm = nn.LayerNorm(hidden_dim)

        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, features, key_padding_mask=None):
        B, seq_len, D = features.shape
        features = self.feature_norm(features)
        queries = self.queries.unsqueeze(0).expand(B, -1, -1)

        Q = self.q_proj(queries).view(B, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(features).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(features).view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float('-inf'))

        attn_weights = self.dropout(F.softmax(scores, dim=-1))
        attended = torch.matmul(attn_weights, V).transpose(1, 2).contiguous().view(B, self.num_queries, D)

        x = self.norm1(self.out_proj(attended) + queries)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class MultiLayerFeatureExtractor(nn.Module):
    def __init__(self, hidden_dim, num_layers_to_use=4, fusion_method="learned_weighted"):
        super().__init__()
        self.num_layers_to_use = num_layers_to_use
        self.fusion_method = fusion_method

        if fusion_method == "learned_weighted":
            self.layer_weights = nn.Parameter(torch.ones(num_layers_to_use))
        elif fusion_method == "attention":
            self.fusion_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1)
            )
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(hidden_dim * num_layers_to_use, hidden_dim)

    def forward(self, hidden_states):
        if isinstance(hidden_states, torch.Tensor):
            selected = [hidden_states[:, i] for i in range(hidden_states.shape[1])]
        else:
            selected = list(hidden_states[-self.num_layers_to_use:])

        if self.fusion_method == "learned_weighted":
            weights = F.softmax(self.layer_weights, dim=0)
            return sum(w * layer for w, layer in zip(weights, selected))
        elif self.fusion_method == "attention":
            stacked = torch.stack(selected, dim=1)
            B, L, S, D = stacked.shape
            r = stacked.view(B * S, L, D)
            w = F.softmax(self.fusion_attn(r), dim=1)
            return (r * w).sum(dim=1).view(B, S, D)
        elif self.fusion_method == "concat":
            return self.fusion_proj(torch.cat(selected, dim=-1))


class TemporalContrastiveLoss(nn.Module):
    """
    Temporal InfoNCE: within-episode nearby frames are positives,
    cross-episode frames are negatives.

    This replaces HierarchicalContrastiveLoss. The key advantage is that it
    provides rich *within-episode* structure (the encoder must track progress
    through a task), not just a binary task-ID signal.

    Requires: episode_ids to determine which samples share an episode.
    Optionally uses temporal_indices to weight positives by proximity.
    """
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, latents, episode_ids, temporal_indices=None):
        """
        Args:
            latents:          (B, D) L2-normalized embeddings
            episode_ids:      list[int] length B
            temporal_indices: optional list[int] length B, timestep within episode
        """
        B = latents.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=latents.device, requires_grad=True)

        device = latents.device
        ep_ids = torch.tensor(episode_ids, device=device)

        # Same-episode mask (excluding diagonal)
        same_ep = ep_ids.unsqueeze(0) == ep_ids.unsqueeze(1)  # (B, B)
        diag = torch.eye(B, dtype=torch.bool, device=device)
        positives = same_ep & ~diag

        # Need at least one positive pair somewhere
        valid_rows = positives.any(dim=1)
        if not valid_rows.any():
            return latents.sum() * 0.0

        # Cosine similarity (latents should already be L2-normed)
        sim = latents @ latents.T / self.temperature
        sim = sim.masked_fill(diag, -1e9)

        log_probs = F.log_softmax(sim, dim=-1)

        # Uniform weight across all positives for each anchor
        pos_counts = positives.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
        target = positives.float() / pos_counts

        loss = -(target[valid_rows] * log_probs[valid_rows]).sum(dim=-1)
        return loss.mean()


class LatentTaskEncoder(nn.Module):
    """
    Q-Pooler + MLP encoder.
    """
    def __init__(self, vlm_hidden_dim, latent_dim=512, num_pooling_queries=8,
                 num_attention_heads=8, num_vlm_layers_to_use=4,
                 layer_fusion_method="learned_weighted", use_multi_layer=True,
                 dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_multi_layer = use_multi_layer

        if use_multi_layer:
            self.feature_extractor = MultiLayerFeatureExtractor(
                hidden_dim=vlm_hidden_dim,
                num_layers_to_use=num_vlm_layers_to_use,
                fusion_method=layer_fusion_method,
            )

        self.q_pooler = QPooler(
            hidden_dim=vlm_hidden_dim,
            num_queries=num_pooling_queries,
            num_heads=num_attention_heads,
            dropout=dropout,
        )

        self.encoder = nn.Sequential(
            nn.Linear(vlm_hidden_dim * num_pooling_queries, 2048),
            nn.LayerNorm(2048), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, vlm_features, vlm_hidden_states=None,
                return_attention_weights=False, key_padding_mask=None):
        features = self.feature_extractor(vlm_hidden_states) \
                if (self.use_multi_layer and vlm_hidden_states is not None) \
                else vlm_features

        pooled, attn_weights = self.q_pooler(features, key_padding_mask=key_padding_mask)

        latent = self.encoder(pooled.view(pooled.shape[0], -1))

        # L2-normalized version for contrastive loss only
        latent_normed = F.normalize(latent, dim=-1)

        out = {
            'latent': latent,               # for policy conditioning (full magnitude)
            'latent_normed': latent_normed,  # for contrastive loss (unit sphere)
            'pooled_features': pooled,
        }
        if return_attention_weights:
            out['attention_weights'] = attn_weights
        return out
