import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class QPooler(nn.Module):
    """
    Q-Pooling mechanism that uses learnable queries to pool information
    from sequence features. More expressive than simple additive attention.

    key_padding_mask (bool, optional): (batch, seq_len), True = ignore that position.
    In practice seq_len is constant for fixed image size + fixed instruction phrasings,
    so the mask is almost always None and adds zero overhead.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_queries: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.queries = nn.Parameter(torch.randn(num_queries, hidden_dim))
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        features: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, hidden_dim)
            key_padding_mask: (batch, seq_len) bool, True = ignore (padding position). Pass None when seq_len is constant — this is the fast path.
        Returns:
            pooled: (batch, num_queries, hidden_dim)
            attn_weights: (batch, num_heads, num_queries, seq_len)
        """
        batch_size, seq_len, hidden_dim = features.shape

        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        Q = self.q_proj(queries) # (batch, num_queries, hidden_dim)
        K = self.k_proj(features) # (batch, seq_len, hidden_dim)
        V = self.v_proj(features) # (batch, seq_len,  hidden_dim)

        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # shapes: (batch, num_heads, queries/seq_len, head_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # (batch, num_heads, num_queries, seq_len)

        # Apply key_padding_mask 
        # Masked positions get -inf before softmax → zero attention weight.
        # No-op when mask is None (the common case with fixed seq_len).
        if key_padding_mask is not None:
            # (batch, seq_len) → (batch, 1, 1, seq_len) for broadcast
            mask = key_padding_mask[:, None, None, :]
            scores = scores.masked_fill(mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, num_queries, seq_len)
        attn_weights = self.dropout(attn_weights)

        attended = torch.matmul(attn_weights, V)  # (batch, num_heads, num_queries, head_dim)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.num_queries, hidden_dim)

        output = self.out_proj(attended)
        output = self.layer_norm(output + queries)  # residual

        return output, attn_weights


class MultiLayerFeatureExtractor(nn.Module):
    """
    Extracts and fuses features from multiple VLM layers.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_layers_to_use: int = 4,
        fusion_method: str = "learned_weighted", # "concat", "learned_weighted", "attention"
    ):
        super().__init__()
        self.num_layers_to_use = num_layers_to_use
        self.fusion_method = fusion_method

        if fusion_method == "learned_weighted":
            self.layer_weights = nn.Parameter(torch.ones(num_layers_to_use))
        elif fusion_method == "attention":
            self.fusion_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        elif fusion_method == "concat":
            self.fusion_proj = nn.Linear(hidden_dim * num_layers_to_use, hidden_dim)

    def forward(self, hidden_states) -> torch.Tensor:
        """
        Args:
            hidden_states: tuple of tensors OR stacked tensor (batch, num_layers, seq_len, hidden_dim)
        Returns:
            fused: (batch, seq_len, hidden_dim)
        """
        # Accept both tuple (live VLM) and stacked tensor (precomputed) 
        if isinstance(hidden_states, torch.Tensor):
            # (batch, num_layers, seq_len, hidden_dim) → list of (batch, seq_len, hidden_dim)
            selected = [hidden_states[:, i] for i in range(hidden_states.shape[1])]
        else:
            # tuple of all VLM layers — take last num_layers_to_use
            selected = list(hidden_states[-self.num_layers_to_use:])

        if self.fusion_method == "learned_weighted":
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * layer for w, layer in zip(weights, selected))

        elif self.fusion_method == "attention":
            stacked = torch.stack(selected, dim=1)  # (batch, num_layers, seq_len, hidden_dim)
            batch, num_layers, seq_len, hidden_dim = stacked.shape
            stacked_reshaped = stacked.view(batch * seq_len, num_layers, hidden_dim)
            attn_scores = self.fusion_attn(stacked_reshaped)           # (B*seq, layers, 1)
            attn_weights = F.softmax(attn_scores, dim=1)
            weighted = (stacked_reshaped * attn_weights).sum(dim=1)  # (B*seq, hidden)
            fused = weighted.view(batch, seq_len, hidden_dim)

        elif self.fusion_method == "concat":
            concatenated = torch.cat(selected, dim=-1)
            fused = self.fusion_proj(concatenated)

        return fused


class HierarchicalContrastiveLoss(nn.Module):
    """
    Three-level contrastive loss:
        Same episode          -> soft label 1.0
        Same task, diff ep    -> soft label 0.5
        Different task        -> soft label 0.0
    """
    def __init__(
        self,
        temperature: float = 0.07,
        same_episode_weight: float = 1.0,
        same_task_weight: float = 0.5,
    ):
        super().__init__()
        self.temperature         = temperature
        self.same_episode_weight = same_episode_weight
        self.same_task_weight    = same_task_weight

    def _build_labels(self, task_names, episode_ids, device):
        B = len(task_names)
        task_hashes = torch.tensor([hash(t) for t in task_names], device=device)
        same_task = (task_hashes.unsqueeze(0) == task_hashes.unsqueeze(1))
        ep_ids = torch.tensor(episode_ids, device=device)
        same_ep = (ep_ids.unsqueeze(0) == ep_ids.unsqueeze(1))

        labels = torch.zeros(B, B, device=device)
        labels[same_task] = self.same_task_weight
        labels[same_ep]   = self.same_episode_weight
        labels.fill_diagonal_(0.0)
        return labels

    def forward(self, latents, task_names, episode_ids):
        B = latents.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=latents.device)

        labels = self._build_labels(task_names, episode_ids, latents.device)
        if labels.sum() == 0:
            return torch.tensor(0.0, device=latents.device)

        latents_norm = F.normalize(latents, dim=-1)
        similarity   = (latents_norm @ latents_norm.T) / self.temperature

        eye        = torch.eye(B, dtype=torch.bool, device=latents.device)
        similarity = similarity.masked_fill(eye, -1e9)

        exp_sim = similarity.exp()
        weighted_pos = (exp_sim * labels).sum(dim=-1)
        total_sum = exp_sim.sum(dim=-1)

        has_signal = labels.sum(dim=-1) > 0
        if not has_signal.any():
            return torch.tensor(0.0, device=latents.device)

        loss = -torch.log(weighted_pos[has_signal] / (total_sum[has_signal] + 1e-8))
        return loss.mean()


class LatentTaskEncoder(nn.Module):
    """
    Enhanced task encoder with Q-Pooling and multi-layer feature extraction.
    """
    def __init__(
        self,
        vlm_hidden_dim: int,
        latent_dim: int = 512,
        num_pooling_queries: int = 8,
        num_attention_heads: int = 8,
        num_vlm_layers_to_use: int = 4,
        layer_fusion_method: str = "learned_weighted",
        use_multi_layer: bool = True,
        dropout: float = 0.1,
    ):
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
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
        )

        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, vlm_hidden_dim * num_pooling_queries),
        )

        self.sequence_reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, vlm_hidden_dim),
        )

    def forward(
        self,
        vlm_features: torch.Tensor,
        vlm_hidden_states=None,
        return_attention_weights: bool = False,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            vlm_features: (batch, seq_len, hidden_dim) — last layer
            vlm_hidden_states : tuple of all layers (live path) OR
                                (batch, num_layers, seq_len, hidden_dim) tensor (precomputed)
                                OR None (disables multi-layer fusion)
            return_attention_weights: whether to include Q-Pooler attn in output
            key_padding_mask: (batch, seq_len) bool, True = padding. Almost always
                                None since seq_len is constant.
        """
        # Multi-layer feature fusion
        if self.use_multi_layer and vlm_hidden_states is not None:
            features = self.feature_extractor(vlm_hidden_states)
        else:
            features = vlm_features

        # Q-Pooling - mask is a no-op when None (constant seq_len case)
        pooled_features, attn_weights = self.q_pooler(features, key_padding_mask=key_padding_mask)

        batch_size = pooled_features.shape[0]
        pooled_flat = pooled_features.view(batch_size, -1)

        latent = self.encoder(pooled_flat)

        reconstructed_pooled = self.reconstruction_head(latent)
        reconstructed_sequence = self.sequence_reconstruction_head(latent)

        output = {
            'latent': latent,
            'reconstructed_pooled': reconstructed_pooled,
            'pooled_features': pooled_features,
            'reconstructed_sequence': reconstructed_sequence,
            'pooled_flat': pooled_flat,
        }

        if return_attention_weights:
            output['attention_weights'] = attn_weights

        return output


class TemporalConsistencyLoss(nn.Module):
    """Encourage temporally consistent latents between VLM update timesteps."""
    def __init__(self, weight=0.1, epsilon=1e-6):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon

    def forward(self, latents, update_mask):
        if latents.shape[1] <= 1:
            return torch.tensor(0.0, device=latents.device), {}

        if update_mask.ndim == 1:
            update_mask = update_mask.unsqueeze(0).expand(latents.shape[0], -1)

        latent_diffs = latents[:, 1:] - latents[:, :-1]
        non_update_transitions = ~update_mask[:, 1:]

        if non_update_transitions.any():
            consistency_loss = (latent_diffs ** 2).sum(dim=-1)
            masked_loss = consistency_loss * non_update_transitions.float()
            num_non_updates = non_update_transitions.float().sum() + self.epsilon
            consistency_loss = masked_loss.sum() / num_non_updates
            total_loss = self.weight * consistency_loss

            return total_loss, {
                'temporal_consistency': total_loss.item(),
                'consistency_raw': consistency_loss.item(),
                'num_non_update_transitions': non_update_transitions.sum().item(),
            }
        else:
            return torch.tensor(0.0, device=latents.device), {
                'temporal_consistency': 0.0,
                'num_non_update_transitions': 0,
            }


class SceneGoalLoss(nn.Module):
    """
    Pushes the current-frame latent toward the goal-frame latent.

    This is the fix for goal-state blindness: without this loss the encoder
    can only learn WHAT task is happening (task category), not WHERE the goal
    is in the scene. Two pick-place episodes with opposite goal positions would
    produce identical latents — and DP3 would have no way to distinguish them.

    With this loss, the encoder learns that "same task, different goal position"
    = different latent region, giving DP3 the spatial goal information it needs.

    Implementation:
        loss = 1 - cosine_similarity(current_latent, stop_grad(goal_latent))

    Stop-gradient on goal latent is important: we want the current frame to
    chase the goal, not the goal to collapse toward the current frame.

    Weight: 0.1 - meaningful signal but subordinate to contrastive (0.01 scale
    is too weak here since goal separation is the primary spatial signal).
    """
    def __init__(self, weight: float = 0.1):
        super().__init__()
        self.weight = weight

    def forward(
        self,
        current_latent: torch.Tensor,   # (B, latent_dim)
        goal_latent:    torch.Tensor,   # (B, latent_dim)
    ) -> torch.Tensor:
        # Stop gradient on goal - current chases goal, not the other way around
        goal_latent_sg = goal_latent.detach()

        cos_sim = F.cosine_similarity(current_latent, goal_latent_sg, dim=-1)  # (B,)
        loss = (1.0 - cos_sim).mean()

        return self.weight * loss

class ReconstructionLoss(nn.Module):
    """Multi-component reconstruction loss for pre-alignment training."""
    def __init__(
        self,
        pooled_weight: float = 1.0,
        sequence_weight: float = 0.5,
        latent_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.pooled_weight = pooled_weight
        self.sequence_weight = sequence_weight
        self.latent_reg_weight = latent_reg_weight

    def forward(self, encoder_output, vlm_features, pooled_flat_target):
        pooled_recon_loss = F.mse_loss(
            encoder_output['reconstructed_pooled'],
            pooled_flat_target.detach(),
        )

        mean_pooled_target = vlm_features.mean(dim=1)
        sequence_recon_loss = F.mse_loss(
            encoder_output['reconstructed_sequence'],
            mean_pooled_target.detach(),
        )

        latent = encoder_output['latent']
        latent_reg = -torch.log(latent.std(dim=0).mean() + 1e-8).clamp(max=10.0)

        total_loss = (
            self.pooled_weight * pooled_recon_loss +
            self.sequence_weight * sequence_recon_loss +
            self.latent_reg_weight * latent_reg
        )

        return total_loss, {
            'total': total_loss.item(),
            'pooled_recon': pooled_recon_loss.item(),
            'sequence_recon': sequence_recon_loss.item(),
            'latent_reg': latent_reg.item(),
        }
