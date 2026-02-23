import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

class QPooler(nn.Module):
    """
    Q-Pooling mechanism that uses learnable queries to pool information
    from sequence features. More expressive than simple additive attention.
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
        
    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (batch, seq_len, hidden_dim)
        Returns:
            pooled: (batch, num_queries, hidden_dim)
            attn_weights: (batch, num_heads, num_queries, seq_len)
        """
        batch_size, seq_len, hidden_dim = features.shape
        
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)  # (batch, num_queries, hidden_dim)
        
        # Project Q, K, V
        Q = self.q_proj(queries)  # (batch, num_queries, hidden_dim)
        K = self.k_proj(features)  # (batch, seq_len, hidden_dim)
        V = self.v_proj(features)  # (batch, seq_len, hidden_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, self.num_queries, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # (batch, num_heads, num_queries/seq_len, head_dim)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)  # (batch, num_heads, num_queries, seq_len)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attended = torch.matmul(attn_weights, V)  # (batch, num_heads, num_queries, head_dim)
        
        # Concatenate heads
        attended = attended.transpose(1, 2).contiguous().view(batch_size, self.num_queries, hidden_dim)
        
        # Output projection
        output = self.out_proj(attended)
        output = self.layer_norm(output + queries)  # Residual connection
        
        return output, attn_weights

class MultiLayerFeatureExtractor(nn.Module):
    """
    Extracts and fuses features from multiple VLM layers.
    This provides richer representations than using only the last layer.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_layers_to_use: int = 4,
        fusion_method: str = "learned_weighted"  # "concat", "learned_weighted", "attention"
    ):
        super().__init__()
        self.num_layers_to_use = num_layers_to_use
        self.fusion_method = fusion_method
        
        if fusion_method == "learned_weighted":
            # Learnable weights for each layer
            self.layer_weights = nn.Parameter(torch.ones(num_layers_to_use))
        elif fusion_method == "attention":
            # Attention-based fusion
            self.fusion_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Linear(hidden_dim // 4, 1)
            )
        elif fusion_method == "concat":
            # Projection to reduce concatenated dimension
            self.fusion_proj = nn.Linear(hidden_dim * num_layers_to_use, hidden_dim)
    
    def forward(self, hidden_states: Tuple[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            hidden_states: Tuple of tensors, each (batch, seq_len, hidden_dim)
        Returns:
            fused_features: (batch, seq_len, hidden_dim)
        """
        # Take the last num_layers_to_use layers
        selected_layers = hidden_states[-self.num_layers_to_use:]
        
        if self.fusion_method == "learned_weighted":
            # Weighted sum with learnable weights
            weights = F.softmax(self.layer_weights, dim=0)
            fused = sum(w * layer for w, layer in zip(weights, selected_layers))
            
        elif self.fusion_method == "attention":
            # Stack layers and use attention to weight them
            stacked = torch.stack(selected_layers, dim=1)  # (batch, num_layers, seq_len, hidden_dim)
            batch, num_layers, seq_len, hidden_dim = stacked.shape
            
            # Reshape for attention computation
            stacked_reshaped = stacked.view(batch * seq_len, num_layers, hidden_dim)
            attn_scores = self.fusion_attn(stacked_reshaped)  # (batch*seq_len, num_layers, 1)
            attn_weights = F.softmax(attn_scores, dim=1)
            
            # Apply attention weights
            weighted = (stacked_reshaped * attn_weights).sum(dim=1)  # (batch*seq_len, hidden_dim)
            fused = weighted.view(batch, seq_len, hidden_dim)
            
        elif self.fusion_method == "concat":
            # Concatenate and project
            concatenated = torch.cat(selected_layers, dim=-1)  # (batch, seq_len, hidden_dim * num_layers)
            fused = self.fusion_proj(concatenated)
        
        return fused
    
class HierarchicalContrastiveLoss(nn.Module):
    """
    Three-level contrastive loss:

    Level 1 — Same episode     → similarity target 1.0  (same scene, same goal)
    Level 2 — Same task, diff episode → target 0.5      (same intent, diff scene)
    Level 3 — Different task   → target 0.0             (pushed apart)

    Why three levels?
    Keying only on task_name collapses all episodes of 'pick-place' to one
    point, making the latent blind to where the puck actually is.
    Keying only on episode_id gives no signal that 'pick-place' episodes
    are related to each other at all.
    The soft middle target lets the encoder learn:
        "same task = nearby region of latent space,
        same episode = almost identical latent,
        different task = far apart"
    """
    def __init__(
        self,
        temperature: float = 0.07,
        same_episode_weight: float = 1.0,   # how hard to pull same-episode pairs
        same_task_weight: float = 0.5,      # how hard to pull same-task pairs
    ):
        super().__init__()
        self.temperature         = temperature
        self.same_episode_weight = same_episode_weight
        self.same_task_weight    = same_task_weight

    def _build_labels(
        self,
        task_names:  List[str],
        episode_ids: List[int],
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build a soft similarity target matrix.

        Returns:
            labels: (B, B) float tensor with values in {0.0, 0.5, 1.0}
        """
        B = len(task_names)

        # Vectorized task comparison
        task_hashes = torch.tensor(
            [hash(t) for t in task_names], device=device
        )
        same_task = (task_hashes.unsqueeze(0) == task_hashes.unsqueeze(1))  # (B, B) bool

        # Vectorized episode comparison
        ep_ids = torch.tensor(episode_ids, device=device)
        same_ep = (ep_ids.unsqueeze(0) == ep_ids.unsqueeze(1))              # (B, B) bool

        # Build soft label matrix — order matters: episode overrides task
        labels = torch.zeros(B, B, device=device)
        labels[same_task]                   = self.same_task_weight    # 0.5
        labels[same_ep]                     = self.same_episode_weight # 1.0 (overrides 0.5)
        labels.fill_diagonal_(0.0)          # exclude self-similarity

        return labels

    def forward(
        self,
        latents:     torch.Tensor,   # (B, latent_dim)
        task_names:  List[str],
        episode_ids: List[int],
    ) -> torch.Tensor:
        B = latents.shape[0]
        if B < 2:
            return torch.tensor(0.0, device=latents.device)

        labels = self._build_labels(task_names, episode_ids, latents.device)

        # Skip batch if no useful signal (all different tasks, no pairs)
        if labels.sum() == 0:
            return torch.tensor(0.0, device=latents.device)

        latents_norm = F.normalize(latents, dim=-1)
        similarity   = (latents_norm @ latents_norm.T) / self.temperature  # (B, B)

        # Mask diagonal
        eye        = torch.eye(B, dtype=torch.bool, device=latents.device)
        similarity = similarity.masked_fill(eye, -1e9)

        # Soft InfoNCE: instead of binary positive/negative,
        # weight the numerator by the soft label (0, 0.5, or 1.0)
        exp_sim      = similarity.exp()                        # (B, B)
        weighted_pos = (exp_sim * labels).sum(dim=-1)          # (B,) — soft positive sum
        total_sum    = exp_sim.sum(dim=-1)                     # (B,)

        # Only compute loss for rows that have at least one positive-ish pair
        has_signal = labels.sum(dim=-1) > 0
        if not has_signal.any():
            return torch.tensor(0.0, device=latents.device)

        loss = -torch.log(weighted_pos[has_signal] / (total_sum[has_signal] + 1e-8))
        return loss.mean()

class LatentTaskEncoder(nn.Module):
    """
    Enhanced task encoder with Q-Pooling and multi-layer feature extraction.
    Includes reconstruction head for self-supervised pre-alignment.
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
        
        # Multi-layer feature extraction 
        if use_multi_layer:
            self.feature_extractor = MultiLayerFeatureExtractor(
                hidden_dim=vlm_hidden_dim,
                num_layers_to_use=num_vlm_layers_to_use,
                fusion_method=layer_fusion_method
            )
        
        # Q-Pooling mechanism
        self.q_pooler = QPooler(
            hidden_dim=vlm_hidden_dim,
            num_queries=num_pooling_queries,
            num_heads=num_attention_heads,
            dropout=dropout
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
            nn.LayerNorm(latent_dim)
        )
        
        # Reconstruction head 
        # Reconstructs the pooled VLM features (not the full sequence)
        self.reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, vlm_hidden_dim * num_pooling_queries)
        )
        
        # Predict original sequence length features for richer reconstruction
        self.sequence_reconstruction_head = nn.Sequential(
            nn.Linear(latent_dim, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2048, vlm_hidden_dim)
        )

    def forward(
        self,
        vlm_features: torch.Tensor,
        vlm_hidden_states: Optional[Tuple[torch.Tensor]] = None,
        return_attention_weights: bool = False
    ) -> dict:
        """
        Args:
            vlm_features: (batch, seq_len, hidden_dim) - last layer features
            vlm_hidden_states: Optional tuple of all layer hidden states for multi-layer extraction
            return_attention_weights: Whether to return Q-Pooler attention weights
            
        Returns:
            dict with keys:
                - latent: (batch, latent_dim)
                - reconstructed_pooled: (batch, num_queries * hidden_dim)
                - pooled_features: (batch, num_queries, hidden_dim)
                - reconstructed_sequence: (batch, hidden_dim) - mean pool reconstruction
                - attention_weights: Optional (batch, num_heads, num_queries, seq_len)
        """
        # Multi-layer feature fusion if enabled
        if self.use_multi_layer and vlm_hidden_states is not None:
            features = self.feature_extractor(vlm_hidden_states)
        else:
            features = vlm_features
        
        # Q-Pooling
        pooled_features, attn_weights = self.q_pooler(features)  
        # pooled_features: (batch, num_queries, hidden_dim)
        
        # Flatten pooled features for encoder
        batch_size = pooled_features.shape[0]
        pooled_flat = pooled_features.view(batch_size, -1)
        
        # Encode to latent space
        latent = self.encoder(pooled_flat)
        
        # Reconstruction
        reconstructed_pooled = self.reconstruction_head(latent)
        reconstructed_sequence = self.sequence_reconstruction_head(latent)
        
        output = {
            'latent': latent,
            'reconstructed_pooled': reconstructed_pooled,
            'pooled_features': pooled_features,
            'reconstructed_sequence': reconstructed_sequence,
            'pooled_flat': pooled_flat,  # For reconstruction loss
        }
        
        if return_attention_weights:
            output['attention_weights'] = attn_weights
        
        return output

class TemporalConsistencyLoss(nn.Module):
    """
    Encourage temporally consistent latents between VLM update timesteps.
    This ensures smooth behavior when System 2 runs at lower frequency than System 1.
    """
    def __init__(self, weight=0.1, epsilon=1e-6):
        super().__init__()
        self.weight = weight
        self.epsilon = epsilon
    
    def forward(self, latents, update_mask):
        """
        Args:
            latents: (B, T, latent_dim) - computed latents for each timestep
            update_mask: (T,) or (B, T) - bool tensor, True = VLM was actually computed
        
        Returns:
            loss: scalar tensor
            loss_dict: dict with breakdown
        """
        if latents.shape[1] <= 1:
            # No temporal consistency needed for single timestep
            return torch.tensor(0.0, device=latents.device), {}
        
        # Handle both (T,) and (B, T) masks
        if update_mask.ndim == 1:
            update_mask = update_mask.unsqueeze(0).expand(latents.shape[0], -1)
        
        # Compute differences between consecutive timesteps
        latent_diffs = latents[:, 1:] - latents[:, :-1]  # (B, T-1, latent_dim)
        
        # Create mask for non-update transitions (where latent should be constant)
        # non_update_transitions[t] = True if timestep t+1 is NOT an update
        non_update_transitions = ~update_mask[:, 1:]  # (B, T-1)
        
        if non_update_transitions.any():
            # Latents should be identical between updates
            # Compute MSE only for non-update transitions
            consistency_loss = (latent_diffs ** 2).sum(dim=-1)  # (B, T-1)
            
            # Mask and average
            masked_loss = consistency_loss * non_update_transitions.float()
            num_non_updates = non_update_transitions.float().sum() + self.epsilon
            consistency_loss = masked_loss.sum() / num_non_updates
            
            total_loss = self.weight * consistency_loss
            
            loss_dict = {
                'temporal_consistency': total_loss.item(),
                'consistency_raw': consistency_loss.item(),
                'num_non_update_transitions': non_update_transitions.sum().item()
            }
            
            return total_loss, loss_dict
        else:
            # All timesteps are updates, no consistency constraint needed
            return torch.tensor(0.0, device=latents.device), {
                'temporal_consistency': 0.0,
                'num_non_update_transitions': 0
            }

class ReconstructionLoss(nn.Module):
    """
    Multi-component reconstruction loss for pre-alignment training.
    Helps the encoder learn meaningful representations before DP3 gradients.
    """
    def __init__(
        self,
        pooled_weight: float = 1.0,
        sequence_weight: float = 0.5,
        latent_reg_weight: float = 0.01
    ):
        super().__init__()
        self.pooled_weight = pooled_weight
        self.sequence_weight = sequence_weight
        self.latent_reg_weight = latent_reg_weight
    
    def forward(
        self,
        encoder_output: dict,
        vlm_features: torch.Tensor,
        pooled_flat_target: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            encoder_output: Output dict from LatentTaskEncoder
            vlm_features: (batch, seq_len, hidden_dim) - original VLM features
            pooled_flat_target: (batch, num_queries * hidden_dim) - target pooled features
            
        Returns:
            total_loss: scalar
            loss_dict: dict with individual loss components
        """
        # Pooled feature reconstruction loss
        pooled_recon_loss = F.mse_loss(
            encoder_output['reconstructed_pooled'],
            pooled_flat_target.detach()
        )
        
        # Sequence-level reconstruction loss (reconstruct mean-pooled features)
        mean_pooled_target = vlm_features.mean(dim=1)  # (batch, hidden_dim)
        sequence_recon_loss = F.mse_loss(
            encoder_output['reconstructed_sequence'],
            mean_pooled_target.detach()
        )
        
        # Latent regularization (prevent collapse, encourage spread)
        latent = encoder_output['latent']
        latent_reg = -torch.log(latent.std(dim=0).mean() + 1e-8).clamp(max=10.0)  # Encourage variance
        
        # Total loss
        total_loss = (
            self.pooled_weight * pooled_recon_loss +
            self.sequence_weight * sequence_recon_loss +
            self.latent_reg_weight * latent_reg
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'pooled_recon': pooled_recon_loss.item(),
            'sequence_recon': sequence_recon_loss.item(),
            'latent_reg': latent_reg.item()
        }
        
        return total_loss, loss_dict
