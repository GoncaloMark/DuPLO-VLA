import torch
import torch.nn as nn
import torch.nn.functional as F

class LatentTaskEncoder(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int,
        latent_dim: int = 512,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        self.attention = nn.Sequential(
            nn.Linear(vlm_hidden_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        ).to(torch.bfloat16)

        self.encoder = nn.Sequential(
            nn.Linear(vlm_hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),  
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim)
        ).to(torch.bfloat16)

    def attention_pooling(self, features: torch.Tensor) -> torch.Tensor:
        """Attention-based pooling over sequence dimension"""
        # features: (batch, seq_len, hidden_dim)
        attn_weights = self.attention(features)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)
        pooled = (features * attn_weights).sum(dim=1)  # (batch, hidden_dim)
        return pooled

    def forward(
        self,
        vlm_features: torch.Tensor,
    ) -> torch.Tensor:
        pooled = self.attention_pooling(vlm_features)
        latent = self.encoder(pooled)
        
        return latent
