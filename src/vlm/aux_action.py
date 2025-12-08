import torch
import torch.nn as nn

class ActionPredictorAuxHead(nn.Module):
    def __init__(self, latent_dim, action_dim, horizon, num_heads=8, hidden_dim=512, dropout=0.1):
        super().__init__()
        self.horizon = horizon
        self.action_dim = action_dim

        # Learned pool query one vector that attends over all latent slots
        self.pool_q = nn.Parameter(torch.empty(1, 1, latent_dim))
        nn.init.trunc_normal_(self.pool_q, std=0.02)

        self.norm_kv = nn.LayerNorm(latent_dim)
        self.attn = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )

        self.net = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, latent_seq: torch.Tensor) -> torch.Tensor:
        B = latent_seq.shape[0]
        kv = self.norm_kv(latent_seq)
        q = self.pool_q.expand(B, -1, -1)
        pooled, _ = self.attn(q, kv, kv, need_weights=False)
        pooled = pooled.squeeze(1)
        return self.net(pooled).view(B, self.horizon, self.action_dim)
