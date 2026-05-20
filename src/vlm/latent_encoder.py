import torch
import torch.nn as nn
import torch.nn.functional as F

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

class QPoolerBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        # 1. Self-Attention (Queries coordinate with each other)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 2. Cross-Attention (Queries extract from VLM memory)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 3. FFN
        self.norm3 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )

    def forward(self, queries, memory, memory_key_padding_mask=None):
        # Self-Attention
        q_norm = self.norm1(queries)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm)
        queries = queries + self_out
        
        # Cross-Attention
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
        
        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        
        return queries, attn_weights


class QPooler(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 4,      # Number of VLM layers sampled
        num_pooler_blocks: int = 3, # Depth of the pooler itself
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.num_layers = num_layers

        self.layer_norms = nn.ModuleList([RMSNorm(input_dim) for _ in range(num_layers)])
        self.layer_projs = nn.ModuleList([nn.Linear(input_dim, hidden_dim) for _ in range(num_layers)])

        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        nn.init.trunc_normal_(self.queries, std=0.2)

        # Stack of Q-Pooler blocks (Self-Attn -> Cross-Attn -> FFN)
        self.blocks = nn.ModuleList([
            QPoolerBlock(hidden_dim, num_heads, dropout) 
            for _ in range(num_pooler_blocks)
        ])

    def forward(self, all_hidden_states, key_padding_mask=None):
        if isinstance(all_hidden_states, torch.Tensor):
            layer_list = [all_hidden_states[:, i] for i in range(self.num_layers)]
        else:
            layer_list = list(all_hidden_states)

        B = layer_list[0].size(0)

        projected = []
        for i, h in enumerate(layer_list):
            h = self.layer_projs[i](self.layer_norms[i](h))
            projected.append(h)

        memory = torch.cat(projected, dim=1)

        attn_key_padding = None
        if key_padding_mask is not None:
            mask_valid = torch.cat([key_padding_mask] * self.num_layers, dim=1)
            attn_key_padding = ~mask_valid.bool()

        x = self.queries.unsqueeze(0).expand(B, -1, -1)
        
        # Pass through iterative blocks
        for block in self.blocks:
            x, last_attn_weights = block(x, memory, memory_key_padding_mask=attn_key_padding)

        return x, last_attn_weights

class LatentTaskEncoder(nn.Module):
    def __init__(
        self,
        vlm_hidden_dim: int = 2560,
        num_layers: int = 4,
        q_hidden_dim: int = 768,
        latent_dim: int = 512,
        num_pooling_queries: int = 32,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        self.q_pooler = QPooler(
            input_dim=vlm_hidden_dim,
            hidden_dim=q_hidden_dim,
            num_queries=num_pooling_queries,
            num_heads=num_attention_heads,
            num_layers=num_layers,
            num_pooler_blocks=3, # 3 layers deep for better coordination
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

    def forward(self, vlm_hidden_states, key_padding_mask=None, return_attention_weights=False):
        pooled, attn_weights = self.q_pooler(vlm_hidden_states, key_padding_mask=key_padding_mask)
        latent_seq = self.encoder(pooled)
        latent_vec = latent_seq.mean(dim=1)
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

def vcreg_loss(latent_seq: torch.Tensor, eps: float = 1e-4):
    """
    latent_seq shape: (B, Q, D) -> (Batch, Queries=64, Dim=512)
    Calcula o VICReg ao longo do Batch para cada query de forma independente.
    """
    B, Q, D = latent_seq.shape
    total_var_loss = 0.0
    total_cov_loss = 0.0
    
    for q in range(Q):
        # Isola uma query de cada vez ao longo de todo o batch: (B, D)
        z_q = latent_seq[:, q, :] 
        z_q = z_q - z_q.mean(dim=0, keepdim=True)
        
        # Variância
        std = torch.sqrt(z_q.var(dim=0) + eps)
        total_var_loss += F.relu(1.0 - std).mean()
        
        # Covariância
        cov = (z_q.T @ z_q) / max(B - 1, 1)
        off_diag = cov - torch.diag(torch.diagonal(cov))
        total_cov_loss += off_diag.pow(2).sum() / D

    return total_var_loss / Q, total_cov_loss / Q

