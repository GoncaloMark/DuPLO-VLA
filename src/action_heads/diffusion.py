"""
RDT-style diffusion policy with AdaLN-zero timestep conditioning
and adaptable cross-attention context.

Main self-attention sequence: [state, a_0, ..., a_{H-1}]  (length horizon + 1)
Timestep is NOT a token; it modulates every block via AdaLN-zero.
Cross-attention context comes from a user-supplied `cond_dict`:
each modality gets its own input projection, learned modality tag,
and an optional learned temporal/spatial positional embedding.

Training:  DDPM scheduler (clean add_noise loss).
Sampling:  DPMSolver (few-step inference).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp, RmsNorm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import (
    DPMSolverMultistepScheduler,
)


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def modulate(x, shift, scale):
    """
    Apply AdaLN shift/scale to a sequence.
    x: (B, N, D), shift/scale: (B, D). Broadcast over the sequence dim.
    """
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class _NoAffineRMSNorm(nn.Module):
    """
    Plain RMS norm with no learnable gamma. Used inside AdaLN blocks because
    the adaLN projection already supplies the (shift, scale) affine transform.
    """

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


# --------------------------------------------------------------------------- #
# Embedding layers
# --------------------------------------------------------------------------- #
class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep features -> 2-layer MLP -> hidden_dim."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.freq_size = frequency_embedding_size

    def forward(self, t):
        half = self.freq_size // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(t_freq.to(self.mlp[0].weight.dtype))


# --------------------------------------------------------------------------- #
# Cross-attention (query = main seq, key/value = context)
# --------------------------------------------------------------------------- #
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 norm_layer=RmsNorm):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, c, mask=None):
        B, N, C = x.shape
        L = c.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        kv = (
            self.kv(c)
            .reshape(B, L, 2, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        k, v = kv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if mask is not None:
            mask = mask.reshape(B, 1, 1, L).expand(-1, -1, N, -1)

        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


# --------------------------------------------------------------------------- #
# RDT block with AdaLN-zero
# --------------------------------------------------------------------------- #
class RDTBlock(nn.Module):
    """
    Three sublayers (self-attn, cross-attn, FFN), each wrapped with
    AdaLN-zero conditioning driven by the timestep embedding t_emb:

        shift, scale, gate = per-sublayer params derived from t_emb.
        x = x + gate * sublayer(modulate(norm(x), shift, scale))

    At init, all gates are 0 (via zero-init of the adaLN projection),
    so every sublayer starts as a no-op and the network is identity.
    """

    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.norm1 = _NoAffineRMSNorm(hidden_size)
        self.norm2 = _NoAffineRMSNorm(hidden_size)
        self.norm3 = _NoAffineRMSNorm(hidden_size)

        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )
        self.cross_attn = CrossAttention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=True,
            norm_layer=RmsNorm,
        )
        self.ffn = Mlp(
            in_features=hidden_size,
            hidden_features=hidden_size * 4,
            out_features=hidden_size,
            act_layer=lambda: nn.GELU(approximate="tanh"),
        )

        # One projection produces 9 modulation params:
        # (shift, scale, gate) × (self-attn, cross-attn, FFN).
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 9 * hidden_size, bias=True),
        )

    def forward(self, x, c, t_emb, cross_mask=None):
        (
            shift_sa, scale_sa, gate_sa,
            shift_ca, scale_ca, gate_ca,
            shift_mlp, scale_mlp, gate_mlp,
        ) = self.adaLN(t_emb).chunk(9, dim=-1)

        x = x + gate_sa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_sa, scale_sa)
        )
        x = x + gate_ca.unsqueeze(1) * self.cross_attn(
            modulate(self.norm2(x), shift_ca, scale_ca), c, cross_mask
        )
        x = x + gate_mlp.unsqueeze(1) * self.ffn(
            modulate(self.norm3(x), shift_mlp, scale_mlp)
        )
        return x


# --------------------------------------------------------------------------- #
# Final layer (also AdaLN-modulated; linear is zero-init so output starts at 0)
# --------------------------------------------------------------------------- #
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_dim):
        super().__init__()
        self.norm = _NoAffineRMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_dim)
        self.adaLN = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True),
        )

    def forward(self, x, t_emb):
        shift, scale = self.adaLN(t_emb).chunk(2, dim=-1)
        x = modulate(self.norm(x), shift, scale)
        return self.linear(x)


# --------------------------------------------------------------------------- #
# Policy
# --------------------------------------------------------------------------- #
class RDTPolicy(nn.Module):
    """
    Args:
        action_dim:  dim of one action in the horizon.
        horizon:     number of action steps predicted per forward pass.
        state_dim:   dim of the proprio/state vector (goes in the main sequence).
        cond_dims:   {modality_name: input_feature_dim} for cross-attn context.
                     e.g. {"img": 512, "latent": 4096}.
        cond_seq_lens: optional {modality_name: sequence_length}. Set for
                     modalities whose tokens have meaningful order (spatial
                     image tokens, temporal stacks). Omit for order-invariant
                     ones (Q-Former queries, pooled vectors).
        hidden_dim:  transformer width.
        depth:       number of RDTBlocks.
        num_heads:   attention heads.
        num_train_timesteps / num_inference_timesteps: diffusion step counts.
        prediction_type: 'sample' (predict x0) or 'epsilon' (predict noise).
        beta_schedule: passed through to both schedulers.
    """

    def __init__(
        self,
        action_dim,
        horizon,
        state_dim,
        cond_dims,
        cond_seq_lens=None,
        hidden_dim=512,
        depth=6,
        num_heads=8,
        num_train_timesteps=1000,
        num_inference_timesteps=20,
        prediction_type="sample",
        beta_schedule="squaredcos_cap_v2",
    ):
        super().__init__()
        self.action_dim = action_dim
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.prediction_type = prediction_type
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_timesteps = num_inference_timesteps

        # ---- Main sequence: [state, a_0, ..., a_{H-1}] ----
        self.t_embedder = TimestepEmbedder(hidden_dim)
        self.state_proj = nn.Linear(state_dim, hidden_dim)
        self.action_proj = nn.Linear(action_dim, hidden_dim)

        # Learned pos embed for length H+1 (state + actions; no time token now)
        self.main_pos_embed = nn.Parameter(torch.zeros(1, horizon + 1, hidden_dim))
        nn.init.trunc_normal_(self.main_pos_embed, std=0.02)

        # ---- Context (cross-attention) ----
        self.cond_proj = nn.ModuleDict(
            {k: nn.Linear(v, hidden_dim) for k, v in cond_dims.items()}
        )
        self.modality_embeds = nn.ParameterDict(
            {
                k: nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
                for k in cond_dims.keys()
            }
        )
        cond_seq_lens = cond_seq_lens or {}
        self.cond_pos_embeds = nn.ParameterDict()
        for k, L in cond_seq_lens.items():
            if L is not None and L > 1:
                p = nn.Parameter(torch.zeros(1, L, hidden_dim))
                nn.init.trunc_normal_(p, std=0.02)
                self.cond_pos_embeds[k] = p

        # ---- Transformer backbone ----
        self.blocks = nn.ModuleList(
            [RDTBlock(hidden_dim, num_heads) for _ in range(depth)]
        )
        self.final_layer = FinalLayer(hidden_dim, action_dim)

        self._init_weights()

        # ---- Noise schedulers ----
        self.train_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
            clip_sample=False,  # actions may live outside [-1, 1]
        )
        self.sample_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule=beta_schedule,
            prediction_type=prediction_type,
        )

    # ------------------------------------------------------------------ #
    def _init_weights(self):
        # Zero-init every AdaLN projection: all gates start at 0, so every
        # sublayer is a no-op at init and the residual stream passes through
        # unchanged. This is the "zero" in AdaLN-zero.
        for block in self.blocks:
            nn.init.zeros_(block.adaLN[-1].weight)
            nn.init.zeros_(block.adaLN[-1].bias)

        nn.init.zeros_(self.final_layer.adaLN[-1].weight)
        nn.init.zeros_(self.final_layer.adaLN[-1].bias)

        # Zero-init final projection so the model's output starts at 0
        # (the diffusion-standard "no update" initial condition).
        nn.init.zeros_(self.final_layer.linear.weight)
        nn.init.zeros_(self.final_layer.linear.bias)

    # ------------------------------------------------------------------ #
    def _build_context(self, cond_dict):
        """Project each modality, add modality tag + optional pos embed, concat."""
        tokens = []
        for key, value in cond_dict.items():
            if value.dim() == 2:
                value = value.unsqueeze(1)
            proj = self.cond_proj[key](value)
            proj = proj + self.modality_embeds[key]
            if key in self.cond_pos_embeds:
                L = proj.shape[1]
                proj = proj + self.cond_pos_embeds[key][:, :L]
            tokens.append(proj)
        return torch.cat(tokens, dim=1)

    # ------------------------------------------------------------------ #
    def forward_model(self, noisy_action, timesteps, state, cond_dict,
                      cross_mask=None):
        """
        noisy_action: (B, horizon, action_dim)
        timesteps:    (B,)
        state:        (B, state_dim) or (B, 1, state_dim)
        cond_dict:    dict of context inputs
        cross_mask:   optional (B, L_total) bool mask over flattened context
        Returns: (B, horizon, action_dim) matching prediction_type.
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)

        # Global conditioning signal for AdaLN modulation in every block.
        t_emb = self.t_embedder(timesteps)                       # (B, D)

        s_emb = self.state_proj(state)                           # (B, 1, D)
        a_emb = self.action_proj(noisy_action)                   # (B, H, D)
        main_seq = torch.cat([s_emb, a_emb], dim=1)              # (B, H+1, D)
        main_seq = main_seq + self.main_pos_embed

        context = self._build_context(cond_dict)                 # (B, L, D)

        for block in self.blocks:
            main_seq = block(main_seq, context, t_emb, cross_mask)

        # Drop the state token; keep only action positions.
        out = self.final_layer(main_seq[:, 1:], t_emb)
        return out                                               # (B, H, action_dim)

    # ------------------------------------------------------------------ #
    def compute_loss(self, action_gt, state, cond_dict, cross_mask=None):
        B = action_gt.shape[0]
        device = action_gt.device

        noise = torch.randn_like(action_gt)
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=device, dtype=torch.long
        )
        noisy_action = self.train_scheduler.add_noise(action_gt, noise, timesteps)

        pred = self.forward_model(
            noisy_action, timesteps, state, cond_dict, cross_mask
        )

        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "sample":
            target = action_gt
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        return F.mse_loss(pred, target)

    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(self, state, cond_dict, cross_mask=None,
               num_inference_steps=None):
        """Few-step sampling with DPMSolver. Returns (B, horizon, action_dim)."""
        B = state.shape[0]
        device = state.device
        dtype = state.dtype

        steps = num_inference_steps or self.num_inference_timesteps
        self.sample_scheduler.set_timesteps(steps, device=device)

        x = torch.randn(B, self.horizon, self.action_dim,
                        device=device, dtype=dtype)

        for t in self.sample_scheduler.timesteps:
            t_batch = t.expand(B).to(device)
            model_out = self.forward_model(x, t_batch, state, cond_dict, cross_mask)
            x = self.sample_scheduler.step(model_out, t, x).prev_sample
            x = x.to(dtype)

        return x

    # ------------------------------------------------------------------ #
    def forward(self, *args, **kwargs):
        return self.compute_loss(*args, **kwargs)
    
    def load_pretrained(
        self,
        source,
        key: str = "ema_policy",
        strict: bool = True,
        map_location: str = "cpu",
    ):
        """
        Load pre-trained weights into self.
    
        Args:
            source: one of
                - path to a .pt / .pth checkpoint (str or Path)
                - an already-loaded dict (result of torch.load)
                - a raw state_dict (flat dict of tensors)
            key: which sub-dict inside a checkpoint to use.
                Common choices from the DuPLO training script:
                * "ema_policy" — EMA copy (preferred for eval)
                * "policy"     — live, non-EMA weights
                Ignored if `source` is already a raw state_dict.
            strict: passed through to load_state_dict. Set False to tolerate
                missing or unexpected keys (e.g. when evaluating an older
                checkpoint after an architecture tweak — but verify the
                mismatch is harmless).
            map_location: where torch.load lands tensors before copying.
                "cpu" is safe; the subsequent copy into the module will
                move them to the right device/dtype automatically.
    
        Returns:
            (missing_keys, unexpected_keys) from load_state_dict, for
            inspection. With strict=True an error is raised on mismatch.
        """
        import torch
        from pathlib import Path
    
        # 1) Normalize `source` to a state_dict
        if isinstance(source, (str, Path)):
            obj = torch.load(str(source), map_location=map_location)
        else:
            obj = source
    
        if isinstance(obj, dict) and key in obj:
            state_dict = obj[key]
        elif isinstance(obj, dict) and all(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            # Already a raw state_dict
            state_dict = obj
        elif isinstance(obj, dict):
            available = [k for k in obj.keys() if not k.startswith("_")]
            raise KeyError(
                f"Checkpoint has no key '{key}'. Available keys: {available}. "
                f"Pass key=... to pick a different one, or pass a raw state_dict."
            )
        else:
            raise TypeError(f"Unsupported source type: {type(obj)}")
    
        # 2) Match dtype/device of the module before load
        target_dtype  = next(self.parameters()).dtype
        target_device = next(self.parameters()).device
        state_dict = {
            k: v.to(device=target_device, dtype=target_dtype)
            if v.is_floating_point() else v.to(device=target_device)
            for k, v in state_dict.items()
        }
    
        # 3) Load
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
    
        if missing or unexpected:
            print(f"[RDTPolicy.load_pretrained] missing: {missing}")
            print(f"[RDTPolicy.load_pretrained] unexpected: {unexpected}")
        else:
            print(f"[RDTPolicy.load_pretrained] loaded {len(state_dict)} tensors "
                f"from key='{key}'")
    
        return missing, unexpected
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path,
        key: str = "ema_policy",
        strict: bool = True,
        map_location: str = "cpu",
        **kwargs,
    ):
        """
        Construct an RDTPolicy and load pre-trained weights in one call.
    
        All RDTPolicy.__init__ args must be passed via **kwargs and must
        match the values used during training. Specifically:
            action_dim, horizon, state_dim, cond_dims, cond_seq_lens,
            hidden_dim, depth, num_heads, num_train_timesteps,
            num_inference_timesteps, prediction_type, beta_schedule.
    
        If any architectural value mismatches the checkpoint, load_state_dict
        will raise a shape error (with strict=True) or silently load the
        wrong subset (with strict=False).
    
        Example:
            policy = RDTPolicy.from_checkpoint(
                "duplo_pusht_epoch_100.pt",
                key            = "ema_policy",
                action_dim     = 2,
                horizon        = 16,
                state_dim      = 4,
                cond_dims      = {"latent": 512},
                cond_seq_lens  = {},
                hidden_dim     = 512,
                depth          = 6,
                num_heads      = 8,
                num_train_timesteps     = 1000,
                num_inference_timesteps = 16,
                prediction_type         = "sample",
            )
            policy.eval()
        """
        policy = cls(**kwargs)
        policy.load_pretrained(
            checkpoint_path,
            key          = key,
            strict       = strict,
            map_location = map_location,
        )
        return policy

