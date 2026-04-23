"""
Ablation configs for LIBERO-Object thesis experiments.

Seven variants total:
  1. Duplo                 — main method (VLM + Q-Pooler + SSL)
  2. Ablation A            — remove SSL losses, keep Q-Pooler
  3. Ablation B            — last-layer only (no pyramid)
  4. Ablation C            — mean-pool VLM (no Q-Pooler)
  5. Ablation D            — no VLM latent at all
  6. Baseline CLIP         — CLIP pooled features
  7. Baseline DP           — no language/VLM conditioning (lower bound)

Each config is a plain dataclass so it serializes cleanly for wandb/checkpoints.
The base trainer reads these and wires up the right components.
"""

from dataclasses import dataclass, field
from typing import Optional


# --------------------------------------------------------------------------- #
# Planner config — which visual-language encoder produces the latent
# --------------------------------------------------------------------------- #
@dataclass
class PlannerConfig:
    kind: str                              # 'qpooler' | 'vlm_mean' | 'clip' | 'none'
    vlm_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    clip_name: str = "openai/clip-vit-base-patch32"

    # Q-Pooler specifics (ignored if kind != 'qpooler')
    layer_indices: tuple = (8, 16, 24, 32)
    num_pooling_queries: int = 16
    q_hidden_dim: int = 768
    num_attention_heads: int = 8

    # Shared output dim that the policy sees per latent token
    latent_dim: int = 512

    # SSL losses (only applied when planner produces a latent vector)
    use_contrastive: bool = True
    use_vicreg: bool = True
    contrastive_weight: float = 1.0
    vicreg_var_weight: float = 1.0
    vicreg_cov_weight: float = 1.0


# --------------------------------------------------------------------------- #
# Policy config — how the latent enters the diffusion transformer
# --------------------------------------------------------------------------- #
@dataclass
class PolicyConfig:
    # Conditioning mode:
    #   'sequence' -> cond_dict["latent"] is (B, num_queries, D); set of tokens
    #   'vector'   -> cond_dict["latent"] is (B, D);              single token
    #   'none'     -> no latent modality at all (baseline DP)
    latent_mode: str = "sequence"

    # Backbone
    hidden_dim: int = 512
    depth: int = 6
    num_heads: int = 8
    horizon: int = 16

    # Diffusion
    num_train_timesteps: int = 1000
    num_inference_timesteps: int = 20
    prediction_type: str = "sample"
    beta_schedule: str = "squaredcos_cap_v2"


# --------------------------------------------------------------------------- #
# Full variant config
# --------------------------------------------------------------------------- #
@dataclass
class VariantConfig:
    name: str
    description: str
    planner: PlannerConfig
    policy: PolicyConfig
    # Whether the image encoder feeds the policy directly (all variants do,
    # for fair comparison — the VLM path is the *additional* signal).
    use_resnet_vision: bool = True


# --------------------------------------------------------------------------- #
# The 7 variants
# --------------------------------------------------------------------------- #
VARIANTS = {
    # ------------------------------------------------------------------ #
    "duplo": VariantConfig(
        name="duplo",
        description="Main: Qwen3-VL + multi-layer Q-Pooler + contrastive + VICReg",
        planner=PlannerConfig(kind="qpooler"),
        policy=PolicyConfig(latent_mode="sequence"),
    ),

    # ------------------------------------------------------------------ #
    "ablation_a_no_ssl": VariantConfig(
        name="ablation_a_no_ssl",
        description="Q-Pooler without contrastive/VICReg losses",
        planner=PlannerConfig(
            kind="qpooler",
            use_contrastive=False,
            use_vicreg=False,
        ),
        policy=PolicyConfig(latent_mode="sequence"),
    ),

    # ------------------------------------------------------------------ #
    "ablation_b_last_layer": VariantConfig(
        name="ablation_b_last_layer",
        description="Q-Pooler but last VLM layer only (no pyramid)",
        planner=PlannerConfig(
            kind="qpooler",
            layer_indices=(36,),   # Qwen3-VL-4B has 36 layers; take the last
        ),
        policy=PolicyConfig(latent_mode="sequence"),
    ),

    # ------------------------------------------------------------------ #
    "ablation_c_mean_pool": VariantConfig(
        name="ablation_c_mean_pool",
        description="VLM last layer + mean pool (no Q-Pooler at all)",
        planner=PlannerConfig(
            kind="vlm_mean",
            use_contrastive=False,   # no structured queries to regularize
            use_vicreg=False,
        ),
        policy=PolicyConfig(latent_mode="vector"),
    ),

    # ------------------------------------------------------------------ #
    "ablation_d_no_latent": VariantConfig(
        name="ablation_d_no_latent",
        description="No VLM latent at all — ResNet + proprio only",
        planner=PlannerConfig(kind="none"),
        policy=PolicyConfig(latent_mode="none"),
    ),

    # ------------------------------------------------------------------ #
    "baseline_clip": VariantConfig(
        name="baseline_clip",
        description="CLIP pooled embedding as used in prior work",
        planner=PlannerConfig(
            kind="clip",
            use_contrastive=False,
            use_vicreg=False,
        ),
        policy=PolicyConfig(latent_mode="vector"),
    ),

    # ------------------------------------------------------------------ #
    "baseline_dp": VariantConfig(
        name="baseline_dp",
        description="Diffusion Policy lower bound — ResNet only, no language",
        planner=PlannerConfig(kind="none"),
        policy=PolicyConfig(latent_mode="none"),
    ),
}


def get_config(name: str) -> VariantConfig:
    if name not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{name}'. Options: {list(VARIANTS.keys())}"
        )
    return VARIANTS[name]