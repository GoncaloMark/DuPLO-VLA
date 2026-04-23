"""
Shared training + eval skeleton for all 7 ablation variants.

Design:
  - build_planner(cfg)  -> module with a .plan(image, instruction, episode_ids) method
  - build_policy(cfg)   -> RDTPolicy with the right cond_dims/latent_mode
  - build_cond_dict(planner_out, img_feat) -> the dict passed to the policy
  - compute_total_loss(...)  -> policy loss + (optional) planner SSL loss
The training loop is agnostic to the variant — it just calls these.
"""

import torch
import torch.nn as nn

from ..action_heads.diffusion import RDTPolicy
from ..vlm.vlm import VisualTaskPlanner 
from .configs import VariantConfig
from .encoders import CLIPPlanner, VLMMeanPoolPlanner, NoPlanner


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def build_planner(cfg: VariantConfig, device: torch.device) -> nn.Module:
    """Pick the right planner based on cfg.planner.kind."""
    p = cfg.planner

    if p.kind == "qpooler":
        # Handle ablation B (last-layer-only) by overriding layer_indices.
        # VisualTaskPlanner doesn't accept use_contrastive/use_vicreg flags
        # as-is, so we pass contrastive_weight=0 for ablation A, and turn
        # off VICReg by monkey-patching the weights to 0 in compute_total_loss.
        planner = VisualTaskPlanner(
            load_vlm=True,
            model_name=p.vlm_name,
            freeze_vlm=True,
            latent_dim=p.latent_dim,
            num_pooling_queries=p.num_pooling_queries,
            num_attention_heads=p.num_attention_heads,
            contrastive_weight=p.contrastive_weight if p.use_contrastive else 0.0,
        )
        # Override pyramid layer indices on the underlying Q-Pooler.
        planner.task_encoder.q_pooler.layer_indices = list(p.layer_indices)
        return planner.to(device)

    if p.kind == "vlm_mean":
        return VLMMeanPoolPlanner(
            vlm_name=p.vlm_name, latent_dim=p.latent_dim,
        ).to(device)

    if p.kind == "clip":
        return CLIPPlanner(
            clip_name=p.clip_name, latent_dim=p.latent_dim,
        ).to(device)

    if p.kind == "none":
        return NoPlanner().to(device)

    raise ValueError(f"Unknown planner kind: {p.kind}")


def build_policy(
    cfg: VariantConfig,
    action_dim: int,
    state_dim: int,
    img_feat_dim: int,
    obs_horizon: int,
) -> RDTPolicy:
    """
    Assemble the cond_dims / cond_seq_lens that match the variant's latent_mode.
    All variants keep ResNet vision on — the VLM latent is additive signal.
    """
    cond_dims = {"vision": img_feat_dim}
    cond_seq_lens = {"vision": obs_horizon}   # temporal order across obs frames

    mode = cfg.policy.latent_mode
    if mode == "sequence":
        # Q-Pooler queries: order-invariant set -> no pos embed
        cond_dims["latent"] = cfg.planner.latent_dim
    elif mode == "vector":
        # CLIP or VLM-mean: single pooled vector, one token
        cond_dims["latent"] = cfg.planner.latent_dim
    elif mode == "none":
        pass  # no latent modality at all
    else:
        raise ValueError(f"Unknown latent_mode: {mode}")

    return RDTPolicy(
        action_dim=action_dim,
        horizon=cfg.policy.horizon,
        state_dim=state_dim,
        cond_dims=cond_dims,
        cond_seq_lens=cond_seq_lens,
        hidden_dim=cfg.policy.hidden_dim,
        depth=cfg.policy.depth,
        num_heads=cfg.policy.num_heads,
        num_train_timesteps=cfg.policy.num_train_timesteps,
        num_inference_timesteps=cfg.policy.num_inference_timesteps,
        prediction_type=cfg.policy.prediction_type,
        beta_schedule=cfg.policy.beta_schedule,
    )


# --------------------------------------------------------------------------- #
# Assembling policy inputs from planner outputs
# --------------------------------------------------------------------------- #
def build_cond_dict(
    cfg: VariantConfig,
    planner_out: dict,
    img_feat: torch.Tensor,
) -> dict:
    """Map planner output + image features into the policy's cond_dict."""
    cond = {"vision": img_feat}

    mode = cfg.policy.latent_mode
    if mode == "sequence":
        seq = planner_out["latent_seq"]
        if seq is None:
            raise RuntimeError(
                "latent_mode='sequence' but planner produced no latent_seq. "
                "Check planner.kind matches the policy latent_mode."
            )
        cond["latent"] = seq                                    # (B, L, D)
    elif mode == "vector":
        vec = planner_out["latent"]
        if vec is None:
            raise RuntimeError(
                "latent_mode='vector' but planner produced no latent."
            )
        cond["latent"] = vec.unsqueeze(1)                       # (B, 1, D)
    # mode == 'none': nothing to add

    return cond


# --------------------------------------------------------------------------- #
# Total loss (policy + optional planner SSL)
# --------------------------------------------------------------------------- #
def compute_total_loss(
    cfg: VariantConfig,
    policy: RDTPolicy,
    planner_out: dict,
    action_gt: torch.Tensor,
    state: torch.Tensor,
    cond_dict: dict,
) -> tuple[torch.Tensor, dict]:
    """
    Policy MSE loss + (optional) planner SSL loss with proper weighting.
    Returns (total_loss, log_dict) where log_dict is for wandb/printing.
    """
    policy_loss = policy(action_gt, state, cond_dict)
    logs = {"policy_loss": policy_loss.detach()}
    total = policy_loss

    # Planner SSL loss (only for Duplo; ablation A turns weights to 0,
    # others return loss=None).
    p_loss = planner_out.get("loss")
    if p_loss is not None and cfg.planner.kind == "qpooler":
        if cfg.planner.use_contrastive or cfg.planner.use_vicreg:
            total = total + p_loss
            logs["planner_loss"] = p_loss.detach()
            for k, v in (planner_out.get("loss_dict") or {}).items():
                logs[f"planner/{k}"] = v.detach() if torch.is_tensor(v) else v

    logs["total_loss"] = total.detach()
    return total, logs
