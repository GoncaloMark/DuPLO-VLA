"""
Alternative visual-language encoders for the ablations.

All encoders return a dict shaped like the VisualTaskPlanner output:
    {
      'latent_seq':    (B, L, D)  or None,    # for sequence conditioning
      'latent':        (B, D)     or None,    # for vector conditioning / SSL
      'latent_normed': (B, D)     or None,
      'loss':          scalar tensor or None,
      'loss_dict':     dict or None,
    }

Keeping the interface identical lets the base trainer swap encoders by name
without any branching in the training loop.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# CLIP wrapper
# --------------------------------------------------------------------------- #
class CLIPPlanner(nn.Module):
    """
    Uses CLIP to produce a single pooled (image + text) embedding per sample.
    Simple, standard, and what a lot of VLM-policy prior art actually does.
    """

    def __init__(self, clip_name: str, latent_dim: int, freeze: bool = True):
        super().__init__()
        from transformers import CLIPModel, CLIPProcessor

        self.clip = CLIPModel.from_pretrained(clip_name)
        self.processor = CLIPProcessor.from_pretrained(clip_name)
        clip_dim = self.clip.config.projection_dim  # already the joint space

        if freeze:
            for p in self.clip.parameters():
                p.requires_grad = False
            self.clip.eval()

        # Concatenate image + text pooled embeddings, then project.
        self.proj = nn.Sequential(
            nn.Linear(clip_dim * 2, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def plan(self, image, instruction, episode_ids=None):
        images = [image] if not isinstance(image, list) else image
        texts  = [instruction] if not isinstance(instruction, list) else instruction

        inputs = self.processor(
            text=texts, images=images,
            return_tensors="pt", padding=True,
        ).to(next(self.clip.parameters()).device)

        with torch.no_grad():
            img_feat  = self.clip.get_image_features(pixel_values=inputs["pixel_values"])
            text_feat = self.clip.get_text_features(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )

        combined = torch.cat([img_feat, text_feat], dim=-1)
        latent = self.proj(combined)                               # (B, latent_dim)

        return {
            "latent_seq":    None,
            "latent":        latent,
            "latent_normed": F.normalize(latent, dim=-1),
            "loss":          None,
            "loss_dict":     None,
        }

    def train(self, mode=True):
        super().train(mode)
        self.clip.eval()        # always frozen-eval
        return self


# --------------------------------------------------------------------------- #
# VLM last-layer + mean pool (ablation C)
# --------------------------------------------------------------------------- #
class VLMMeanPoolPlanner(nn.Module):
    """
    Runs the VLM, takes the last hidden state, mean-pools over the sequence,
    projects to latent_dim. Isolates the value of the Q-Pooler: same VLM,
    no learned queries, no pyramid.
    """

    def __init__(self, vlm_name: str, latent_dim: int, freeze: bool = True):
        super().__init__()
        from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
        from torch.amp import autocast

        self._autocast = autocast

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            vlm_name, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.processor = Qwen3VLProcessor.from_pretrained(vlm_name)
        vlm_dim = self.vlm.config.hidden_size

        if freeze:
            for p in self.vlm.parameters():
                p.requires_grad = False
            self.vlm.eval()

        self.proj = nn.Sequential(
            nn.Linear(vlm_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, latent_dim),
            nn.LayerNorm(latent_dim),
        ).to(torch.bfloat16)

    @torch.no_grad()
    def _vlm_forward(self, images, texts):
        inputs = self.processor(
            text=[f"<|vision_start|><|image_pad|><|vision_end|>{t}" for t in texts],
            images=images, return_tensors="pt", padding=True,
        ).to(self.vlm.device)

        with self._autocast(device_type="cuda", dtype=torch.bfloat16):
            out = self.vlm(**inputs, output_hidden_states=True)

        last = out.hidden_states[-1]                        # (B, L, vlm_dim)
        mask = inputs.attention_mask.unsqueeze(-1).float()  # (B, L, 1)
        pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled                                        # (B, vlm_dim)

    def plan(self, image, instruction, episode_ids=None):
        images = [image] if not isinstance(image, list) else image
        texts  = [instruction] if not isinstance(instruction, list) else instruction

        pooled = self._vlm_forward(images, texts).to(torch.bfloat16)
        latent = self.proj(pooled).float()

        return {
            "latent_seq":    None,
            "latent":        latent,
            "latent_normed": F.normalize(latent, dim=-1),
            "loss":          None,
            "loss_dict":     None,
        }

    def train(self, mode=True):
        super().train(mode)
        self.vlm.eval()
        return self


# --------------------------------------------------------------------------- #
# Null planner (ablation D, baseline DP)
# --------------------------------------------------------------------------- #
class NoPlanner(nn.Module):
    """No-op. Used when the variant doesn't condition on a VLM latent at all."""

    def plan(self, image, instruction, episode_ids=None):
        return {
            "latent_seq":    None,
            "latent":        None,
            "latent_normed": None,
            "loss":          None,
            "loss_dict":     None,
        }
