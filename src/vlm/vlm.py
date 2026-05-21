"""
Visual task planner: frozen Qwen3-VL + trainable LatentTaskEncoder.

Changes vs. previous version:
  * `extract_features_batch` now takes a list of MAIN images AND a list
    of WRIST images. Each sample's prompt has two `<|image_pad|>`
    placeholders; the processor expands them with both views' vision
    tokens. The VLM cross-attends over both views in a single forward
    pass, and the returned hidden states cover
        [main_vision_tokens, wrist_vision_tokens, text_tokens].
  * `plan_from_features` and `plan` pass through to the new encoder
    which understands optional temporal axes — see latent_encoder.py.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .latent_encoder import LatentTaskEncoder


class VisualTaskPlanner(nn.Module):

    def __init__(
        self,
        load_vlm: bool = True,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        freeze_vlm: bool = True,
        latent_dim: int = 512,
        vlm_dim: int = 3072,                # overwritten if VLM is loaded
        num_sampled_layers: int = 4,
        q_hidden_dim: int = 768,
        num_pooling_queries: int = 64,
        num_attention_heads: int = 8,
        max_obs_horizon: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vlm = None
        self.processor = None
        self.model_name = model_name

        if load_vlm:
            from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
            self.processor = Qwen3VLProcessor.from_pretrained(model_name)
            vlm_dim = self.vlm.config.text_config.hidden_size

        self.task_encoder = LatentTaskEncoder(
            vlm_hidden_dim=vlm_dim,
            num_layers=num_sampled_layers,
            q_hidden_dim=q_hidden_dim,
            latent_dim=latent_dim,
            num_pooling_queries=num_pooling_queries,
            num_attention_heads=num_attention_heads,
            max_obs_horizon=max_obs_horizon,
            dropout=dropout,
        ).to(torch.bfloat16)

        if freeze_vlm and self.vlm is not None:
            for p in self.vlm.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Feature extraction with TWO cameras per sample.
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_features_batch(self, main_images, wrist_images, texts):
        """
        Args:
            main_images:  list of length B, each (H, W, 3) float32 in [0, 255]
            wrist_images: list of length B, each (H, W, 3) float32 in [0, 255]
            texts:        list of length B of instruction strings

        Returns:
            all_hidden: list of length (num_decoder_layers + 1) of tensors,
                        each (B, L_total, vlm_hidden_dim), where L_total =
                        main_vision_tokens + wrist_vision_tokens + text_tokens
                        + padding. Stored as bfloat16.
            attention_mask: (B, L_total) — 1 for valid positions, 0 for pad.
        """
        assert len(main_images) == len(wrist_images) == len(texts), (
            "main_images, wrist_images and texts must all have length B"
        )

        # Two image placeholders per prompt.
        prompts = [
            f"<|vision_start|><|image_pad|><|vision_end|>"
            f"<|vision_start|><|image_pad|><|vision_end|>{t}"
            for t in texts
        ]

        # Qwen3VLProcessor accepts a flat list of images in the order they
        # appear in the prompts. With 2 placeholders per prompt this means
        # we interleave main/wrist per sample.
        images_flat = []
        for m, w in zip(main_images, wrist_images):
            images_flat.append(m)
            images_flat.append(w)

        inputs = self.processor(
            text=prompts,
            images=images_flat,
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.vlm(**inputs, output_hidden_states=True)

        all_hidden = list(outputs.hidden_states)
        return all_hidden, inputs.attention_mask

    # ------------------------------------------------------------------ #
    # Offline path: encoder runs on pre-extracted features.
    # ------------------------------------------------------------------ #
    def plan_from_features(self, vlm_hidden_states, attention_mask=None):
        """
        Args:
            vlm_hidden_states: tensor of shape
                (B, num_layers, L, D) or (B, T, num_layers, L, D), bfloat16
            attention_mask: optional, shape (B, L) or (B, T, L)
        Returns:
            dict with 'latent_seq', 'latent', 'latent_normed', 'pooled'
        """
        device = next(self.task_encoder.parameters()).device

        vlm_hidden_states = vlm_hidden_states.to(device, dtype=torch.bfloat16)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        return self.task_encoder(
            vlm_hidden_states=vlm_hidden_states,
            key_padding_mask=attention_mask,
        )

    # ------------------------------------------------------------------ #
    # Live path: VLM forward + encoder. Used at eval time.
    # ------------------------------------------------------------------ #
    def plan(self, main_image, wrist_image, instruction,
             layer_indices=(4, 12, 25, 34)):
        """
        Args:
            main_image:  one (H, W, 3) array OR a list of them.
            wrist_image: one (H, W, 3) array OR a list of them.
            instruction: one string OR a list of strings, same length.
        """
        if not isinstance(main_image, list):
            main_image = [main_image]
            wrist_image = [wrist_image]
        if isinstance(instruction, str):
            instruction = [instruction]

        all_h, mask = self.extract_features_batch(main_image, wrist_image, instruction)

        if len(layer_indices) != self.task_encoder.num_layers:
            raise ValueError(
                f"layer_indices has length {len(layer_indices)} but the encoder "
                f"was built with num_sampled_layers={self.task_encoder.num_layers}."
            )

        sampled = torch.stack([all_h[i] for i in layer_indices], dim=1)
        # (B, num_layers, L, D) — single frame; encoder handles 4D natively.
        return self.plan_from_features(sampled, mask)

    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        if self.vlm is not None:
            self.vlm.eval()
        return self

    # ------------------------------------------------------------------ #
    # Checkpoint loading — same API as before.
    # ------------------------------------------------------------------ #
    def load_encoder_weights(self, source, key: str = "ema_encoder",
                             strict: bool = True, map_location: str = "cpu"):
        from pathlib import Path

        if isinstance(source, (str, Path)):
            obj = torch.load(str(source), map_location=map_location)
        else:
            obj = source

        if isinstance(obj, dict) and key in obj:
            state_dict = obj[key]
        elif isinstance(obj, dict) and all(
            isinstance(v, torch.Tensor) for v in obj.values()
        ):
            state_dict = obj
        elif isinstance(obj, dict):
            available = [k for k in obj.keys() if not k.startswith("_")]
            raise KeyError(
                f"Checkpoint has no key '{key}'. Available: {available}."
            )
        else:
            raise TypeError(f"Unsupported source type: {type(obj)}")

        target_dtype = next(self.task_encoder.parameters()).dtype
        target_device = next(self.task_encoder.parameters()).device
        state_dict = {
            k: v.to(device=target_device, dtype=target_dtype)
            if v.is_floating_point() else v.to(device=target_device)
            for k, v in state_dict.items()
        }

        missing, unexpected = self.task_encoder.load_state_dict(state_dict, strict=strict)
        if missing or unexpected:
            print(f"[load_encoder_weights] missing: {missing}")
            print(f"[load_encoder_weights] unexpected: {unexpected}")
        else:
            print(f"[load_encoder_weights] loaded {len(state_dict)} tensors "
                  f"from key='{key}'")
        return missing, unexpected

    @classmethod
    def from_checkpoint(cls, checkpoint_path, key: str = "ema_encoder",
                        load_vlm: bool = True, strict: bool = True, **kwargs):
        planner = cls(load_vlm=load_vlm, **kwargs)
        planner.load_encoder_weights(checkpoint_path, key=key, strict=strict)
        return planner
