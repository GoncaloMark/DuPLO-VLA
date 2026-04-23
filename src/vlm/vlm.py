import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast

from .latent_encoder import (
    LatentTaskEncoder,
    TemporalContrastiveLoss,
    vicreg_loss,
)

LATENT_VAR_REG_WEIGHT = 1.0
LATENT_COV_REG_WEIGHT = 1.0


class VisualTaskPlanner(nn.Module):
    """
    Frozen Qwen3-VL feature extractor + trainable LatentTaskEncoder.

    Call paths:
      - plan(image, instruction, episode_ids)
            Live path: run VLM, run encoder, optionally compute SSL loss.
      - plan_from_features(vlm_hidden_states, attention_mask, episode_ids)
            Offline path: skip VLM, run encoder on pre-extracted features.
      - extract_features_batch(images, texts)
            Used by the offline extractor. Returns RAW layer outputs.
    """

    def __init__(
        self,
        load_vlm: bool = True,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        freeze_vlm: bool = True,
        latent_dim: int = 512,
        vlm_dim: int = 3072,              # Qwen3-VL-4B default; overwritten if VLM is loaded
        num_sampled_layers: int = 4,      # matches the len() of layer_indices at extraction time
        q_hidden_dim: int = 768,
        num_pooling_queries: int = 16,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        contrastive_weight: float = 1.0,
        gate_output: bool = True,
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
            vlm_dim = self.vlm.config.hidden_size

        # LatentTaskEncoder lives in bfloat16 to match the VLM's dtype
        self.task_encoder = LatentTaskEncoder(
            vlm_hidden_dim=vlm_dim,
            num_layers=num_sampled_layers,
            q_hidden_dim=q_hidden_dim,
            latent_dim=latent_dim,
            num_pooling_queries=num_pooling_queries,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            gate_output=gate_output,
        ).to(torch.bfloat16)

        self.contrastive_loss = TemporalContrastiveLoss()
        self.contrastive_weight = contrastive_weight

        if freeze_vlm and self.vlm is not None:
            for p in self.vlm.parameters():
                p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Loss
    # ------------------------------------------------------------------ #
    def _compute_encoder_loss(self, out_dict: dict, episode_ids):
        latent_normed = out_dict["latent_normed"].float()
        raw_latent = out_dict["latent"].float()

        contrastive = self.contrastive_loss(latent_normed, episode_ids)
        var_loss, cov_loss = vicreg_loss(raw_latent)

        total = (
            self.contrastive_weight * contrastive
            + LATENT_VAR_REG_WEIGHT * var_loss
            + LATENT_COV_REG_WEIGHT * cov_loss
        )
        return total, {
            "contrastive": contrastive,
            "var_reg":     var_loss,
            "cov_reg":     cov_loss,
            "gate":        out_dict["gate_value"],
        }

    # ------------------------------------------------------------------ #
    # Feature extraction — RAW, no final-norm trick
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def extract_features_batch(self, images, texts):
        """
        Run the VLM and return raw hidden states for every decoder layer,
        plus the text attention mask.

        Returns:
            all_hidden: list of tensors, length (num_decoder_layers + 1).
                        Each tensor is (B, L_text, vlm_hidden_dim), bfloat16.
                        These are the RAW residual-stream outputs of each
                        transformer block — NOT passed through any final norm.
            attention_mask: (B, L_text) bool-ish tensor, 1 = valid token.
        """
        inputs = self.processor(
            text=[f"<|vision_start|><|image_pad|><|vision_end|>{t}" for t in texts],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)

        with autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.vlm(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of length (num_layers + 1).
        # Index 0 is the embedding layer output, index i is the output of
        # transformer block i. We return them raw — the Q-Pooler will
        # normalize per-layer.
        all_hidden = list(outputs.hidden_states)
        return all_hidden, inputs.attention_mask

    # ------------------------------------------------------------------ #
    # Offline path: encoder on pre-extracted features
    # ------------------------------------------------------------------ #
    def plan_from_features(
        self,
        vlm_hidden_states,
        attention_mask=None,
        episode_ids=None,
    ):
        """
        Args:
            vlm_hidden_states: either a list of per-layer tensors or a
                stacked tensor (B, num_layers, L_text, D). Must match the
                num_sampled_layers passed at init.
            attention_mask: optional (B, L_text) bool — True = valid.
            episode_ids:    optional iterable of length B for SSL loss.

        Returns:
            {'latent_seq', 'latent', 'latent_normed', 'loss', 'loss_dict'}
        """
        device = next(self.task_encoder.parameters()).device

        # Move + cast inputs
        if isinstance(vlm_hidden_states, (list, tuple)):
            vlm_hidden_states = [
                h.to(device, dtype=torch.bfloat16) for h in vlm_hidden_states
            ]
        else:
            vlm_hidden_states = vlm_hidden_states.to(device, dtype=torch.bfloat16)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        enc_out = self.task_encoder(
            vlm_hidden_states=vlm_hidden_states,
            key_padding_mask=attention_mask,
        )

        loss, loss_dict = (None, None)
        if episode_ids is not None:
            loss, loss_dict = self._compute_encoder_loss(enc_out, episode_ids)

        return {
            "latent_seq":    enc_out["latent_seq"],      # for policy
            "latent":        enc_out["latent"],          # for VICReg / logging
            "latent_normed": enc_out["latent_normed"],   # for contrastive
            "gate_value":    enc_out["gate_value"],
            "loss":          loss,
            "loss_dict":     loss_dict,
        }

    # ------------------------------------------------------------------ #
    # Live path: run VLM + encoder end to end
    # ------------------------------------------------------------------ #
    def plan(self, image, instruction, episode_ids=None,
             layer_indices=(8, 16, 24, 32)):
        """
        Live convenience path. For efficient training, precompute features
        and use plan_from_features() instead.

        Args:
            image / instruction: single or list.
            episode_ids:         optional ids for SSL loss.
            layer_indices:       which decoder layers to sample. Length MUST
                                 equal self.task_encoder.q_pooler.num_layers.
        """
        images = [image] if not isinstance(image, list) else image
        texts = [instruction] if not isinstance(instruction, list) else instruction

        all_h, mask = self.extract_features_batch(images, texts)

        if len(layer_indices) != self.task_encoder.q_pooler.num_layers:
            raise ValueError(
                f"layer_indices has length {len(layer_indices)} but the "
                f"Q-Pooler expects {self.task_encoder.q_pooler.num_layers} "
                "sampled layers. Either pass matching indices or rebuild the "
                "planner with num_sampled_layers set correctly."
            )
        sampled = [all_h[i] for i in layer_indices]

        e_ids = (
            episode_ids if episode_ids is not None
            else list(range(len(texts)))
        )
        return self.plan_from_features(sampled, mask, e_ids)

    # ------------------------------------------------------------------ #
    def train(self, mode: bool = True):
        super().train(mode)
        if self.vlm is not None:
            self.vlm.eval()   # VLM stays in eval; only encoder trains
        return self
