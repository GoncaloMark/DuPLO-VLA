from typing import Union, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from torch.amp import autocast

from .latent_encoder import TemporalContrastiveLoss, LatentTaskEncoder

# Anti-collapse: penalizes dimensions with std < 1.0
# Keeps the latent space spread out even before BC gradients kick in.
LATENT_VAR_REG_WEIGHT = 1.0


class VisualTaskPlanner(nn.Module):
    def __init__(
        self,
        load_vlm: bool = True,
        model_name: str = "Qwen/Qwen3-VL-4B-Instruct",
        freeze_vlm: bool = True,
        latent_dim: int = 512,
        vlm_dim: int | None = None,
        num_pooling_queries: int = 16,
        num_attention_heads: int = 8,
        num_vlm_layers_to_use: int = 4,
        layer_fusion_method: str = "learned_weighted",
        use_multi_layer: bool = True,
        dropout: float = 0.1,
        contrastive_weight: float = 1.0,
        contrastive_temperature: float = 0.1,
    ):
        super().__init__()

        self.vlm = None

        print(load_vlm)

        if load_vlm:
            self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto",
            )
            self.processor = Qwen3VLProcessor.from_pretrained(model_name)

        if self.vlm is not None:
            vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        else:
            vlm_hidden_dim = vlm_dim

        self.task_encoder = LatentTaskEncoder(
            vlm_hidden_dim=vlm_hidden_dim,
            latent_dim=latent_dim,
            num_pooling_queries=num_pooling_queries,
            num_attention_heads=num_attention_heads,
            num_vlm_layers_to_use=num_vlm_layers_to_use,
            layer_fusion_method=layer_fusion_method,
            use_multi_layer=use_multi_layer,
            dropout=dropout,
        ).to(torch.bfloat16)

        self.num_pooling_queries = num_pooling_queries
        self.vlm_hidden_dim = vlm_hidden_dim
        self.use_multi_layer = use_multi_layer
        self.contrastive_weight = contrastive_weight

        self.contrastive_loss = TemporalContrastiveLoss(
            temperature=contrastive_temperature,
        )

        if freeze_vlm and load_vlm:
            assert self.vlm
            for p in self.vlm.parameters():
                p.requires_grad = False

    def _compute_encoder_loss(self, latent_normed, raw_latent, episode_ids):
        """
        Encoder-specific losses (contrastive + variance regularization).

        Args:
            latent_normed: (B, D) L2-normalized latents for contrastive loss
            raw_latent:    (B, D) unnormalized latents for variance regularization
            episode_ids:   list[int] episode IDs for contrastive pairing
        """
        latent_normed_f32 = latent_normed.float()
        raw_latent_f32 = raw_latent.float()

        contrastive = self.contrastive_loss(latent_normed_f32, episode_ids)

        # Variance regularization: push per-dimension std toward >= 1.0
        std_per_dim = torch.sqrt(raw_latent_f32.var(dim=0) + 1e-8)
        var_reg = F.relu(1.0 - std_per_dim).mean()

        loss = self.contrastive_weight * contrastive + LATENT_VAR_REG_WEIGHT * var_reg
        return loss, {
            'contrastive': contrastive.item(),
            'var_reg': var_reg.item(),
        }

    def extract_features_batch(self, images, texts, training=False, return_all_layers=True):
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                if images.shape[0] == 3:
                    images = np.transpose(images, (1, 2, 0))
                images = [images]
            elif images.ndim == 4:
                if images.shape[1] == 3:
                    images = np.transpose(images, (0, 2, 3, 1))
                images = [images[i] for i in range(len(images))]
        elif not isinstance(images, list):
            raise TypeError(f"Unexpected images type: {type(images)}")

        inputs = self.processor(
            text=[f"<|vision_start|><|image_pad|><|vision_end|>{t}" for t in texts],
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)

        vlm_needs_grad = any(p.requires_grad for p in self.vlm.parameters())
        ctx = torch.enable_grad() if (training and vlm_needs_grad) else torch.no_grad()
        with ctx, autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.vlm(**inputs, output_hidden_states=True)

        # Apply the VLM's final RMSNorm to each hidden state layer.
        # This matches the precomputed dataset which stores post-RMSNorm values.
        # Without this, the live path would feed [-2000,8000] to the encoder
        # while the precomputed path feeds [-5,5] — a distribution mismatch.
        #
        # Only normalize layers we actually use (last N) to avoid wasting compute
        # on 36+ layers when we only need the last 2.
        final_norm = self.vlm.model.language_model.norm
        last_hidden = final_norm(outputs.hidden_states[-1])

        all_hidden = None
        if return_all_layers:
            all_hidden = tuple(
                final_norm(h) for h in outputs.hidden_states[-self.task_encoder.feature_extractor.num_layers_to_use:]
            ) if self.use_multi_layer else tuple(
                final_norm(h) for h in outputs.hidden_states
            )

        return last_hidden, all_hidden

    def _encode_hidden_states(self, hs_tensor, seq_len_tensor=None):
        """Shared encode path for precomputed hidden states → latent."""
        enc_device = next(self.task_encoder.parameters()).device
        hs = hs_tensor.to(dtype=torch.bfloat16, device=enc_device)
        last_hidden = hs[:, -1]

        key_padding_mask = None
        if seq_len_tensor is not None:
            _, max_seq = last_hidden.shape[:2]
            pos = torch.arange(max_seq, device=enc_device).unsqueeze(0)
            key_padding_mask = pos >= seq_len_tensor.to(enc_device).unsqueeze(1)

        return self.task_encoder(
            vlm_features=last_hidden,
            vlm_hidden_states=hs,
            key_padding_mask=key_padding_mask,
        ), last_hidden

    def plan_from_features(
        self,
        vlm_hidden_states: torch.Tensor,
        vlm_seq_len: Optional[torch.Tensor],
        episode_ids=None,
        return_encoder_loss: bool = False,
    ) -> dict:
        enc_out, _ = self._encode_hidden_states(vlm_hidden_states, vlm_seq_len)
        latent = enc_out['latent']

        encoder_loss = None
        encoder_loss_dict = None
        if return_encoder_loss and episode_ids is not None:
            encoder_loss, encoder_loss_dict = self._compute_encoder_loss(
                enc_out['latent_normed'], latent, episode_ids,
            )

        return {
            'latent': latent,
            'encoder_output': enc_out,
            'encoder_loss': encoder_loss,
            'encoder_loss_dict': encoder_loss_dict,
        }

    def plan(
        self,
        image,
        instruction: Union[str, List[str]],
        episode_ids=None,
        training: bool = True,
        return_attention_weights: bool = False,
        return_encoder_loss: bool = False,
    ) -> dict:
        """Live VLM path."""
        if not isinstance(instruction, list):
            images = image.unsqueeze(0) if torch.is_tensor(image) else np.expand_dims(image, 0)
            instructions = [instruction]
        else:
            images = image
            instructions = instruction

        prompts = [
            f"{instr}\nIdentify the target object, its current location, goal position, and any obstacles."
            for instr in instructions
        ]

        last_h, all_h = self.extract_features_batch(
            images, prompts, training=training, return_all_layers=self.use_multi_layer
        )
        enc_out = self.task_encoder(
            vlm_features=last_h,
            vlm_hidden_states=all_h,
            return_attention_weights=return_attention_weights,
        )
        latent = enc_out['latent']

        encoder_loss = None
        encoder_loss_dict = None
        if return_encoder_loss:
            ei = episode_ids if episode_ids is not None else list(range(len(instructions)))
            encoder_loss, encoder_loss_dict = self._compute_encoder_loss(
                enc_out['latent_normed'], latent, ei,
            )

        if not isinstance(instruction, list):
            latent = latent[0]
            enc_out = {k: v[0] if isinstance(v, torch.Tensor) and v.ndim > 1 else v
                       for k, v in enc_out.items()}

        return {
            'latent': latent,
            'encoder_output': enc_out,
            'encoder_loss': encoder_loss,
            'encoder_loss_dict': encoder_loss_dict,
        }

    def train(self, mode=True):
        super().train(mode)
        if self.vlm is not None:
            self.vlm.eval()
        return self

    def get_encoder_parameters(self): return self.task_encoder.parameters()
    def get_vlm_parameters(self): return self.vlm.parameters() or None
    def freeze_vlm(self):
        if self.vlm:
            for p in self.vlm.parameters(): p.requires_grad = False
    def unfreeze_vlm(self):
        if self.vlm:
            for p in self.vlm.parameters(): p.requires_grad = True
