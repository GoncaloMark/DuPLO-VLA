from typing import Union, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from torch.amp import autocast

from .latent_encoder import HierarchicalContrastiveLoss, LatentTaskEncoder, ReconstructionLoss, SceneGoalLoss


class VisualTaskPlanner(nn.Module):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        freeze_vlm: bool = True,
        latent_dim: int = 512,
        num_pooling_queries: int = 16,
        num_attention_heads: int = 8,
        num_vlm_layers_to_use: int = 4,
        layer_fusion_method: str = "learned_weighted",
        use_multi_layer: bool = True,
        dropout: float = 0.1,
        contrastive_weight: float = 0.01,
        contrastive_temperature: float = 0.07,
    ):
        super().__init__()

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.processor = Qwen3VLProcessor.from_pretrained(model_name)

        vlm_hidden_dim = self.vlm.config.text_config.hidden_size

        self.task_encoder = LatentTaskEncoder(
            vlm_hidden_dim=vlm_hidden_dim,
            latent_dim=latent_dim,
            num_pooling_queries=num_pooling_queries,
            num_attention_heads=num_attention_heads,
            num_vlm_layers_to_use=num_vlm_layers_to_use,
            layer_fusion_method=layer_fusion_method,
            use_multi_layer=use_multi_layer,
            dropout=dropout,
        )
        self.task_encoder = self.task_encoder.to(torch.bfloat16)

        self.num_pooling_queries = num_pooling_queries
        self.vlm_hidden_dim      = vlm_hidden_dim

        self.reconstruction_loss = ReconstructionLoss(
            pooled_weight=1.0,
            sequence_weight=0.5,
            latent_reg_weight=0.01,
        )
        self.contrastive_loss = HierarchicalContrastiveLoss(
            temperature=contrastive_temperature,
            same_episode_weight=1.0,
            same_task_weight=0.5,
        )
        self.contrastive_weight = contrastive_weight

        self.goal_loss = SceneGoalLoss(weight=0.1)

        if freeze_vlm:
            for p in self.vlm.parameters():
                p.requires_grad = False

        self.use_multi_layer = use_multi_layer

    def _make_pooled_flat_target(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        B, seq_len, hidden_dim = last_hidden_state.shape
        n = self.num_pooling_queries
        indices = torch.linspace(0, seq_len - 1, n).long().to(last_hidden_state.device)
        return last_hidden_state[:, indices, :].reshape(B, n * hidden_dim).detach()

    def extract_features_batch(
        self,
        images,
        texts,
        training: bool = False,
        return_all_layers: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
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
            else:
                raise ValueError(f"Unexpected image ndim: {images.ndim}")
        elif not isinstance(images, list):
            raise TypeError(f"Unexpected images type: {type(images)}")

        text_with_image = [
            f"<|vision_start|><|image_pad|><|vision_end|>{t}" for t in texts
        ]
        inputs = self.processor(
            text=text_with_image,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)

        vlm_needs_grad = any(p.requires_grad for p in self.vlm.parameters())
        ctx = torch.enable_grad() if (training and vlm_needs_grad) else torch.no_grad()
        with ctx:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.vlm(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        all_hidden_states = outputs.hidden_states if return_all_layers else None

        return last_hidden_state, all_hidden_states

    def plan_from_features(
        self,
        vlm_hidden_states: torch.Tensor,
        vlm_seq_len: Optional[torch.Tensor],
        task_names=None,
        episode_ids=None,
        goal_hidden_states: Optional[torch.Tensor] = None,
        goal_seq_len: Optional[torch.Tensor] = None,
        return_attention_weights: bool = False,
        return_reconstruction_loss: bool = False,
    ) -> dict:
        """
        Skip the 8B VLM forward — use precomputed hidden states directly.

        Args:
            vlm_hidden_states : (B, num_layers, max_seq_len, hidden_dim)  float16
            vlm_seq_len       : (B,) int32  actual token count before zero-padding.
                                Image tokens are constant but text tokens vary
                                by ~3-14 across tasks, so seq_len varies slightly.
                                We zero-pad in precompute (zero vectors add nothing
                                to Q-Pooler output), but we still build the mask
                                to keep attention focused on real tokens.
            task_names        : list[str]  for contrastive loss
            episode_ids       : list[int]  for contrastive loss
        """
        enc_device = next(self.task_encoder.parameters()).device
        hs = vlm_hidden_states.to(dtype=torch.bfloat16, device=enc_device)

        # Last layer → reconstruction target (matches live-VLM behaviour)
        last_hidden_state = hs[:, -1]   # (B, max_seq_len, hidden_dim)

        # Build key_padding_mask: True = padding position (zero-filled, ignore)
        # Shape (B, max_seq_len) — broadcast to (B, 1, 1, max_seq_len) in QPooler
        if vlm_seq_len is not None:
            B, max_seq = last_hidden_state.shape[:2]
            positions = torch.arange(max_seq, device=enc_device).unsqueeze(0)
            key_padding_mask = positions >= vlm_seq_len.to(enc_device).unsqueeze(1)
        else:
            key_padding_mask = None

        encoder_output = self.task_encoder(
            vlm_features=last_hidden_state,
            vlm_hidden_states=hs,
            return_attention_weights=return_attention_weights,
            key_padding_mask=key_padding_mask,
        )

        latent = encoder_output['latent']

        reconstruction_loss      = None
        reconstruction_loss_dict = None

        if return_reconstruction_loss:
            pooled_flat_target = self._make_pooled_flat_target(last_hidden_state)

            reconstruction_loss, reconstruction_loss_dict = self.reconstruction_loss(
                encoder_output=encoder_output,
                vlm_features=last_hidden_state,
                pooled_flat_target=pooled_flat_target,
            )

            if task_names is not None and episode_ids is not None:
                contrastive_loss         = self.contrastive_loss(latent, task_names, episode_ids)
                reconstruction_loss      = reconstruction_loss + self.contrastive_weight * contrastive_loss
                reconstruction_loss_dict = reconstruction_loss_dict or {}
                reconstruction_loss_dict['contrastive'] = contrastive_loss.item()

            # Goal loss: push current latent toward goal-frame latent 
            # Teaches the encoder WHERE the goal is, not just WHAT task it is.
            # Without this, pick-place with goal-left and goal-right produce
            # identical latents and DP3 cannot distinguish them.
            if goal_hidden_states is not None:
                goal_plan = self.plan_from_features(
                    vlm_hidden_states=goal_hidden_states,
                    vlm_seq_len=goal_seq_len,
                    task_names=None,
                    episode_ids=None,
                    return_reconstruction_loss=False,
                )
                goal_loss_val            = self.goal_loss(latent, goal_plan['latent'])
                reconstruction_loss      = reconstruction_loss + goal_loss_val
                reconstruction_loss_dict = reconstruction_loss_dict or {}
                reconstruction_loss_dict['goal_loss'] = goal_loss_val.item()

        return {
            'latent':                   latent,
            'encoder_output':           encoder_output,
            'reconstruction_loss':      reconstruction_loss,
            'reconstruction_loss_dict': reconstruction_loss_dict,
        }

    def plan(
        self,
        image,
        instruction: Union[str, List[str]],
        task_names=None,
        episode_ids=None,
        goal_image=None,
        training: bool = True,
        return_attention_weights: bool = False,
        return_reconstruction_loss: bool = False,
    ) -> dict:
        """Live VLM path. goal_image: (B, H, W, C) final-frame images for SceneGoalLoss."""
        if not isinstance(instruction, list):
            images       = image.unsqueeze(0) if torch.is_tensor(image) else np.expand_dims(image, 0)
            instructions = [instruction]
        else:
            images       = image
            instructions = instruction

        prompts = [
            f"{instr}\n"
            f"Identify the target object, its current location, goal position, and any obstacles."
            for instr in instructions
        ]

        last_hidden_state, all_hidden_states = self.extract_features_batch(
            images, prompts, training=training, return_all_layers=self.use_multi_layer,
        )

        encoder_output = self.task_encoder(
            vlm_features=last_hidden_state,
            vlm_hidden_states=all_hidden_states,
            return_attention_weights=return_attention_weights,
        )

        latent = encoder_output['latent']

        reconstruction_loss      = None
        reconstruction_loss_dict = None

        if return_reconstruction_loss:
            pooled_flat_target = self._make_pooled_flat_target(last_hidden_state)

            reconstruction_loss, reconstruction_loss_dict = self.reconstruction_loss(
                encoder_output=encoder_output,
                vlm_features=last_hidden_state,
                pooled_flat_target=pooled_flat_target,
            )

            contrastive_loss = self.contrastive_loss(
                latent,
                task_names  if task_names  is not None else instructions,
                episode_ids if episode_ids is not None else list(range(len(instructions))),
            )
            reconstruction_loss = reconstruction_loss + self.contrastive_weight * contrastive_loss
            reconstruction_loss_dict = reconstruction_loss_dict or {}
            reconstruction_loss_dict['contrastive'] = contrastive_loss.item()

            if goal_image is not None:
                goal_last_h, goal_all_h = self.extract_features_batch(
                    goal_image, prompts, training=False, return_all_layers=self.use_multi_layer,
                )
                goal_enc_out = self.task_encoder(
                    vlm_features=goal_last_h,
                    vlm_hidden_states=goal_all_h,
                    return_attention_weights=False,
                )
                goal_loss_val = self.goal_loss(latent, goal_enc_out['latent'])
                reconstruction_loss = reconstruction_loss + goal_loss_val
                reconstruction_loss_dict['goal_loss'] = goal_loss_val.item()

        if not isinstance(instruction, list):
            latent = latent[0]
            encoder_output = {
                k: v[0] if isinstance(v, torch.Tensor) and v.ndim > 1 else v
                for k, v in encoder_output.items()
            }

        return {
            'latent': latent,
            'encoder_output': encoder_output,
            'reconstruction_loss': reconstruction_loss,
            'reconstruction_loss_dict': reconstruction_loss_dict,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vlm is not None:
            self.vlm.eval()
        return self

    def get_encoder_parameters(self): return self.task_encoder.parameters()
    def get_vlm_parameters(self): return self.vlm.parameters()
    def freeze_vlm(self):
        for p in self.vlm.parameters(): p.requires_grad = False
    def unfreeze_vlm(self):
        for p in self.vlm.parameters(): p.requires_grad = True
    def freeze_encoder(self):
        for p in self.task_encoder.parameters(): p.requires_grad = False
    def unfreeze_encoder(self):
        for p in self.task_encoder.parameters(): p.requires_grad = True
