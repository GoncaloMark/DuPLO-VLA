from typing import Union, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from torch.amp import autocast

from .latent_encoder import HierarchicalContrastiveLoss, LatentTaskEncoder, ReconstructionLoss

class VisualTaskPlanner(nn.Module):
    """
    Visual task planner that uses a VLM to extract visual-semantic features
    and a Q-Pooler based encoder to create task-specific latent representations.
    """
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
            device_map="auto"
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

        # pooled_weight targets evenly-spaced VLM sequence positions (external target).
        # sequence_weight targets mean-pooled VLM hidden states (also external).
        # Neither target passes through the task encoder == no circularity.
        self.reconstruction_loss = ReconstructionLoss(
            pooled_weight=1.0,
            sequence_weight=0.5,
            latent_reg_weight=0.01
        )

        self.contrastive_loss = HierarchicalContrastiveLoss(
            temperature=contrastive_temperature,
            same_episode_weight=1.0,
            same_task_weight=0.5,
        )
        self.contrastive_weight = contrastive_weight

        if freeze_vlm:
            for p in self.vlm.parameters():
                p.requires_grad = False

        self.use_multi_layer = use_multi_layer

    def _make_pooled_flat_target(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        """
        Build a reconstruction target for the pooled features that does NOT
        go through the task encoder (fixes the circular target bug).

        Strategy: evenly sample num_pooling_queries token positions from the
        VLM last-layer hidden states and flatten them. These positions tend to
        cover both vision tokens and language tokens.

        Args:
            last_hidden_state: (B, seq_len, hidden_dim)
        Returns:
            target: (B, num_pooling_queries * hidden_dim)
        """
        B, seq_len, hidden_dim = last_hidden_state.shape
        n = self.num_pooling_queries

        # Evenly spaced indices, clamped to valid range
        indices = torch.linspace(0, seq_len - 1, n).long().to(last_hidden_state.device)
        target  = last_hidden_state[:, indices, :]              # (B, n, hidden_dim)
        return target.reshape(B, n * hidden_dim).detach()       # stop-gradient

    def extract_features_batch(
        self,
        images,
        texts,
        training: bool = False,
        return_all_layers: bool = True,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        """
        Extract features from VLM for a batch of images and texts.

        Args:
            images:           (B, H, W, C) numpy array or torch tensor
            texts:            list[str] length B
            training:         whether in training mode (affects grad ctx)
            return_all_layers: whether to return all hidden states

        Returns:
            last_hidden_state: (B, seq_len, hidden_dim)
            all_hidden_states: tuple of layer tensors, or None
        """
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()

        if isinstance(images, np.ndarray):
            # Normalise to (B, H, W, C)
            if images.ndim == 3:
                # Single image — could be (H, W, C) or (C, H, W)
                if images.shape[0] == 3:          # (C, H, W)
                    images = np.transpose(images, (1, 2, 0))
                images = [images]
            elif images.ndim == 4:
                # Batch: (B, C, H, W) → (B, H, W, C)  or already (B, H, W, C)
                if images.shape[1] == 3:
                    images = np.transpose(images, (0, 2, 3, 1))
                images = [images[i] for i in range(len(images))]
            else:
                raise ValueError(f"Unexpected image ndim: {images.ndim}")
        elif not isinstance(images, list):
            raise TypeError(f"Unexpected images type: {type(images)}")

        text_with_image = [
            f"<|vision_start|><|image_pad|><|vision_end|>{t}"
            for t in texts
        ]

        inputs = self.processor(
            text=text_with_image,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.vlm.device)

        # VLM forward 
        vlm_needs_grad = any(p.requires_grad for p in self.vlm.parameters())
        ctx = torch.enable_grad() if (training and vlm_needs_grad) else torch.no_grad()
        with ctx:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.vlm(**inputs, output_hidden_states=True)

        last_hidden_state = outputs.hidden_states[-1]
        all_hidden_states = outputs.hidden_states if return_all_layers else None

        return last_hidden_state, all_hidden_states

    def plan(
        self,
        image,
        instruction: Union[str, List[str]],
        task_names=None,
        episode_ids = None,
        training: bool = True,
        return_attention_weights: bool = False,
        return_reconstruction_loss: bool = False,
    ) -> dict:
        """
        Generate task-specific latent from image + instruction.

        Args:
            image:                       (H, W, C) or (B, H, W, C)
            instruction:                 str or list[str]
            training:                    whether in training mode
            return_attention_weights:    return Q-Pooler attn weights
            return_reconstruction_loss:  compute reconstruction + contrastive loss

        Returns dict with keys:
            latent               (B, latent_dim) or (latent_dim,) if unbatched
            encoder_output       full encoder output dict
            reconstruction_loss  Optional scalar
            reconstruction_loss_dict  Optional dict
        """
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
            images,
            prompts,
            training=training,
            return_all_layers=self.use_multi_layer,
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

            reconstruction_loss      = reconstruction_loss + self.contrastive_weight * contrastive_loss
            reconstruction_loss_dict = reconstruction_loss_dict or {}
            reconstruction_loss_dict['contrastive'] = contrastive_loss.item()

        # Unbatch for single-item calls
        if not isinstance(instruction, list):
            latent = latent[0]
            encoder_output = {
                k: v[0] if isinstance(v, torch.Tensor) and v.ndim > 1 else v
                for k, v in encoder_output.items()
            }

        return {
            'latent':                    latent,
            'encoder_output':            encoder_output,
            'reconstruction_loss':       reconstruction_loss,
            'reconstruction_loss_dict':  reconstruction_loss_dict,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        if self.vlm is not None:
            self.vlm.eval()   # VLM always stays in eval (BN/dropout stability)
        return self

    def get_encoder_parameters(self):
        return self.task_encoder.parameters()

    def get_vlm_parameters(self):
        return self.vlm.parameters()

    def freeze_vlm(self):
        for p in self.vlm.parameters():
            p.requires_grad = False

    def unfreeze_vlm(self):
        for p in self.vlm.parameters():
            p.requires_grad = True

    def freeze_encoder(self):
        for p in self.task_encoder.parameters():
            p.requires_grad = False

    def unfreeze_encoder(self):
        for p in self.task_encoder.parameters():
            p.requires_grad = True
