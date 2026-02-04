from typing import Union, List, Optional, Tuple
import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from torch.amp import autocast

from .latent_encoder import LatentTaskEncoder, ReconstructionLoss

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
        num_pooling_queries: int = 8,
        num_attention_heads: int = 8,
        num_vlm_layers_to_use: int = 4,
        layer_fusion_method: str = "learned_weighted",
        use_multi_layer: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # Load VLM
        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = Qwen3VLProcessor.from_pretrained(model_name)
        
        # Get VLM hidden dimension
        vlm_hidden_dim = self.vlm.config.text_config.hidden_size
        
        # Initialize task encoder
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
        
        # Move encoder to bfloat16
        self.task_encoder = self.task_encoder.to(torch.bfloat16)
        
        # Reconstruction loss for pre-alignment
        self.reconstruction_loss = ReconstructionLoss(
            pooled_weight=1.0,
            sequence_weight=0.5,
            latent_reg_weight=0.01
        )
        
        # Freeze VLM 
        if freeze_vlm:
            for p in self.vlm.parameters():
                p.requires_grad = False
        
        self.use_multi_layer = use_multi_layer

    def extract_features_batch(
        self,
        images,
        texts,
        training = False,
        return_all_layers = True
    ):
        """
        Extract features from VLM for a batch of images and texts.
        
        Args:
            images: (B, H, W, C) or (B, C, H, W)
            texts: list[str] length B
            training: whether in training mode
            return_all_layers: whether to return all hidden states for multi-layer fusion
            
        Returns:
            last_hidden_state: (B, seq_len, hidden_dim)
            all_hidden_states: Optional tuple of all layer hidden states
        """
        # Convert to numpy if tensor
        if isinstance(images, torch.Tensor):
            images = images.cpu().numpy()
        
        # Handle different image formats
        if isinstance(images, np.ndarray):
            # Single image (H, C, W) -> (H, W, C)
            if images.ndim == 3 and images.shape[1] == 3:
                images = np.transpose(images, (0, 2, 1))
                images = [images]
            # Batch of images (B, C, H, W) -> (B, H, W, C)
            elif images.ndim == 4 and images.shape[1] == 3:
                images = np.transpose(images, (0, 2, 3, 1))
                images = [images[i] for i in range(len(images))]
            # Already in correct format
            else:
                if images.ndim == 3:
                    images = [images]
                else:
                    images = [images[i] for i in range(len(images))]
        
        # Prepare text with image tokens
        text_with_image = [
            f"<|vision_start|><|image_pad|><|vision_end|>{t}"
            for t in texts
        ]
        
        # Process inputs
        inputs = self.processor(
            text=text_with_image,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.vlm.device)
        
        # Extract features from VLM
        if training and not self.vlm.training:
            # If in training mode but VLM is frozen, still use no_grad
            with torch.no_grad():
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = self.vlm(**inputs, output_hidden_states=True)
        else:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = self.vlm(**inputs, output_hidden_states=True)
        
        last_hidden_state = outputs.hidden_states[-1]
        all_hidden_states = outputs.hidden_states if return_all_layers else None
        
        return last_hidden_state, all_hidden_states

    def plan(
        self,
        image,
        instruction,
        training = True,
        return_attention_weights = False,
        return_reconstruction_loss = False,
    ) -> dict:
        """
        Generate task-specific latent representation from image and instruction.
        
        Args:
            image: Single image (H, W, C) or batch (B, H, W, C)
            instruction: Single string or list of strings (length B)
            training: whether in training mode
            return_attention_weights: whether to return Q-Pooler attention weights
            return_reconstruction_loss: whether to compute reconstruction loss
            
        Returns:
            dict with keys:
                - latent: (B, latent_dim) or (latent_dim,) if single
                - encoder_output: full output dict from encoder
                - reconstruction_loss: Optional, if return_reconstruction_loss=True
                - reconstruction_loss_dict: Optional dict with loss components
        """
        # Handle batching
        is_batched = isinstance(instruction, list)
        
        if not is_batched:
            images = image.unsqueeze(0) if torch.is_tensor(image) else np.expand_dims(image, 0)
            instructions = [instruction]
        else:
            images = image
            instructions = instruction
        
        prompts = [
            f"""You are controlling a robotic arm that needs to complete the following instruction: {instr}
Given the image and instruction, provide a detailed plan enriched with spatial cues and object descriptions with relative positions.

Use common sense to identify objects (cups, bottles, fruits, boxes, tools, etc.).

Principles:
- Avoid blocked or surrounded objects
- Avoid objects that would cause others to topple
- Select objects matching the user prompt

Plan:"""
            for instr in instructions
        ]
        
        # Extract VLM features
        last_hidden_state, all_hidden_states = self.extract_features_batch(
            images, 
            prompts, 
            training=training,
            return_all_layers=self.use_multi_layer
        )
        
        # Encode with task encoder
        encoder_output = self.task_encoder(
            vlm_features=last_hidden_state,
            vlm_hidden_states=all_hidden_states,
            return_attention_weights=return_attention_weights
        )
        
        latent = encoder_output['latent']
        
        # Compute reconstruction loss if requested
        reconstruction_loss = None
        reconstruction_loss_dict = None
        if return_reconstruction_loss:
            reconstruction_loss, reconstruction_loss_dict = self.reconstruction_loss(
                encoder_output=encoder_output,
                vlm_features=last_hidden_state,
                pooled_flat_target=encoder_output['pooled_flat']  # Use encoder's own pooling as target
            )
        
        # Unbatch if single input
        if not is_batched:
            latent = latent[0]
            encoder_output = {k: v[0] if isinstance(v, torch.Tensor) and v.ndim > 1 else v 
                            for k, v in encoder_output.items()}
        
        return {
            'latent': latent,
            'encoder_output': encoder_output,
            'reconstruction_loss': reconstruction_loss,
            'reconstruction_loss_dict': reconstruction_loss_dict
        }
    
    def train(self, mode: bool = True):
        """
        Override train mode to ensure VLM stays in eval mode
        """
        super().train(mode)
        # Force VLM to eval, even if planner is training
        if self.vlm is not None:
            self.vlm.eval()
        return self
    
    def get_encoder_parameters(self):
        """Get parameters of the task encoder only (for selective training)"""
        return self.task_encoder.parameters()
    
    def get_vlm_parameters(self):
        """Get parameters of the VLM only"""
        return self.vlm.parameters()
    
    def freeze_vlm(self):
        """Freeze VLM parameters"""
        for p in self.vlm.parameters():
            p.requires_grad = False
    
    def unfreeze_vlm(self):
        """Unfreeze VLM parameters"""
        for p in self.vlm.parameters():
            p.requires_grad = True
    
    def freeze_encoder(self):
        """Freeze task encoder parameters"""
        for p in self.task_encoder.parameters():
            p.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze task encoder parameters"""
        for p in self.task_encoder.parameters():
            p.requires_grad = True
