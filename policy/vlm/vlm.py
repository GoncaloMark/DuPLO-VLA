from typing import Union, List

import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
from torch.amp import autocast


from vlm.latent_encoder import LatentTaskEncoder

class VisualTaskPlanner(nn.Module): 
    def __init__(
        self,
        model_name="Qwen/Qwen3-VL-8B-Instruct",
        freeze_vlm=False,
        latent_dim=512,
    ):
        super().__init__()

        self.vlm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        self.processor = Qwen3VLProcessor.from_pretrained(model_name)

        vlm_hidden_dim=self.vlm.config.text_config.hidden_size
        self.task_encoder = LatentTaskEncoder(vlm_hidden_dim, latent_dim)

        if freeze_vlm:
            for p in self.vlm.parameters():
                p.requires_grad = False

    def extract_features_batch(
        self,
        images: torch.Tensor,
        texts: List[str],
        training: bool
    ) -> torch.Tensor:
        """
        images: (B, H, W, C)
        texts:  list[str] length B
        returns: (B, seq_len, hidden_dim)
        """

        text_with_image = [
            f"<|vision_start|><|image_pad|><|vision_end|>{t}"
            for t in texts
        ]

        inputs = self.processor(
            text=text_with_image,
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.vlm.device)

        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            with autocast(dtype=torch.bfloat16):
                out = self.vlm(**inputs)

        return out.last_hidden_state

    def generate_text(self, images, prompt, max_tokens=256):
        if isinstance(images, np.ndarray):
            images = [images]

        inputs = self.processor(
            text=[prompt],
            images=images,
            return_tensors="pt"
        ).to(self.vlm.device)

        ids = self.vlm.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False
        )

        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]

    def plan(
        self,
        image: Union[torch.Tensor, np.ndarray],
        instruction: Union[str, List[str]],
        training: bool = True,
        get_text: bool = False
    ):
        """
        image:
            - single: (H, W, C)
            - batch:  (B, H, W, C)

        instruction:
            - single: str
            - batch:  List[str] length B
        """

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

        hidden = self.extract_features_batch(images, prompts, training)
        # hidden: (B, seq_len, hidden_dim)

        latent_task = self.task_encoder(hidden)
        # latent_task: (B, latent_dim)

        plan_text = None
        if get_text:
            plan_text = [
                self.generate_text(images[i], prompts[i])
                for i in range(len(prompts))
            ]

        if not is_batched:
            latent_task = latent_task[0]
            if plan_text is not None:
                plan_text = plan_text[0]

        return latent_task, plan_text

