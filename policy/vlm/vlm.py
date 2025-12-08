from typing import Union, List

import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor

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

        vlm_hidden_dim = self.vlm.config.hidden_size
        self.task_encoder = LatentTaskEncoder(vlm_hidden_dim, latent_dim)

        if freeze_vlm:
            for p in self.vlm.parameters():
                p.requires_grad = False

    def extract_features(
        self,
        images: Union[torch.Tensor, List[np.ndarray]],
        text: str
    ) -> torch.Tensor:
        if isinstance(images, torch.Tensor):
            images = images.permute(1,2,0).cpu().numpy()
        if isinstance(images, np.ndarray):
            images = [images]

        inputs = self.processor(
            text=[text],
            images=images,
            return_tensors="pt",
            padding=True
        ).to(self.vlm.device)

        with torch.no_grad():
            out = self.vlm(**inputs, output_hidden_states=True)

        multimodal_hidden = out.hidden_states[-1]  # (1, seq_len, hidden_dim)

        return multimodal_hidden

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

    def plan(self, image, instruction, get_text=False):
        prompt = f"""You are controlling a robotic arm that needs to complete the following instruction: {instruction}
Given the image and instruction, provide a detailed plan enriched with spatial cues and object descriptions with relative positions.

Use common sense to identify objects (cups, bottles, fruits, boxes, tools, etc.).

Principles:
- Avoid blocked or surrounded objects
- Avoid objects that would cause others to topple
- Select objects matching the user prompt

Plan:"""

        hidden = self.extract_features(image, prompt)
        latent_task = self.task_encoder(hidden)

        plan_text = None
        if get_text:
            plan_text = self.generate_text(image, prompt)

        return latent_task, plan_text
