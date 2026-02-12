# Copyright (c) [2025] [FastVideo Team]
# Copyright (c) [2025] [ByteDance Ltd. and/or its affiliates.]
# SPDX-License-Identifier: [Apache License 2.0]
#
# This file has been modified by [ByteDance Ltd. and/or its affiliates.] in 2025.
#
# Original file was released under [Apache License 2.0], with the full license text
# available at [https://github.com/hao-ai-lab/FastVideo/blob/main/LICENSE].
#
# This modified file is released under the same license.

import torch
from torch.utils.data import Dataset
import json
import os
import random
import logging
from pathlib import Path
from PIL import Image
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class FluxFailLatentDataset(Dataset):
    """
    FLUX dataset for FAIL training with teacher image loading capability.
    Extends the standard FLUX dataset to include teacher images for imitation learning.
    """

    def __init__(
        self,
        json_path,
        num_latent_t,
        cfg_rate,
        teacher_image_root=None,
        rollout_number=8,
        teacher_number=None,
        target_image_size=1024,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        """
        Initialize FAIL dataset for FLUX.

        Args:
            json_path: Path to the JSON file containing prompt embeddings metadata
            num_latent_t: Number of latent timesteps (unused but kept for compatibility)
            cfg_rate: Rate for classifier-free guidance during training
            teacher_image_root: Root directory containing teacher images organized by UUID
            rollout_number: Number of rollout samples per prompt (kept for compatibility)
            teacher_number: Number of teacher images to sample per prompt (defaults to rollout_number if None)
            target_image_size: Target size for teacher image preprocessing
            negative_prompt_embeds: Pre-computed negative prompt embeddings for CFG
            negative_pooled_prompt_embeds: Pre-computed negative pooled prompt embeddings for CFG
        """
        # Standard FLUX dataset initialization
        self.json_path = json_path
        self.cfg_rate = cfg_rate
        self.datase_dir_path = os.path.dirname(json_path)

        # FLUX embedding directories
        self.prompt_embed_dir = os.path.join(self.datase_dir_path, "prompt_embed")
        self.pooled_prompt_embeds_dir = os.path.join(
            self.datase_dir_path, "pooled_prompt_embeds"
        )
        self.text_ids_dir = os.path.join(
            self.datase_dir_path, "text_ids"
        )

        # GAIL-specific parameters
        self.teacher_image_root = teacher_image_root
        self.rollout_number = rollout_number
        # teacher_number defaults to rollout_number for backward compatibility
        self.teacher_number = teacher_number if teacher_number is not None else rollout_number
        assert self.teacher_number > 0, "teacher_number must be > 0"
        self.target_image_size = target_image_size

        # Load dataset annotations
        with open(self.json_path, "r") as f:
            self.data_anno = json.load(f)

        self.num_latent_t = num_latent_t

        # Store negative prompt embeddings for CFG
        # If not provided, use zero embeddings as fallback
        if negative_prompt_embeds is not None:
            self.negative_prompt_embeds = negative_prompt_embeds
        else:
            # FLUX unconditional embeddings (zero embeddings [256, 4096])
            self.negative_prompt_embeds = torch.zeros(256, 4096).to(torch.float32)

        if negative_pooled_prompt_embeds is not None:
            self.negative_pooled_prompt_embeds = negative_pooled_prompt_embeds
        else:
            # Zero pooled embeddings as fallback [768]
            self.negative_pooled_prompt_embeds = torch.zeros(768).to(torch.float32)

        # Image transforms for teacher images
        self.image_transforms = transforms.Compose([
            transforms.Resize((self.target_image_size, self.target_image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        self.lengths = [
            data_item["length"] if "length" in data_item else 1
            for data_item in self.data_anno
        ]

    def _load_teacher_images(self, sample_uuid: str) -> torch.Tensor:
        """
        Load teacher images for a given sample UUID.

        Args:
            sample_uuid: UUID of the sample to load images for

        Returns:
            Tensor of shape (teacher_number, 3, H, W) containing processed teacher images
        """
        if self.teacher_image_root is None:
            logger.warning("teacher_image_root not specified, using random noise")
            return torch.randn(self.teacher_number, 3, self.target_image_size, self.target_image_size)

        image_dir = Path(self.teacher_image_root) / sample_uuid
        if not image_dir.exists():
            logger.warning(f"Teacher image directory {image_dir} not found for UUID {sample_uuid}")
            # Return random noise as fallback
            return torch.randn(self.teacher_number, 3, self.target_image_size, self.target_image_size)

        # Get all PNG files in the directory
        image_files = list(image_dir.glob("*.png"))
        if len(image_files) == 0:
            logger.warning(f"No PNG images found in {image_dir}")
            return torch.randn(self.teacher_number, 3, self.target_image_size, self.target_image_size)

        # Sample k different images first, use random choice only if total images < teacher_number
        if len(image_files) >= self.teacher_number:
            # Sample without replacement when we have enough images
            selected_images = random.sample(image_files, k=self.teacher_number)
        else:
            # Use all available images, then randomly choose with replacement to reach target
            selected_images = random.choices(image_files, k=self.teacher_number)

        processed_images = []
        for img_path in selected_images:
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    processed_img = self.image_transforms(img)
                    processed_images.append(processed_img)
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                # Use random noise as fallback
                processed_images.append(torch.randn(3, self.target_image_size, self.target_image_size))

        return torch.stack(processed_images, dim=0)

    def __getitem__(self, idx):
        """
        Get a single sample including embeddings and teacher images.

        Returns:
            Tuple of (prompt_embed, pooled_prompt_embeds, text_ids, caption, teacher_images, uuid)
        """
        sample_data = self.data_anno[idx]

        # Load FLUX embeddings
        prompt_embed_file = sample_data["prompt_embed_path"]
        pooled_prompt_embeds_file = sample_data["pooled_prompt_embeds_path"]
        text_ids_file = sample_data["text_ids"]
        caption = sample_data['caption']

        # Extract UUID from the filename (remove .pt extension)
        # The filename is the UUID used during preprocessing
        uuid = os.path.splitext(prompt_embed_file)[0]

        # Handle classifier-free guidance by replacing with negative prompt embeddings
        if random.random() < self.cfg_rate:
            # Use negative prompt embeddings (both prompt_embed and pooled_prompt_embeds)
            prompt_embed = self.negative_prompt_embeds.clone().squeeze()
            pooled_prompt_embeds = self.negative_pooled_prompt_embeds.clone().squeeze()
            caption = '(Uncondition)' + caption
        else:
            # Use positive prompt embeddings
            prompt_embed = torch.load(
                os.path.join(self.prompt_embed_dir, prompt_embed_file),
                map_location="cpu",
                weights_only=True,
            )
            pooled_prompt_embeds = torch.load(
                os.path.join(
                    self.pooled_prompt_embeds_dir, pooled_prompt_embeds_file
                ),
                map_location="cpu",
                weights_only=True,
            )

        # Create text_ids (FLUX uses these for position encoding)
        text_ids = torch.zeros(prompt_embed.shape[0], 3).to(device=prompt_embed.device, dtype=prompt_embed.dtype)

        # Load teacher images
        teacher_images = self._load_teacher_images(uuid)

        return prompt_embed, pooled_prompt_embeds, text_ids, caption, teacher_images, uuid

    def __len__(self):
        return len(self.data_anno)


def flux_fail_latent_collate_function(batch):
    """
    Collate function for FLUX FAIL dataset including teacher images.

    Args:
        batch: List of tuples from __getitem__

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images, uuids)
    """
    prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images, uuids = zip(*batch)

    # Stack embeddings
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)

    # Stack teacher images: [batch_size, teacher_number, 3, H, W]
    teacher_images = torch.stack(teacher_images, dim=0)

    return prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images, uuids


if __name__ == "__main__":
    # Test the FAIL dataset
    dataset = FluxFailLatentDataset(
        json_path="data/flux_rl_embeddings/videos2caption.json",
        num_latent_t=28,
        cfg_rate=0.0,
        teacher_image_root="data/teacher_images",
        rollout_number=4,
        target_image_size=1024,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=flux_fail_latent_collate_function
    )

    for i, (prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images, uuids) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"  Prompt embeds shape: {prompt_embeds.shape}")
        print(f"  Pooled prompt embeds shape: {pooled_prompt_embeds.shape}")
        print(f"  Text IDs shape: {text_ids.shape}")
        print(f"  Teacher images shape: {teacher_images.shape}")
        print(f"  Captions: {captions}")
        print(f"  UUIDs: {uuids}")
        if i >= 1:  # Only show first 2 batches
            break