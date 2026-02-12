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


class FluxSFTLatentDataset(Dataset):
    """
    FLUX dataset for SFT (Supervised Fine-Tuning) / Behavior Cloning training.

    This dataset treats each teacher image independently, expanding the dataset from
    25k prompts with 8 images each to 200k individual training samples.
    Each sample includes the prompt embeddings and one teacher image.
    """

    def __init__(
        self,
        json_path,
        num_latent_t,
        cfg_rate,
        teacher_image_root,
        target_image_size=1024,
        max_images_per_prompt=8,
        negative_prompt_embeds=None,
        negative_pooled_prompt_embeds=None,
    ):
        """
        Initialize SFT dataset for FLUX.

        Args:
            json_path: Path to the JSON file containing prompt embeddings metadata
            num_latent_t: Number of latent timesteps (unused but kept for compatibility)
            cfg_rate: Rate for classifier-free guidance during training
            teacher_image_root: Root directory containing teacher images organized by UUID
            target_image_size: Target size for teacher image preprocessing
            max_images_per_prompt: Maximum number of teacher images to use per prompt
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

        # SFT-specific parameters
        self.teacher_image_root = teacher_image_root
        self.target_image_size = target_image_size
        self.max_images_per_prompt = max_images_per_prompt

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

        # Build expanded dataset: each teacher image becomes a separate sample
        self.expanded_samples = []
        self._build_expanded_dataset()

    def _build_expanded_dataset(self):
        """
        Expand the dataset so each teacher image is a separate training sample.
        This converts 25k prompts x 8 images = 200k samples.
        """
        logger.info(f"Building expanded SFT dataset from {len(self.data_anno)} prompts...")

        for prompt_idx, sample_data in enumerate(self.data_anno):
            # Extract UUID from the filename (remove .pt extension)
            prompt_embed_file = sample_data["prompt_embed_path"]
            uuid = os.path.splitext(prompt_embed_file)[0]

            # Find all teacher images for this prompt
            image_dir = Path(self.teacher_image_root) / uuid

            if not image_dir.exists():
                logger.warning(f"Teacher image directory {image_dir} not found for UUID {uuid}")
                continue

            # Get all PNG files in the directory
            image_files = sorted(list(image_dir.glob("*.png")))[:self.max_images_per_prompt]

            if len(image_files) == 0:
                logger.warning(f"No PNG images found in {image_dir}")
                continue

            # Create a separate sample for each teacher image
            for img_idx, img_path in enumerate(image_files):
                self.expanded_samples.append({
                    "prompt_data_idx": prompt_idx,
                    "teacher_image_path": str(img_path),
                    "uuid": uuid,
                    "image_idx": img_idx,
                })

        logger.info(f"Expanded dataset contains {len(self.expanded_samples)} individual samples")

    def _load_teacher_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess a single teacher image.

        Args:
            image_path: Path to the teacher image

        Returns:
            Tensor of shape (3, H, W) containing the processed teacher image
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")
                processed_img = self.image_transforms(img)
                return processed_img
        except Exception as e:
            logger.warning(f"Failed to load image {image_path}: {e}")
            # Use random noise as fallback
            return torch.randn(3, self.target_image_size, self.target_image_size)

    def __getitem__(self, idx):
        """
        Get a single sample including embeddings and one teacher image.

        Returns:
            Tuple of (prompt_embed, pooled_prompt_embeds, text_ids, caption, teacher_image)
        """
        sample = self.expanded_samples[idx]
        prompt_data = self.data_anno[sample["prompt_data_idx"]]

        # Load FLUX embeddings
        prompt_embed_file = prompt_data["prompt_embed_path"]
        pooled_prompt_embeds_file = prompt_data["pooled_prompt_embeds_path"]
        text_ids_file = prompt_data["text_ids"]
        caption = prompt_data['caption']

        # Handle classifier-free guidance by replacing with negative prompt embeddings
        if random.random() < self.cfg_rate:
            # Use negative prompt embeddings (both prompt_embed and pooled_prompt_embeds)
            prompt_embed = self.negative_prompt_embeds.clone().squeeze()
            pooled_prompt_embeds = self.negative_pooled_prompt_embeds.clone().squeeze()
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

        # Load the specific teacher image for this sample
        teacher_image = self._load_teacher_image(sample["teacher_image_path"])

        return prompt_embed, pooled_prompt_embeds, text_ids, caption, teacher_image

    def __len__(self):
        return len(self.expanded_samples)


def flux_sft_latent_collate_function(batch):
    """
    Collate function for FLUX SFT dataset.

    Args:
        batch: List of tuples from __getitem__

    Returns:
        Tuple of (prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images)
    """
    prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images = zip(*batch)

    # Stack embeddings
    prompt_embeds = torch.stack(prompt_embeds, dim=0)
    pooled_prompt_embeds = torch.stack(pooled_prompt_embeds, dim=0)
    text_ids = torch.stack(text_ids, dim=0)

    # Stack teacher images: [batch_size, 3, H, W]
    teacher_images = torch.stack(teacher_images, dim=0)

    return prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images


if __name__ == "__main__":
    # Test the SFT dataset
    dataset = FluxSFTLatentDataset(
        json_path="data/flux_rl_embeddings/videos2caption.json",
        num_latent_t=28,
        cfg_rate=0.0,
        teacher_image_root="data/teacher_images",
        target_image_size=1024,
        max_images_per_prompt=8,
    )

    print(f"Dataset size: {len(dataset)}")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=flux_sft_latent_collate_function
    )

    for i, (prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images) in enumerate(dataloader):
        print(f"Batch {i}")
        print(f"  Prompt embeds shape: {prompt_embeds.shape}")
        print(f"  Pooled prompt embeds shape: {pooled_prompt_embeds.shape}")
        print(f"  Text IDs shape: {text_ids.shape}")
        print(f"  Teacher images shape: {teacher_images.shape}")
        print(f"  Captions: {captions}")
        if i >= 1:  # Only show first 2 batches
            break
