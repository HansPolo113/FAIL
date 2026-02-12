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

import argparse
import math
import os
from pathlib import Path
from fastvideo.utils.parallel_states import (
    initialize_sequence_parallel_state,
    destroy_sequence_parallel_group,
    get_sequence_parallel_state,
    nccl_info,
)
from fastvideo.utils.communications_flux import sp_parallel_dataloader_wrapper

import time
from torch.utils.data import DataLoader
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.state_dict import get_model_state_dict, set_model_state_dict, StateDictOptions

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_flux_rl_datasets import LatentDataset, latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# FPO imports
from fastvideo.utils.fpo_trainer import FpoTrainer
from fastvideo.models.flux_hf.pipeline_flux import FluxPipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

import logging

logging.basicConfig(level=logging.INFO)


def gather_tensor(tensor):
    """Gather tensor across all processes."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def load_checkpoint_into_transformer(transformer, checkpoint_path: str) -> None:
    """
    Load checkpoint into transformer model (before FSDP wrapping).

    Supports multiple checkpoint formats:
    - .pt files (PyTorch checkpoints)
    - .safetensors files
    - Directories containing diffusion_pytorch_model.safetensors
    """
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)

    main_print(f"Loading transformer checkpoint from {checkpoint_path}")

    # Determine checkpoint format and load state dict
    if checkpoint_path.suffix == ".pt":
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if 'transformer_state_dict' in checkpoint:
            state_dict = checkpoint['transformer_state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    elif checkpoint_path.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_path))
    elif checkpoint_path.is_dir():
        safetensors_path = checkpoint_path / "diffusion_pytorch_model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"Safetensors file not found at {safetensors_path}")
        state_dict = load_file(str(safetensors_path))
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    # Clean state dict (remove common prefixes that may exist in checkpoints)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "")  # In case checkpoint was FSDP-saved
        new_key = new_key.replace("module.", "")  # In case checkpoint was DDP-saved
        cleaned_state_dict[new_key] = value

    # Load into transformer
    missing_keys, unexpected_keys = transformer.load_state_dict(cleaned_state_dict, strict=False)

    if missing_keys:
        main_print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        main_print(f"  Unexpected keys: {len(unexpected_keys)}")
    main_print("Transformer checkpoint loaded successfully")


def parse_args():
    parser = argparse.ArgumentParser(description="FPO training script for FLUX")

    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./data/flux",
                        help="Path to pretrained model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None,
                        help="Variant of the model files of the pretrained model identifier from huggingface.co/models")
    parser.add_argument("--pretrained_transformer_path", type=str, default=None,
                        help="Path to pretrained transformer checkpoint (e.g., from SFT)")

    # Dataset arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the training data JSON file")
    parser.add_argument("--validation_data_path", type=str, default=None,
                        help="Path to the validation data JSON file (optional)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./fpo_flux_output",
                        help="Output directory for saving checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size per device during training")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--lr_scheduler", type=str, default="constant",
                        help="Learning rate scheduler")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Number of steps for the warmup in the lr scheduler")
    parser.add_argument("--max_train_steps", type=int, default=300,
                        help="Total number of training steps to perform")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")

    # FPO specific arguments
    parser.add_argument("--rollout_n", type=int, default=8,
                        help="Number of samples per prompt for FPO rollout")
    parser.add_argument("--num_mc_samples", type=int, default=8,
                        help="Number of Monte Carlo samples for CFM loss")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clipping range for FPO")
    parser.add_argument("--adv_clip_max", type=float, default=5.0,
                        help="Maximum advantage clipping")
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Number of update epochs per FPO step")
    parser.add_argument("--rollout_mini_batch_size", type=int, default=8,
                        help="Mini-batch size for rollout phase")
    parser.add_argument("--filter_ratio_threshold", type=float, default=None,
                        help="Filter samples with ratios outside [1-threshold, 1+threshold]. If None, uses clip_range")
    parser.add_argument("--enable_ratio_filtering", action="store_true",
                        help="Enable filtering of samples with extreme ratios")

    # Generation arguments
    parser.add_argument("--num_sampling_steps", type=int, default=14,
                        help="Number of sampling steps for image generation")
    parser.add_argument("--guidance_scale", type=float, default=1,
                        help="Guidance scale for image generation")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--init_same_noise", action="store_true",
                        help="Use same initial noise across samples")

    # Sequence parallel arguments
    parser.add_argument("--sp_size", type=int, default=1,
                        help="Sequence parallel size")
    parser.add_argument("--train_sp_batch_size", type=int, default=1,
                        help="Sequence parallel batch size")

    # Checkpointing arguments
    parser.add_argument("--checkpointing_steps", type=int, default=500,
                        help="Save a checkpoint every X steps")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Validation arguments
    parser.add_argument("--validation_steps", type=int, default=5000,
                        help="Run validation every X steps")

    # Logging arguments
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="Directory for storing logs")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Logging platform")
    parser.add_argument("--project_name", type=str, default="fpo-flux",
                        help="Project name for logging")
    parser.add_argument("--experiment_name", type=str, default="baseline",
                        help="Experiment_name name for logging")
    parser.add_argument("--image_log_interval", type=int, default=50,
                        help="Log images every N steps (0 to disable)")

    # Hardware arguments
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=1,
                        help="Number of workers for data loading")

    # Reward model arguments
    parser.add_argument("--reward_model_type", type=str, default="hpsv3",
                        choices=["hpsv3", "qwen3vl"],
                        help="Type of reward model to use")

    # HPSv3 reward model arguments
    parser.add_argument("--hpsv3_config_path", type=str, default=None,
                        help="Path to HPSv3 config file")
    parser.add_argument("--hpsv3_checkpoint_path", type=str, default=None,
                        help="Path to HPSv3 checkpoint file")
    parser.add_argument("--hpsv3_offload", action="store_true",
                        help="Offload HPSv3 model to CPU after use")

    # Qwen3VL reward model arguments
    parser.add_argument("--qwen3vl_model_path", type=str, default=None,
                        help="Path to Qwen3VL base model (e.g., Qwen/Qwen3-VL-7B)")
    parser.add_argument("--qwen3vl_checkpoint_path", type=str, default=None,
                        help="Path to trained Qwen3VL reward model checkpoint")
    parser.add_argument("--qwen3vl_model_size", type=str, default="7B",
                        choices=["2B", "7B"],
                        help="Size of Qwen3VL model")
    parser.add_argument("--qwen3vl_using_spec_norm", action="store_true",
                        help="Use spectral normalization in discriminator head")
    parser.add_argument("--qwen3vl_offload", action="store_true",
                        help="Offload Qwen3VL model to CPU after use")
    parser.add_argument("--qwen3vl_image_size", type=int, default=512,
                        help="Image size for Qwen3VL model")

    parser.add_argument("--reward_batch_size", type=int, default=None,
                        help="Batch size for reward compute")

    # KL regularization arguments
    parser.add_argument("--use_kl_regularization", action="store_true",
                        help="Enable KL regularization with reference model")
    parser.add_argument("--kl_beta", type=float, default=0.01,
                        help="Coefficient for KL loss term")
    parser.add_argument("--reference_model_offload", action="store_true",
                        help="Offload reference model to CPU to save GPU memory")

    args = parser.parse_args()
    return args


def run_validation(
    fpo_trainer,
    validation_dataset,
    global_step: int,
    guidance_scale: float,
    output_dir: str,
    local_rank: int,
    max_validation_samples: int = 32,
    rollout_n: int = 1,
):
    """Run validation using test dataset format with batch rollout and wandb image logging."""
    if local_rank != 0:
        return  # Only run validation on rank 0

    main_print(f"Running validation at step {global_step}...")

    # Use first max_validation_samples from validation dataset
    validation_samples = min(len(validation_dataset), max_validation_samples)
    validation_indices = list(range(validation_samples))
    validation_subset = torch.utils.data.Subset(validation_dataset, validation_indices)

    validation_dataloader = DataLoader(
        validation_subset,
        batch_size=8,  # Batch processing for efficiency
        shuffle=False,
        collate_fn=latent_collate_function,
        num_workers=1,
    )

    fpo_trainer.transformer.eval()
    all_rewards = []
    all_images = []
    all_captions = []

    with torch.no_grad():
        for batch_idx, (encoder_hidden_states, pooled_prompt_embeds, text_ids, captions) in enumerate(validation_dataloader):
            # Move to device
            encoder_hidden_states = encoder_hidden_states.to(fpo_trainer.device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(fpo_trainer.device)
            text_ids = text_ids.to(fpo_trainer.device)

            # Generate samples using batch rollout
            final_latents, expanded_captions = fpo_trainer.rollout_phase(
                encoder_hidden_states=encoder_hidden_states,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                captions=captions,
                rollout_n=rollout_n,
                mini_batch_size=8,
                guidance_scale=guidance_scale,
            )

            # Compute rewards
            rewards = fpo_trainer.compute_rewards(final_latents, expanded_captions)
            all_rewards.extend(rewards.cpu().tolist())

            # Decode images for wandb logging (sample a few)
            if len(all_images) < 32:  # Log up to 32 images total
                # Decode latents to images
                final_latents_scaled = (final_latents / fpo_trainer.flux_pipeline.vae.config.scaling_factor) + fpo_trainer.flux_pipeline.vae.config.shift_factor
                decoded_images = fpo_trainer.flux_pipeline.vae.decode(final_latents_scaled.to(fpo_trainer.flux_pipeline.vae.dtype), return_dict=False)[0]
                image_pil = fpo_trainer.flux_pipeline.image_processor.postprocess(decoded_images, output_type="pil")

                # Add images and captions for wandb logging
                for i, (img, caption, reward) in enumerate(zip(image_pil, expanded_captions, rewards)):
                    if len(all_images) >= 32:
                        break
                    all_images.append(img)
                    all_captions.append(f"Reward: {reward:.3f} | {caption}")

            main_print(f"Validation batch {batch_idx + 1}: mean reward = {rewards.mean().item():.4f}")

    # Compute validation metrics
    mean_validation_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    validation_metrics = {
        "validation/mean_reward": mean_validation_reward,
        "validation/max_reward": max(all_rewards) if all_rewards else 0.0,
        "validation/min_reward": min(all_rewards) if all_rewards else 0.0,
        "validation/num_samples": len(all_rewards),
    }

    # Log validation metrics and images to wandb
    import wandb
    validation_metrics["validation/images"] = [
        wandb.Image(img, caption=caption)
        for img, caption in zip(all_images, all_captions)
    ]

    wandb.log(validation_metrics, step=global_step)

    main_print(f"Validation complete. Mean reward: {mean_validation_reward:.4f} ({len(all_rewards)} samples, {len(all_images)} images logged)")

    fpo_trainer.transformer.train()  # Switch back to training mode


def main():
    args = parse_args()

    # Setup distributed training
    local_rank = int(os.getenv("LOCAL_RANK", 0))  # Local rank within node
    global_rank = int(os.getenv("RANK", 0))       # Global rank across all nodes
    world_size = int(os.getenv("WORLD_SIZE", 1))

    main_print("Initializing distributed training...")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)

    # Setup sequence parallel if needed
    if args.sp_size > 1:
        initialize_sequence_parallel_state(args.sp_size)

    # Set seeds
    set_seed(args.seed)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize logging
    if global_rank == 0 and args.report_to == "wandb":
        wandb.init(
            project=args.project_name,
            name=args.experiment_name,
            config=vars(args),
            dir=args.logging_dir,
        )

    main_print("Loading models...")

    # Load FLUX pipeline without text encoders (since we already have embeddings)
    flux_pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
        text_encoder=None,      # Skip text encoder loading
        text_encoder_2=None,    # Skip text encoder 2 loading
        tokenizer=None,         # Skip tokenizer loading
        tokenizer_2=None,       # Skip tokenizer 2 loading
    )
    flux_pipeline = flux_pipeline.to(device)
    flux_pipeline.set_progress_bar_config(disable=True)

    # Extract components
    transformer = flux_pipeline.transformer
    vae = flux_pipeline.vae

    # Load pretrained transformer checkpoint if provided (e.g., SFT model)
    # Load BEFORE FSDP wrapping for cleaner state dict handling
    if args.pretrained_transformer_path is not None:
        load_checkpoint_into_transformer(transformer, args.pretrained_transformer_path)

    # Setup FSDP for transformer
    main_print("Setting up FSDP...")
    # Enable cpu_offload to offload parameters to CPU when not in use
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(transformer, 'full', cpu_offload=False)
    transformer = FSDP(transformer, **fsdp_kwargs)
    apply_fsdp_checkpointing(
        transformer, no_split_modules
    )

    # Update pipeline with FSDP-wrapped transformer
    flux_pipeline.transformer = transformer

    # Load reward model
    main_print(f"Loading reward model ({args.reward_model_type})...")
    if args.reward_model_type == "hpsv3":
        from hpsv3 import HPSv3RewardInferencer

        reward_model = HPSv3RewardInferencer(
            config_path=args.hpsv3_config_path,
            checkpoint_path=args.hpsv3_checkpoint_path,
            device=device,
            differentiable=False
        )

        if args.hpsv3_offload:
            reward_model.model = reward_model.model.to('cpu')
            reward_model.device = 'cpu'
            main_print("HPSv3 model initialized and offloaded to CPU")
        else:
            main_print(f"HPSv3 model initialized on {device}")

    elif args.reward_model_type == "qwen3vl":
        from fastvideo.models.Qwen3VL import Qwen3VLDiscriminator

        if args.qwen3vl_model_path is None:
            raise ValueError("--qwen3vl_model_path is required for qwen3vl reward model")

        main_print(f"Loading Qwen3-VL reward model from: {args.qwen3vl_model_path}")

        # Initialize discriminator (all parameters trainable during training, but frozen for inference)
        reward_model = Qwen3VLDiscriminator(
            model_path=args.qwen3vl_model_path,
            using_spec_norm=args.qwen3vl_using_spec_norm,
            freeze_vision_encoder=False,  # All parameters were trained
            image_size=args.qwen3vl_image_size,
        )
        reward_model = reward_model.to(torch.bfloat16).to(device)

        # Load trained checkpoint if provided
        if args.qwen3vl_checkpoint_path is not None:
            main_print(f"Loading Qwen3VL checkpoint from {args.qwen3vl_checkpoint_path}")
            reward_checkpoint = torch.load(args.qwen3vl_checkpoint_path, map_location="cpu")
            if 'model_state_dict' in reward_checkpoint:
                reward_state_dict = reward_checkpoint['model_state_dict']
            else:
                reward_state_dict = reward_checkpoint

            # Clean state dict (remove DDP/FSDP prefixes if present)
            cleaned_reward_state_dict = {}
            for key, value in reward_state_dict.items():
                new_key = key.replace("_fsdp_wrapped_module.", "")
                new_key = new_key.replace("module.", "")
                cleaned_reward_state_dict[new_key] = value

            reward_model.load_state_dict(cleaned_reward_state_dict, strict=True)
            main_print("Qwen3VL checkpoint loaded successfully")

        # Set to evaluation mode and freeze
        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False

        # Handle offloading
        if args.qwen3vl_offload:
            reward_model = reward_model.to('cpu')
            main_print("Qwen3VL reward model initialized and offloaded to CPU")
        else:
            main_print(f"Qwen3VL reward model initialized on {device}")

    else:
        raise ValueError(f"Unsupported reward model type: {args.reward_model_type}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )

    # Setup learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Load dataset
    main_print("Loading dataset...")
    train_dataset = LatentDataset(args.data_path, None, 0)

    # Load validation dataset if provided
    validation_dataset = None
    if args.validation_data_path:
        main_print("Loading validation dataset...")
        validation_dataset = LatentDataset(args.validation_data_path, None, 0)

    # Setup data loader
    sampler = DistributedSampler(train_dataset, shuffle=True)


    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=latent_collate_function,
        num_workers=args.dataloader_num_workers,
        shuffle=(sampler is None),
    )

    # Wrap dataloader for sequence parallel if needed
    if args.sp_size > 1:
        train_dataloader = sp_parallel_dataloader_wrapper(
            train_dataloader, device, args.train_batch_size, args.sp_size, args.train_sp_batch_size
        )

    # Initialize reference transformer for KL regularization if enabled
    reference_transformer = None
    if args.use_kl_regularization:
        main_print("Initializing reference transformer for KL regularization...")

        # Load a fresh FLUX pipeline for reference model
        reference_flux_pipeline = FluxPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            revision=args.revision,
            variant=args.variant,
            torch_dtype=torch.bfloat16,
            text_encoder=None,      # Skip text encoder loading
            text_encoder_2=None,    # Skip text encoder 2 loading
            tokenizer=None,         # Skip tokenizer loading
            tokenizer_2=None,       # Skip tokenizer 2 loading
        )

        # Extract reference transformer
        reference_transformer = reference_flux_pipeline.transformer
        reference_transformer = reference_transformer.to(device)

        # Load pretrained checkpoint for reference model if provided
        if args.pretrained_transformer_path is not None:
            main_print(f"Loading reference transformer from pretrained checkpoint: {args.pretrained_transformer_path}")
            load_checkpoint_into_transformer(reference_transformer, args.pretrained_transformer_path)

        # Freeze reference model
        for param in reference_transformer.parameters():
            param.requires_grad = False
        reference_transformer.eval()

        # Offload to CPU if requested
        if args.reference_model_offload:
            reference_transformer = reference_transformer.to('cpu')
            main_print("Reference transformer initialized and offloaded to CPU")
        else:
            main_print(f"Reference transformer initialized on {device}")

        # Clean up temporary pipeline
        del reference_flux_pipeline
        torch.cuda.empty_cache()

    # Initialize FPO trainer
    main_print("Initializing FPO trainer...")
    fpo_trainer = FpoTrainer(
        transformer=transformer,
        vae=vae,
        flux_pipeline=flux_pipeline,
        reward_model=reward_model,
        optimizer=optimizer,
        device=device,
        args=args,
        clip_range=args.clip_range,
        adv_clip_max=args.adv_clip_max,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        reward_batch_size=args.reward_batch_size,
        reference_transformer=reference_transformer if args.use_kl_regularization else None,
        kl_beta=args.kl_beta if args.use_kl_regularization else 0.0,
        image_log_interval=args.image_log_interval,
    )

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        main_print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        fpo_trainer.load_checkpoint(args.resume_from_checkpoint)

    # Training loop
    main_print("Starting FPO training...")
    global_step = 0

    for epoch in range(1000):  # Large number, will be limited by max_train_steps
        for step, (encoder_hidden_states, pooled_prompt_embeds, text_ids, captions) in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            # Move data to device
            encoder_hidden_states = encoder_hidden_states.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            text_ids = text_ids.to(device)

            # FPO training step
            metrics = fpo_trainer.train_step(
                encoder_hidden_states=encoder_hidden_states,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                captions=captions,
                rollout_n=args.rollout_n,
                num_mc_samples=args.num_mc_samples,
                mini_batch_size=args.rollout_mini_batch_size,
                guidance_scale=args.guidance_scale,
            )

            # Update learning rate
            lr_scheduler.step()

            # Log metrics
            if global_rank == 0:
                metrics["train/learning_rate"] = lr_scheduler.get_last_lr()[0]
                metrics["train/global_step"] = global_step

                if args.report_to == "wandb":
                    wandb.log(metrics, step=global_step)

            # Save checkpoint
            if global_step > 0 and global_step % args.checkpointing_steps == 0:
                checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                fpo_trainer.save_checkpoint(checkpoint_path)

            # Validation
            if global_step > 0 and global_step % args.validation_steps == 0 and validation_dataset is not None:
                run_validation(
                    fpo_trainer=fpo_trainer,
                    validation_dataset=validation_dataset,
                    global_step=global_step,
                    guidance_scale=args.guidance_scale,
                    output_dir=args.output_dir,
                    local_rank=global_rank,
                    max_validation_samples=8,
                    rollout_n=1,
                )
            dist.barrier()

            global_step += 1

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "checkpoint-final")
    fpo_trainer.save_checkpoint(final_checkpoint_path)

    # Cleanup
    if args.sp_size > 1:
        destroy_sequence_parallel_group()

    if global_rank == 0 and args.report_to == "wandb":
        wandb.finish()

    main_print("FPO training completed!")


if __name__ == "__main__":
    main()