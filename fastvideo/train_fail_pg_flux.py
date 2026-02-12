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
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy, StateDictType, FullStateDictConfig
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

from torch.utils.data.distributed import DistributedSampler
from fastvideo.utils.dataset_utils import LengthGroupedSampler
import wandb
from accelerate.utils import set_seed
from tqdm.auto import tqdm
from fastvideo.utils.fsdp_util import get_dit_fsdp_kwargs, apply_fsdp_checkpointing
from fastvideo.utils.load import load_transformer
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version
from fastvideo.dataset.latent_flux_fail_datasets import FluxFailLatentDataset, flux_fail_latent_collate_function
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from fastvideo.utils.checkpoint import (
    save_checkpoint,
    save_lora_checkpoint,
)
from fastvideo.utils.logging_ import main_print
import cv2
from diffusers.image_processor import VaeImageProcessor

# FAIL-PG imports
from fastvideo.utils.flux_fail_pg_trainer import FluxFailPGTrainer
from diffusers import FluxPipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.30.0.dev0")

import logging

logging.basicConfig(level=logging.INFO)


def load_checkpoint_into_transformer(transformer, checkpoint_path: str) -> None:
    """Load checkpoint into transformer model (before FSDP wrapping).

    Supports:
    - FAIL checkpoint: checkpoint-{step}/hf_weights/diffusion_pytorch_model.safetensors
    - SFT checkpoint: checkpoint-{step}-{epoch}/diffusion_pytorch_model.safetensors
    - Direct safetensors file
    """
    from safetensors.torch import load_file

    checkpoint_path = Path(checkpoint_path)

    main_print(f"Loading transformer checkpoint from {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        state_dict = load_file(str(checkpoint_path))
    elif checkpoint_path.is_dir():
        hf_weights_path = checkpoint_path / "hf_weights" / "diffusion_pytorch_model.safetensors"
        direct_path = checkpoint_path / "diffusion_pytorch_model.safetensors"

        if hf_weights_path.exists():
            main_print(f"  Loading from FAIL checkpoint format: {hf_weights_path}")
            state_dict = load_file(str(hf_weights_path))
        elif direct_path.exists():
            main_print(f"  Loading from SFT checkpoint format: {direct_path}")
            state_dict = load_file(str(direct_path))
        else:
            raise FileNotFoundError(
                f"No safetensors file found at {checkpoint_path}. "
                f"Expected either {hf_weights_path} or {direct_path}"
            )
    else:
        raise ValueError(f"Unsupported checkpoint format at {checkpoint_path}")

    cleaned_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("_fsdp_wrapped_module.", "")
        new_key = new_key.replace("module.", "")
        cleaned_state_dict[new_key] = value

    missing_keys, unexpected_keys = transformer.load_state_dict(cleaned_state_dict, strict=False)

    if missing_keys:
        main_print(f"  Missing keys: {len(missing_keys)}")
    if unexpected_keys:
        main_print(f"  Unexpected keys: {len(unexpected_keys)}")
    main_print("Transformer checkpoint loaded successfully")


def parse_args():
    parser = argparse.ArgumentParser(description="GAIL FPO training script for FLUX")

    # Model arguments
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./data/flux",
                        help="Path to pretrained FLUX model or model identifier from huggingface.co/models.")
    parser.add_argument("--revision", type=str, default=None,
                        help="Revision of pretrained model identifier from huggingface.co/models.")
    parser.add_argument("--variant", type=str, default=None,
                        help="Variant of the model files of the pretrained model identifier from huggingface.co/models")
    parser.add_argument("--pretrained_transformer_path", type=str, default=None,
                        help="Path to pretrained transformer checkpoint (e.g., SFT model) to continue RL training from. Supports .pt, .safetensors, or directory with diffusion_pytorch_model.safetensors")

    # Discriminator arguments (QwenVL only)
    parser.add_argument("--discriminator_model_size", type=str, default="2B", choices=["2B", "7B"],
                        help="Model size for QwenVL discriminator")
    parser.add_argument("--discriminator_using_spec_norm", action="store_true", default=False,
                        help="Use spectral normalization in discriminator")
    parser.add_argument("--discriminator_freeze_backbone", action="store_true", default=False,
                        help="Freeze QwenVL vision encoder")

    # QwenVL discriminator arguments
    parser.add_argument("--qwen3vl_model_path", type=str, required=True,
                        help="Path to Qwen3-VL model checkpoint")
    parser.add_argument("--discriminator_image_size", type=int, default=512,
                        help="Target image size for discriminator preprocessing")

    # General discriminator training arguments
    parser.add_argument("--discriminator_learning_rate", type=float, default=2e-5,
                        help="Discriminator learning rate")
    parser.add_argument("--discriminator_weight_decay", type=float, default=0.,
                        help="Discriminator weight decay")
    parser.add_argument("--discriminator_grad_clip", type=float, default=1.0,
                        help="Discriminator gradient clipping threshold")
    parser.add_argument("--discriminator_warmup_steps", type=int, default=0,
                        help="Number of steps to only update discriminator (no policy updates)")

    # Dataset arguments
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the training data JSON file")
    parser.add_argument("--teacher_image_root", type=str, required=True,
                        help="Root directory containing teacher images organized by UUID")
    parser.add_argument("--validation_data_path", type=str, default=None,
                        help="Path to the validation data JSON file (optional)")

    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./fail_pg_flux_output",
                        help="Output directory for saving checkpoints and logs")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--train_batch_size", type=int, default=1,
                        help="Batch size per device during training")
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

    # GAIL FPO specific arguments
    parser.add_argument("--rollout_n", type=int, default=8,
                        help="Number of samples per prompt for FPO rollout")
    parser.add_argument("--teacher_n", type=int, default=None,
                        help="Number of teacher images per prompt (defaults to rollout_n if not specified)")
    parser.add_argument("--num_mc_samples", type=int, default=8,
                        help="Number of Monte Carlo samples for CFM loss")
    parser.add_argument("--clip_range", type=float, default=0.2,
                        help="PPO clipping range for FPO")
    parser.add_argument("--num_epochs", type=int, default=4,
                        help="Number of update epochs per FPO step")
    parser.add_argument("--rollout_mini_batch_size", type=int, default=8,
                        help="Mini-batch size for rollout phase")
    parser.add_argument("--discriminator_updates_per_step", type=int, default=1,
                        help="Number of discriminator updates per training step")
    parser.add_argument("--include_teacher_in_policy", action="store_true",
                        help="Include teacher images in policy updates")

    # Generation arguments
    parser.add_argument("--num_sampling_steps", type=int, default=28,
                        help="Number of sampling steps for image generation")
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale for image generation")
    parser.add_argument("--height", type=int, default=1024,
                        help="Image height")
    parser.add_argument("--width", type=int, default=1024,
                        help="Image width")
    parser.add_argument("--init_same_noise", action="store_true",
                        help="Use same initial noise across samples")
    parser.add_argument("--cfg_rate", type=float, default=0.0,
                        help="Classifier-free guidance rate for prompt dropout during training")
    parser.add_argument("--negative_prompt", type=str, default="",
                        help="Negative prompt for CFG training")
    parser.add_argument("--max_sequence_length", type=int, default=512,
                        help="Maximum sequence length for text encoding")
    parser.add_argument("--rollout_negative_promot", type=str, default="",
                        help="Negative prompt for CFG rollout")
    parser.add_argument("--rollout_true_cfg", type=float, default=1.0,
                        help="Classifier-free guidance rate for prompt dropout during rolloutout")

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
    parser.add_argument("--validation_steps", type=int, default=500,
                        help="Run validation every X steps")

    # Logging arguments
    parser.add_argument("--logging_dir", type=str, default="logs",
                        help="Directory for storing logs")
    parser.add_argument("--report_to", type=str, default="wandb",
                        help="Logging platform")
    parser.add_argument("--project_name", type=str, default="fail-pg-flux",
                        help="Project name for logging")
    parser.add_argument("--experiment_name", type=str, default="baseline",
                        help="Experiment name for logging")
    parser.add_argument("--image_log_interval", type=int, default=50,
                        help="Log images to wandb every X steps (0 to disable)")

    # Hardware arguments
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training")
    parser.add_argument("--dataloader_num_workers", type=int, default=1,
                        help="Number of workers for data loading")

    # KL regularization arguments
    parser.add_argument("--use_kl_regularization", action="store_true",
                        help="Enable KL regularization with reference model")
    parser.add_argument("--kl_beta", type=float, default=0.01,
                        help="Coefficient for KL loss term")
    parser.add_argument("--reference_model_offload", action="store_true",
                        help="Offload reference model to CPU to save GPU memory")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # Validate and set default for teacher_n
    if args.teacher_n is None:
        args.teacher_n = args.rollout_n
    assert args.teacher_n > 0, "teacher_n must be > 0"

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
    set_seed(args.seed+global_rank)

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

    main_print("Loading FLUX pipeline...")
    from fastvideo.models.flux_hf.transformer_flux import FluxTransformer2DModel
    transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
        subfolder='transformer',
    )

    # Load FLUX pipeline
    flux_pipeline = FluxPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        revision=args.revision,
        variant=args.variant,
        torch_dtype=torch.bfloat16,
    )
    flux_pipeline = flux_pipeline.to(device)
    flux_pipeline.set_progress_bar_config(disable=True)
    flux_pipeline.transformer.set_attention_backend("flash")

    # Extract components
    transformer = flux_pipeline.transformer
    vae = flux_pipeline.vae

    # Load pretrained transformer checkpoint if provided (e.g., SFT model)
    # Load BEFORE FSDP wrapping for cleaner state dict handling
    if args.pretrained_transformer_path is not None:
        load_checkpoint_into_transformer(transformer, args.pretrained_transformer_path)

    # Setup FSDP for transformer
    main_print("Setting up FSDP for transformer...")
    fsdp_kwargs, no_split_modules = get_dit_fsdp_kwargs(transformer, 'full')
    transformer = FSDP(transformer, **fsdp_kwargs)
    apply_fsdp_checkpointing(transformer, no_split_modules)

    # Update pipeline with FSDP-wrapped transformer
    flux_pipeline.transformer = transformer
    flux_pipeline.transformer.to(torch.bfloat16)

    # Encode negative prompt embeddings for CFG
    main_print(f"--> Encoding negative prompt: '{args.negative_prompt}'")
    with torch.no_grad():
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            _,
        ) = flux_pipeline.encode_prompt(
            prompt=args.negative_prompt,
            prompt_2=args.negative_prompt,
            device=device,
            num_images_per_prompt=1,
            prompt_embeds=None,
            pooled_prompt_embeds=None,
            max_sequence_length=args.max_sequence_length,
        )
        # Move to CPU for dataset storage
        negative_prompt_embeds = negative_prompt_embeds.cpu()
        negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.cpu()

    # Initialize QwenVL discriminator
    main_print(f"Loading QwenVL discriminator, model size: {args.discriminator_model_size}")

    from fastvideo.models.Qwen3VL import Qwen3VLDiscriminator

    # Map model size to full model path
    if args.discriminator_model_size in ["2B", "7B"]:
        model_path = args.qwen3vl_model_path
        main_print(f"Loading Qwen3-VL discriminator from: {model_path}")
    else:
        raise ValueError(f"Unsupported Qwen3VL model size: {args.discriminator_model_size}. Use '2B' or '7B'")

    discriminator = Qwen3VLDiscriminator(
        model_path=model_path,
        using_spec_norm=args.discriminator_using_spec_norm,
        freeze_vision_encoder=args.discriminator_freeze_backbone,
        image_size=args.discriminator_image_size,
    )
    discriminator = discriminator.to(torch.bfloat16).to(device)

    main_print("Qwen3VL discriminator initialized")
    main_print(f"Vision encoder frozen: {args.discriminator_freeze_backbone}")

    # Setup FSDP for QwenVL discriminator
    main_print("Setting up FSDP for discriminator...")

    from fastvideo.utils.fsdp_util import get_qwen3vl_discriminator_fsdp_kwargs

    fsdp_kwargs, _ = get_qwen3vl_discriminator_fsdp_kwargs(
        discriminator,
        sharding_strategy="full",
        master_weight_type="bf16",
    )

    discriminator = FSDP(discriminator, **fsdp_kwargs)

    # Setup optimizers
    optimizer = torch.optim.AdamW(
        transformer.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.0,
        eps=1e-8,
    )

    discriminator_optimizer = torch.optim.AdamW(
        [p for p in discriminator.parameters() if p.requires_grad],
        lr=args.discriminator_learning_rate,
        betas=(0.9, 0.999),
        weight_decay=args.discriminator_weight_decay,
    )

    # Setup learning rate schedulers
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    discriminator_lr_scheduler = get_scheduler(
        "constant",  # Use constant scheduler for discriminator
        optimizer=discriminator_optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps,
    )

    # Load dataset
    main_print("Loading dataset...")
    train_dataset = FluxFailLatentDataset(
        json_path=args.data_path,
        num_latent_t=None,
        cfg_rate=args.cfg_rate,
        teacher_image_root=args.teacher_image_root,
        rollout_number=args.rollout_n,
        teacher_number=args.teacher_n,
        target_image_size=min(args.height, args.width),
        negative_prompt_embeds=negative_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
    )

    # Load validation dataset if provided
    validation_dataset = None
    if args.validation_data_path:
        main_print("Loading validation dataset...")
        validation_dataset = FluxFailLatentDataset(
            json_path=args.validation_data_path,
            num_latent_t=None,
            cfg_rate=args.cfg_rate,
            teacher_image_root=args.teacher_image_root,
            rollout_number=args.rollout_n,
            teacher_number=args.teacher_n,
            target_image_size=min(args.height, args.width),
            negative_prompt_embeds=negative_prompt_embeds,
            negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        )

    # Setup data loader
    sampler = DistributedSampler(train_dataset, shuffle=True)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        sampler=sampler,
        collate_fn=flux_fail_latent_collate_function,
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
        )

        # Extract reference transformer
        reference_transformer = reference_flux_pipeline.transformer
        reference_transformer = reference_transformer.to(device)

        # Load from pretrained checkpoint if provided (e.g., SFT model)
        # Reference model should match the starting point of training
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

    # Initialize GAIL FPO trainer
    main_print("Initializing GAIL FPO trainer...")
    fail_pg_trainer = FluxFailPGTrainer(
        transformer=transformer,
        vae=vae,
        flux_pipeline=flux_pipeline,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        discriminator_scheduler=discriminator_lr_scheduler,
        reward_model=None,  # Reward model disabled, monitoring handled in trainer
        optimizer=optimizer,
        device=device,
        args=args,
        clip_range=args.clip_range,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        discriminator_updates_per_step=args.discriminator_updates_per_step,
        include_teacher_in_policy=args.include_teacher_in_policy,
        discriminator_grad_clip=args.discriminator_grad_clip,
        discriminator_warmup_steps=args.discriminator_warmup_steps,
        reference_transformer=reference_transformer if args.use_kl_regularization else None,
        kl_beta=args.kl_beta if args.use_kl_regularization else 0.0,
        image_log_interval=args.image_log_interval,
    )

    # Training loop
    main_print("Starting GAIL FPO training with QwenVL discriminator...")
    global_step = 0

    # Resume from checkpoint if specified
    if args.resume_from_checkpoint:
        main_print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        fail_pg_trainer.load_checkpoint(args.resume_from_checkpoint)
        # Update global_step to match the restored trainer step_count
        global_step = fail_pg_trainer.step_count
        main_print(f"Resumed training from global step: {global_step}")

    for epoch in range(1000):  # Large number, will be limited by max_train_steps
        for step, (prompt_embeds, pooled_prompt_embeds, text_ids, captions, teacher_images, uuids) in enumerate(train_dataloader):
            if global_step >= args.max_train_steps:
                break

            # Move data to device
            prompt_embeds = prompt_embeds.to(device)
            pooled_prompt_embeds = pooled_prompt_embeds.to(device)
            text_ids = text_ids.to(device)
            teacher_images = teacher_images.to(device)

            # GAIL FPO training step
            metrics = fail_pg_trainer.train_step(
                encoder_hidden_states=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                text_ids=text_ids,
                teacher_images=teacher_images,
                captions=captions,
                rollout_n=args.rollout_n,
                teacher_n=args.teacher_n,
                num_mc_samples=args.num_mc_samples,
                mini_batch_size=args.rollout_mini_batch_size,
                guidance_scale=args.guidance_scale,
                rollout_negative_promots=[args.rollout_negative_promot * args.train_batch_size],
                rollout_true_cfg=args.rollout_true_cfg
            )

            # Update learning rate
            lr_scheduler.step()

            # Log metrics
            if global_rank == 0:
                metrics["train/learning_rate"] = lr_scheduler.get_last_lr()[0]
                metrics["train/discriminator_learning_rate"] = discriminator_lr_scheduler.get_last_lr()[0]
                metrics["train/global_step"] = global_step

                if args.report_to == "wandb":
                    wandb.log(metrics, step=global_step)

            # Save checkpoint
            if global_step > 0 and global_step % args.checkpointing_steps == 0:
                fail_pg_trainer.save_checkpoint(args.output_dir, step=global_step)

            # Validation (simplified for now)
            if global_step > 0 and global_step % args.validation_steps == 0 and validation_dataset is not None:
                main_print(f"Validation would run here at step {global_step}")
                # TODO: Implement validation logic similar to FPO trainer

            dist.barrier()

            global_step += 1

            if global_step >= args.max_train_steps:
                break

        if global_step >= args.max_train_steps:
            break

    # Save final checkpoint
    final_checkpoint_path = os.path.join(args.output_dir, "checkpoint-final.pt")
    fail_pg_trainer.save_checkpoint(args.output_dir, step=global_step)

    # Cleanup
    if args.sp_size > 1:
        destroy_sequence_parallel_group()

    if global_rank == 0 and args.report_to == "wandb":
        wandb.finish()

    main_print("GAIL FPO training with QwenVL discriminator completed!")


if __name__ == "__main__":
    main()
