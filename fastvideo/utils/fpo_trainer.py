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
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Union
import numpy as np
import time

from ..models.fpo_utils import FpoBatch, FpoUtils, FpoSampleInfo
from diffusers.image_processor import VaeImageProcessor

from .logging_ import main_print


def gather_tensor(tensor):
    """Gather tensor across all processes."""
    if not dist.is_initialized():
        return tensor

    world_size = dist.get_world_size()
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


class FpoTrainer:
    """
    FPO (Flow Policy Optimization) trainer for FLUX image generation.
    """

    def __init__(
            self,
            transformer,
            vae,
            flux_pipeline,
            reward_model,
            optimizer,
            device: torch.device,
            args,
            clip_range: float = 0.2,
            adv_clip_max: float = 5.0,
            max_grad_norm: float = 1.0,
            num_epochs: int = 4,
            reward_batch_size: int = None,
            reference_transformer=None,
            kl_beta: float = 0.0,
            image_log_interval: int = 50,
    ):
        """
        Initialize FPO trainer.

        Args:
            transformer: FLUX transformer model
            vae: VAE for image decoding
            flux_pipeline: FLUX pipeline for generation
            reward_model: Reward model (e.g., HPSv3 or Qwen3VL)
            optimizer: Optimizer for transformer
            device: Device to run on
            args: Training arguments
            clip_range: PPO clipping range
            adv_clip_max: Maximum advantage clipping
            max_grad_norm: Maximum gradient norm for clipping
            num_epochs: Number of update epochs per batch
            reward_batch_size: Batch size for reward computation
            reference_transformer: Reference transformer for KL regularization (optional)
            kl_beta: Coefficient for KL loss (default: 0.0, no KL regularization)
            image_log_interval: Log images every N steps (0 to disable)
        """
        self.transformer = transformer
        self.vae = vae
        self.flux_pipeline = flux_pipeline
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.device = device
        self.args = args

        self.clip_range = clip_range
        self.adv_clip_max = adv_clip_max
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.reward_batch_size = reward_batch_size

        # KL regularization
        self.reference_transformer = reference_transformer
        self.kl_beta = kl_beta
        if self.reference_transformer is not None:
            # Freeze reference model
            for param in self.reference_transformer.parameters():
                param.requires_grad = False
            self.reference_transformer.eval()

        # Ratio filtering parameters
        self.enable_ratio_filtering = getattr(args, 'enable_ratio_filtering', False)
        self.filter_ratio_threshold = getattr(args, 'filter_ratio_threshold', None) or clip_range

        # Monitoring parameters
        self.image_log_interval = image_log_interval

        # CFM loss computation will be done directly in trainer methods

        # Training statistics
        self.step_count = 0

    def rollout_phase(
            self,
            encoder_hidden_states: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            text_ids: torch.Tensor,
            captions: List[str],
            rollout_n: int = 4,
            mini_batch_size: int = 8,
            guidance_scale: float = 1.0,
            rollout_negative_promots: List[str] = None,
            rollout_true_cfg: float = 1.0,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Rollout phase: Generate images only.

        Args:
            encoder_hidden_states: Text embeddings [batch, seq_len, C]
            pooled_prompt_embeds: Pooled embeddings [batch, C]
            text_ids: Text IDs [batch, seq_len]
            captions: List of prompt strings
            rollout_n: Number of samples per prompt
            mini_batch_size: Mini-batch size for rollout
            guidance_scale: Guidance scale for generation

        Returns:
            Tuple of (final_latents, expanded_captions)
        """
        # Constants
        SPATIAL_DOWNSAMPLE = 8
        IN_CHANNELS = 16
        w, h = self.args.width, self.args.height
        latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

        batch_size = encoder_hidden_states.shape[0]
        total_samples = batch_size * rollout_n

        # Ensure total_samples is divisible by mini_batch_size
        assert total_samples % mini_batch_size == 0, f"total_samples ({total_samples}) must be divisible by mini_batch_size ({mini_batch_size})"

        main_print(f"FPO rollout: {batch_size} prompts × {rollout_n} samples = {total_samples} total")

        # Expand batch with repeat_interleave
        expanded_encoder_hidden_states = encoder_hidden_states.repeat_interleave(rollout_n, dim=0)
        expanded_pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(rollout_n, dim=0)
        expanded_text_ids = text_ids.repeat_interleave(rollout_n, dim=0)
        expanded_captions = [caption for caption in captions for _ in range(rollout_n)]
        if rollout_negative_promots:
            expanded_rollout_negative_promots = [rollout_negative_promot for rollout_negative_promot in rollout_negative_promots for _ in range(rollout_n)]
        else:
            expanded_rollout_negative_promots = ['' for caption in captions for _ in range(rollout_n)]

            # Generate images in mini-batches using FLUX pipeline
        all_final_latents = []

        # Use FSDP summon_full_params context to ensure parameters are on GPU during rollout
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        with torch.no_grad():
            for mini_start in range(0, total_samples, mini_batch_size):
                mini_end = min(mini_start + mini_batch_size, total_samples)
                mini_batch_actual_size = mini_end - mini_start

                # Get mini-batch data
                mini_encoder_hidden_states = expanded_encoder_hidden_states[mini_start:mini_end]
                mini_pooled_prompt_embeds = expanded_pooled_prompt_embeds[mini_start:mini_end]
                mini_text_ids = expanded_text_ids[mini_start:mini_end]
                mini_captions = expanded_captions[mini_start:mini_end]
                mini_rollout_negative_promots = expanded_rollout_negative_promots[mini_start:mini_end]

                # Generate different initial noise for each sample
                if self.args.init_same_noise:
                    base_noise = torch.randn((1, IN_CHANNELS, latent_h, latent_w), device=self.device,
                                             dtype=torch.bfloat16)
                    mini_initial_noise = base_noise.repeat(mini_batch_actual_size, 1, 1, 1)
                else:
                    mini_initial_noise = torch.randn((mini_batch_actual_size, IN_CHANNELS, latent_h, latent_w),
                                                     device=self.device, dtype=torch.bfloat16)

                    # Use FLUX pipeline for generation
                    pipeline_output = self.flux_pipeline(
                        # prompt=mini_captions,
                        height=h,
                        width=w,
                        num_inference_steps=self.args.num_sampling_steps,
                        guidance_scale=guidance_scale,
                        output_type="latent",
                        return_dict=False,
                        prompt_embeds=mini_encoder_hidden_states,
                        pooled_prompt_embeds=mini_pooled_prompt_embeds,
                        latents=self.flux_pipeline._pack_latents(mini_initial_noise, mini_batch_actual_size,
                                                                 IN_CHANNELS, latent_h, latent_w),
                        negative_prompt=mini_rollout_negative_promots,
                        true_cfg_scale=rollout_true_cfg,
                    )

                # Extract and unpack final latents
                final_latents_packed = pipeline_output[0]
                final_latents_unpacked = self.flux_pipeline._unpack_latents(final_latents_packed, h, w,
                                                                            self.flux_pipeline.vae_scale_factor)

                all_final_latents.append(final_latents_unpacked.to(dtype=mini_encoder_hidden_states.dtype))

        # Concatenate all results
        all_final_latents = torch.cat(all_final_latents, dim=0)  # [batch*rollout_n, C, H, W]

        return all_final_latents, expanded_captions

    def _compute_cfm_loss_batch(
            self,
            x_1: torch.Tensor,  # [batch_size, C, H, W] - final denoised images
            t_samples: torch.Tensor,  # [batch_size, num_mc_samples] - timestep samples
            eps_samples: torch.Tensor,  # [batch_size, num_mc_samples, C, H, W] - noise samples
            encoder_hidden_states: torch.Tensor,  # [batch_size, seq_len, dim] - text embeddings
            pooled_prompt_embeds: torch.Tensor,  # [batch_size, dim] - pooled embeddings
            text_ids: torch.Tensor,  # [batch_size, seq_len] - text IDs
            guidance_scale: float = 1.0,
            use_grad: bool = False,  # Whether to compute with gradients
            return_velocity: bool = False,  # Whether to return velocity predictions
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute CFM loss for a batch of samples with MC sampling.

        Args:
            x_1: Final denoised images
            t_samples: Sampled timesteps for CFM loss
            eps_samples: Sampled noise vectors for CFM loss
            encoder_hidden_states: Text embeddings
            pooled_prompt_embeds: Pooled text embeddings
            text_ids: Text token IDs
            guidance_scale: Guidance scale for conditional generation
            use_grad: Whether to compute with gradients (for training) or no_grad (for rollout)
            return_velocity: Whether to return velocity predictions for KL computation

        Returns:
            cfm_losses: CFM losses [batch_size]
            velocity_pred (optional): Velocity predictions if return_velocity=True [batch_size * num_mc_samples, seq_len, channels]
        """
        batch_size, num_mc_samples = t_samples.shape
        _, C, latent_h, latent_w = x_1.shape

        # Pack latents first for efficient computation
        x_1_packed = self.flux_pipeline._pack_latents(x_1, batch_size, C, latent_h,
                                                      latent_w)  # [batch_size, seq_len, channels]

        # Flatten batch and MC dimensions
        t_flat = t_samples.flatten()  # [batch_size * num_mc_samples]
        eps_flat = eps_samples.view(batch_size * num_mc_samples, C, latent_h, latent_w)

        # Pack noise samples
        eps_packed = self.flux_pipeline._pack_latents(eps_flat, batch_size * num_mc_samples, C, latent_h,
                                                      latent_w)  # [batch_size * num_mc_samples, seq_len, channels]

        # Repeat packed x_1 to match flattened batch size
        x_1_packed_flat = x_1_packed.repeat_interleave(num_mc_samples,
                                                       dim=0).detach()  # [batch_size * num_mc_samples, seq_len, channels]

        # Compute interpolated latents in packed space: x_t = (1-t) * x_1 + t * eps (rectified flow)
        t_expanded = t_flat.view(-1, 1, 1)  # [batch_size * num_mc_samples, 1, 1]
        x_t_packed = (1 - t_expanded) * x_1_packed_flat + t_expanded * eps_packed

        # Prepare image IDs using pipeline method
        image_ids = self.flux_pipeline._prepare_latent_image_ids(
            batch_size * num_mc_samples, latent_h // 2, latent_w // 2, self.device, torch.bfloat16
        )

        # Repeat text embeddings to match flattened batch size
        encoder_hidden_states_flat = encoder_hidden_states.repeat_interleave(num_mc_samples, dim=0)
        pooled_prompt_embeds_flat = pooled_prompt_embeds.repeat_interleave(num_mc_samples, dim=0)
        text_ids_flat = text_ids[0]

        # Compute velocity prediction with or without gradients
        def compute_velocity():
                return self.transformer(
                    hidden_states=x_t_packed,
                    encoder_hidden_states=encoder_hidden_states_flat,
                    pooled_projections=pooled_prompt_embeds_flat,
                    timestep=t_flat,
                    img_ids=image_ids,
                    txt_ids=text_ids_flat,
                    guidance=torch.full((batch_size * num_mc_samples,), guidance_scale, device=self.device,
                                        dtype=torch.bfloat16),
                    return_dict=False,
                )[0]

        if use_grad:
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                # For training: keep gradients for policy updates
                velocity_pred = compute_velocity()

                # Compute CFM loss in packed space: ||v_pred - (eps - x_1)||²
                # Loss computation must be inside autocast to maintain dtype consistency during backward
                target_velocity = eps_packed - x_1_packed_flat  # [batch_size * num_mc_samples, seq_len, channels]
                cfm_loss_flat = torch.nn.functional.mse_loss(velocity_pred, target_velocity, reduction='none')
                cfm_loss_flat = cfm_loss_flat.mean(dim=[1, 2])  # [batch_size * num_mc_samples] - mean over seq_len and channels

                # Reshape back to [batch_size, num_mc_samples] and mean over MC samples
                cfm_losses_mc = cfm_loss_flat.view(batch_size, num_mc_samples)  # [batch_size, num_mc_samples]
                cfm_losses = cfm_losses_mc.mean(dim=1)  # [batch_size] - mean over MC samples
        else:
            # For rollout: use no_grad for efficiency
            with torch.no_grad(), \
                torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                velocity_pred = compute_velocity()

                # Compute CFM loss in packed space: ||v_pred - (eps - x_1)||²
                target_velocity = eps_packed - x_1_packed_flat  # [batch_size * num_mc_samples, seq_len, channels]
                cfm_loss_flat = torch.nn.functional.mse_loss(velocity_pred, target_velocity, reduction='none')
                cfm_loss_flat = cfm_loss_flat.mean(dim=[1, 2])  # [batch_size * num_mc_samples] - mean over seq_len and channels

                # Reshape back to [batch_size, num_mc_samples] and mean over MC samples
                cfm_losses_mc = cfm_loss_flat.view(batch_size, num_mc_samples)  # [batch_size, num_mc_samples]
                cfm_losses = cfm_losses_mc.mean(dim=1)  # [batch_size] - mean over MC samples

        if return_velocity:
            return cfm_losses, velocity_pred
        return cfm_losses

    def compute_cfm_losses(
            self,
            final_latents: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            text_ids: torch.Tensor,
            num_mc_samples: int = 100,
            guidance_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute CFM losses for all samples.

        Args:
            final_latents: Final latents [total_samples, C, H, W]
            encoder_hidden_states: Expanded text embeddings [total_samples, seq_len, dim]
            pooled_prompt_embeds: Expanded pooled embeddings [total_samples, dim]
            text_ids: Expanded text IDs [total_samples, seq_len]
            num_mc_samples: Number of Monte Carlo samples
            guidance_scale: Guidance scale

        Returns:
            Tuple of (t_samples, eps_samples, cfm_losses)
        """
        total_samples = final_latents.shape[0]
        IN_CHANNELS = 16
        latent_h, latent_w = final_latents.shape[2], final_latents.shape[3]

        main_print("Computing initial CFM losses...")

        # Sample unique (t, ε) pairs for all samples
        total_mc_samples = num_mc_samples * total_samples
        t_samples_all, eps_samples_all = FpoUtils.sample_cfm_pairs(
            total_mc_samples, (IN_CHANNELS, latent_h, latent_w), self.device
        )
        # Reshape to [total_samples, num_mc_samples, ...]
        t_samples_batch = t_samples_all.view(total_samples, num_mc_samples)
        eps_samples_batch = eps_samples_all.view(total_samples, num_mc_samples, IN_CHANNELS, latent_h, latent_w)

        # Compute CFM losses in batch (with no_grad for old loss computation)
        cfm_losses_batch = self._compute_cfm_loss_batch(
            final_latents,
            t_samples_batch,
            eps_samples_batch,
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            guidance_scale,
            use_grad=False,  # Use no_grad for old loss computation
        )

        return t_samples_batch, eps_samples_batch, cfm_losses_batch.detach()

    def compute_rewards(
            self,
            final_latents: torch.Tensor,
            expanded_captions: List[str],
    ) -> torch.Tensor:
        """
        Compute rewards for all samples.

        Args:
            final_latents: Final latents [total_samples, C, H, W]
            expanded_captions: List of captions for each sample

        Returns:
            Rewards tensor [total_samples]
        """
        main_print("Decoding images and computing rewards...")

        # Decode all images in batch using pipeline's VAE
        final_latents_scaled = (
            final_latents / self.flux_pipeline.vae.config.scaling_factor
        ) + self.flux_pipeline.vae.config.shift_factor
        with torch.no_grad():
            decoded_images = self.flux_pipeline.vae.decode(
                final_latents_scaled.to(self.flux_pipeline.vae.dtype), return_dict=False
            )[0]

        # Process based on reward model type
        reward_model_type = getattr(self.args, 'reward_model_type', 'hpsv3')

        if reward_model_type == "hpsv3":
            # HPSv3 requires PIL images
            image_pil = self.flux_pipeline.image_processor.postprocess(decoded_images, output_type="pil")

            # Handle HPSv3 offloading if enabled
            if hasattr(self.args, 'hpsv3_offload') and self.args.hpsv3_offload:
                self.reward_model.model = self.reward_model.model.to(self.device)
                self.reward_model.device = self.device

            # Process rewards in batches to avoid OOM
            all_rewards = []
            batch_size = self.reward_batch_size if self.reward_batch_size is not None else len(image_pil)

            for i in range(0, len(image_pil), batch_size):
                end_idx = min(i + batch_size, len(image_pil))
                batch_images = image_pil[i:end_idx]
                batch_captions = expanded_captions[i:end_idx]
                with torch.no_grad():
                    batch_rewards = self.reward_model.reward_pil(batch_images, batch_captions)
                batch_rewards = [reward[0] for reward in batch_rewards]
                all_rewards.extend(batch_rewards)

            # Offload back to CPU if enabled
            if hasattr(self.args, 'hpsv3_offload') and self.args.hpsv3_offload:
                self.reward_model.model = self.reward_model.model.to('cpu')
                self.reward_model.device = 'cpu'

            return torch.tensor(all_rewards, device=self.device, dtype=torch.float32)

        elif reward_model_type == "qwen3vl":
            # Qwen3VL discriminator requires tensor images in [-1, 1] range
            # decoded_images is already in [-1, 1] range from VAE

            # Handle Qwen3VL offloading if enabled
            if hasattr(self.args, 'qwen3vl_offload') and self.args.qwen3vl_offload:
                self.reward_model = self.reward_model.to(self.device)

            # Process rewards in batches to avoid OOM
            all_rewards = []
            batch_size = self.reward_batch_size if self.reward_batch_size is not None else len(decoded_images)

            for i in range(0, len(decoded_images), batch_size):
                end_idx = min(i + batch_size, len(decoded_images))
                batch_images = decoded_images[i:end_idx]
                batch_captions = expanded_captions[i:end_idx]

                with torch.no_grad():
                    # Qwen3VL forward returns logits [batch_size, 1]
                    batch_logits = self.reward_model(batch_images, batch_captions)
                    # Use logits as rewards (higher is better)
                    batch_rewards = batch_logits.squeeze(-1)  # [batch_size]
                    all_rewards.append(batch_rewards)

            # Offload back to CPU if enabled
            if hasattr(self.args, 'qwen3vl_offload') and self.args.qwen3vl_offload:
                self.reward_model = self.reward_model.to('cpu')

            # Concatenate all batch rewards
            all_rewards = torch.cat(all_rewards, dim=0)  # [total_samples]
            return all_rewards.to(self.device, dtype=torch.float32)

        else:
            raise ValueError(f"Unsupported reward model type: {reward_model_type}")

    def log_images_to_wandb(
            self,
            rollout_images: torch.Tensor,  # [batch*rollout_n, 3, H, W] in [-1, 1]
            rollout_captions: List[str],
            step: int,
    ):
        """
        Log rollout images to wandb (only on rank 0).

        Args:
            rollout_images: Generated rollout images [batch*rollout_n, 3, H, W] in [-1, 1]
            rollout_captions: Captions for rollout images [batch*rollout_n]
            step: Current training step
        """
        if not dist.is_initialized() or dist.get_rank() != 0:
            return  # Only log on rank 0

        try:
            import wandb
        except ImportError:
            main_print("wandb not installed, skipping image logging")
            return

        # Convert images from [-1, 1] to PIL using image processor
        rollout_pil = self.flux_pipeline.image_processor.postprocess(rollout_images, output_type="pil")

        # Take first few samples to log (e.g., 8 samples)
        num_rollout_to_log = min(8, len(rollout_pil))

        # Create wandb images
        logged_images = []
        for i in range(num_rollout_to_log):
            caption = rollout_captions[i] if i < len(rollout_captions) else ""
            logged_images.append(
                wandb.Image(
                    rollout_pil[i],
                    caption=f"Rollout {i}: {caption}"
                )
            )

        # Log to wandb
        wandb.log({"images": logged_images}, step=step)
        main_print(f"Logged {num_rollout_to_log} rollout images to wandb")

    def train_step(
            self,
            encoder_hidden_states: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            text_ids: torch.Tensor,
            captions: List[str],
            rollout_n: int = 4,
            num_mc_samples: int = 100,
            mini_batch_size: int = 8,
            guidance_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Perform one FPO training step.

        Args:
            encoder_hidden_states: Text embeddings [batch, seq_len, C]
            pooled_prompt_embeds: Pooled embeddings [batch, C]
            text_ids: Text IDs [batch, seq_len, 3]
            captions: List of prompt strings
            rollout_n: Number of samples per prompt
            num_mc_samples: MC samples for CFM loss
            mini_batch_size: Mini-batch size for rollout
            guidance_scale: Guidance scale for generation

        Returns:
            Training metrics dictionary
        """

        batch_size = encoder_hidden_states.shape[0]
        timing_metrics = {}

        main_print(f"FPO Step {self.step_count}: Starting rollout phase")
        rollout_start_time = time.time()
        self.transformer.eval()
        # Phase 1: Rollout - Generate samples only
        final_latents, expanded_captions = self.rollout_phase(
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            captions=captions,
            rollout_n=rollout_n,
            mini_batch_size=mini_batch_size,
            guidance_scale=guidance_scale,
        )
        rollout_time = time.time() - rollout_start_time
        timing_metrics["timing/rollout_phase"] = rollout_time

        # Expand embeddings for CFM computation
        expanded_encoder_hidden_states = encoder_hidden_states.repeat_interleave(rollout_n, dim=0)
        expanded_pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(rollout_n, dim=0)
        expanded_text_ids = text_ids.repeat_interleave(rollout_n, dim=0)

        main_print(f"FPO Step {self.step_count}: Starting CFM loss computation")
        cfm_start_time = time.time()
        # Phase 2: Compute CFM losses
        t_samples_batch, eps_samples_batch, cfm_losses_batch = self.compute_cfm_losses(
            final_latents=final_latents,
            encoder_hidden_states=expanded_encoder_hidden_states,
            pooled_prompt_embeds=expanded_pooled_prompt_embeds,
            text_ids=expanded_text_ids,
            num_mc_samples=num_mc_samples,
            guidance_scale=guidance_scale,
        )
        cfm_time = time.time() - cfm_start_time
        timing_metrics["timing/cfm_computation"] = cfm_time

        main_print(f"FPO Step {self.step_count}: Starting reward computation")
        reward_start_time = time.time()
        # Phase 3: Compute rewards
        rewards = self.compute_rewards(final_latents, expanded_captions)
        reward_time = time.time() - reward_start_time
        timing_metrics["timing/reward_computation"] = reward_time

        # Phase 4: Create FPO batch and compute advantages
        all_sample_infos = []
        total_samples = final_latents.shape[0]
        for i in range(total_samples):
            prompt_idx = i // rollout_n
            sample_idx = i % rollout_n

            sample_info = FpoSampleInfo(
                x_1=final_latents[i].detach().clone(),
                cfm_samples_t=t_samples_batch[i].detach().clone(),
                cfm_samples_eps=eps_samples_batch[i].detach().clone(),
                cfm_samples_loss=cfm_losses_batch[i].detach().clone(),  # Now a scalar value
                initial_noise=None,  # Not needed for policy update
                prompt_idx=prompt_idx,
                sample_idx=sample_idx,
            )
            all_sample_infos.append(sample_info)
        advantages = FpoUtils.compute_advantages_grouped(rewards, all_sample_infos)

        self.transformer.train()
        fpo_batch = FpoBatch(
            sample_infos=all_sample_infos,
            rewards=rewards,
            advantages=advantages,
            prompts=captions,
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
        )

        main_print(f"FPO Step {self.step_count}: Starting policy update phase")
        update_start_time = time.time()

        # Phase 5: Policy Update
        update_metrics = self._update_policy(fpo_batch, guidance_scale)
        update_time = time.time() - update_start_time
        timing_metrics["timing/policy_update"] = update_time

        # Gather metrics from all GPUs for proper logging
        gathered_rewards = gather_tensor(rewards)

        # Calculate total time
        total_time = rollout_time + cfm_time + reward_time + update_time
        timing_metrics["timing/total_step"] = total_time

        # Combine metrics (computed from all GPUs)
        metrics = {
            "rollout/mean_reward": gathered_rewards.mean().item(),
            "rollout/max_reward": gathered_rewards.max().item(),
            "rollout/min_reward": gathered_rewards.min().item(),
            "rollout/std_reward": gathered_rewards.std().item(),
            "rollout/num_samples": gathered_rewards.shape[0],
            **update_metrics,
            **timing_metrics,
        }

        # Log to wandb
        if dist.get_rank() == 0:
            wandb.log(metrics, step=self.step_count)

        # Log images if enabled
        if self.image_log_interval > 0 and self.step_count % self.image_log_interval == 0 and self.step_count > 0:
            main_print(f"[Step {self.step_count}] Logging images to wandb")
            # Decode latents to images
            final_latents_scaled = (
                final_latents / self.flux_pipeline.vae.config.scaling_factor
            ) + self.flux_pipeline.vae.config.shift_factor
            with torch.no_grad():
                rollout_images = self.flux_pipeline.vae.decode(
                    final_latents_scaled.to(self.flux_pipeline.vae.dtype), return_dict=False
                )[0]
            self.log_images_to_wandb(
                rollout_images=rollout_images,
                rollout_captions=expanded_captions,
                step=self.step_count,
            )

        main_print(f"FPO Step {self.step_count} complete. Mean reward: {metrics['rollout/mean_reward']:.4f}")

        self.step_count += 1

        return metrics

    def _compute_reference_velocity(
            self,
            x_1: torch.Tensor,
            t_samples: torch.Tensor,
            eps_samples: torch.Tensor,
            encoder_hidden_states: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            text_ids: torch.Tensor,
            guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Compute reference model velocity predictions without gradients.

        Args:
            x_1: Final denoised images [batch_size, C, H, W]
            t_samples: Sampled timesteps [batch_size, num_mc_samples]
            eps_samples: Sampled noise [batch_size, num_mc_samples, C, H, W]
            encoder_hidden_states: Text embeddings [batch_size, seq_len, dim]
            pooled_prompt_embeds: Pooled embeddings [batch_size, dim]
            text_ids: Text IDs [batch_size, seq_len]
            guidance_scale: Guidance scale

        Returns:
            Reference velocity predictions [batch_size * num_mc_samples, seq_len, channels]
        """
        if self.reference_transformer is None:
            raise ValueError("Reference transformer is not set. Cannot compute reference velocity.")

        # Check if reference model is on CPU and needs to be moved to GPU
        reference_on_cpu = next(self.reference_transformer.parameters()).device.type == 'cpu'
        if reference_on_cpu:
            self.reference_transformer = self.reference_transformer.to(self.device)

        batch_size, num_mc_samples = t_samples.shape
        _, C, latent_h, latent_w = x_1.shape

        # Pack latents first
        x_1_packed = self.flux_pipeline._pack_latents(x_1, batch_size, C, latent_h, latent_w)

        # Flatten batch and MC dimensions
        t_flat = t_samples.flatten()
        eps_flat = eps_samples.view(batch_size * num_mc_samples, C, latent_h, latent_w)

        # Pack noise samples
        eps_packed = self.flux_pipeline._pack_latents(eps_flat, batch_size * num_mc_samples, C, latent_h, latent_w)

        # Repeat packed x_1 to match flattened batch size
        x_1_packed_flat = x_1_packed.repeat_interleave(num_mc_samples, dim=0)

        # Compute interpolated latents in packed space
        t_expanded = t_flat.view(-1, 1, 1)
        x_t_packed = (1 - t_expanded) * x_1_packed_flat + t_expanded * eps_packed

        # Prepare image IDs
        image_ids = self.flux_pipeline._prepare_latent_image_ids(
            batch_size * num_mc_samples, latent_h // 2, latent_w // 2, self.device, torch.bfloat16
        )

        # Repeat text embeddings
        encoder_hidden_states_flat = encoder_hidden_states.repeat_interleave(num_mc_samples, dim=0)
        pooled_prompt_embeds_flat = pooled_prompt_embeds.repeat_interleave(num_mc_samples, dim=0)
        text_ids_flat = text_ids[0]

        # Compute reference velocity without gradients using mixed precision
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                velocity_pred = self.reference_transformer(
                    hidden_states=x_t_packed,
                    encoder_hidden_states=encoder_hidden_states_flat,
                    pooled_projections=pooled_prompt_embeds_flat,
                    timestep=t_flat,
                    img_ids=image_ids,
                    txt_ids=text_ids_flat,
                    guidance=torch.full((batch_size * num_mc_samples,), guidance_scale, device=self.device,
                                        dtype=torch.bfloat16),
                    return_dict=False,
                )[0]

        # Move reference model back to CPU if it was originally on CPU
        if reference_on_cpu:
            self.reference_transformer = self.reference_transformer.to('cpu')
            torch.cuda.empty_cache()

        return velocity_pred

    def _update_policy(self, fpo_batch: FpoBatch, guidance_scale: float) -> Dict[str, float]:
        """
        Update policy using FPO loss with gradient accumulation.

        Args:
            fpo_batch: Batch of FPO samples
            guidance_scale: Guidance scale for generation

        Returns:
            Update metrics dictionary
        """

        total_samples = len(fpo_batch.sample_infos)

        # Prepare data for updates
        all_x1 = torch.stack([info.x_1 for info in fpo_batch.sample_infos])  # [total_samples, C, H, W]
        all_t_samples = torch.stack([info.cfm_samples_t for info in fpo_batch.sample_infos])  # [total_samples, num_mc]
        all_eps_samples = torch.stack(
            [info.cfm_samples_eps for info in fpo_batch.sample_infos])  # [total_samples, num_mc, C, H, W]
        all_old_cfm_losses = torch.stack(
            [info.cfm_samples_loss for info in fpo_batch.sample_infos])  # [total_samples] - scalar losses

        # Expand embeddings to match sample count
        expanded_encoder_hidden_states = fpo_batch.encoder_hidden_states.repeat_interleave(
            total_samples // fpo_batch.encoder_hidden_states.shape[0], dim=0
        )
        expanded_pooled_prompt_embeds = fpo_batch.pooled_prompt_embeds.repeat_interleave(
            total_samples // fpo_batch.pooled_prompt_embeds.shape[0], dim=0
        )
        expanded_text_ids = fpo_batch.text_ids

        # Compute reference velocities once before training (if KL regularization is enabled)
        all_reference_velocities = None
        if self.reference_transformer is not None and self.kl_beta > 0:
            main_print("Computing reference velocities for KL regularization...")
            all_reference_velocities = self._compute_reference_velocity(
                all_x1, all_t_samples, all_eps_samples,
                expanded_encoder_hidden_states, expanded_pooled_prompt_embeds,
                expanded_text_ids, guidance_scale,
            )

        # Training metrics
        policy_losses = []
        kl_losses = []
        combined_losses = []
        fpo_ratios_mean = []
        fpo_ratios_max = []
        fpo_ratios_min = []
        advantages_list_mean = []
        advantages_list_max = []
        advantages_list_min = []
        clipped_ratios = []

        # Multiple epochs of updates
        for epoch in range(self.num_epochs):
            # Shuffle samples for each epoch
            indices = torch.randperm(total_samples, device=self.device)
            # Process all samples at once
            self.optimizer.zero_grad()

            # Extract sample mask (if provided)
            sample_weight = None
            if fpo_batch.sample_mask is not None:
                sample_weight = fpo_batch.sample_mask[indices].float()
                actual_batch_size = sample_weight.sum().item()
            else:
                actual_batch_size = total_samples

            # Compute new CFM losses (with gradients for training)
            if all_reference_velocities is not None:
                batch_new_cfm_losses, batch_velocity_pred = self._compute_cfm_loss_batch(
                    all_x1[indices],
                    all_t_samples[indices],
                    all_eps_samples[indices],
                    expanded_encoder_hidden_states[indices],
                    expanded_pooled_prompt_embeds[indices],
                    expanded_text_ids,
                    guidance_scale,
                    use_grad=True,
                    return_velocity=True,
                )
            else:
                batch_new_cfm_losses = self._compute_cfm_loss_batch(
                    all_x1[indices],
                    all_t_samples[indices],
                    all_eps_samples[indices],
                    expanded_encoder_hidden_states[indices],
                    expanded_pooled_prompt_embeds[indices],
                    expanded_text_ids,
                    guidance_scale,
                    use_grad=True,
                )

            # Compute FPO ratios
            fpo_ratio = FpoUtils.compute_fpo_ratio_online(all_old_cfm_losses[indices], batch_new_cfm_losses).float()

            # Clip advantages
            clipped_advantages = torch.clamp(fpo_batch.advantages[indices], -self.adv_clip_max, self.adv_clip_max)

            # Compute per-sample surrogate loss
            per_sample_loss = - fpo_ratio * clipped_advantages
            clipped_per_sample_loss = - torch.clamp(fpo_ratio, 1 - self.clip_range,
                                                    1 + self.clip_range) * clipped_advantages
            per_sample_policy_loss = torch.maximum(per_sample_loss, clipped_per_sample_loss)

            # Apply sample mask
            if sample_weight is not None:
                per_sample_policy_loss = per_sample_policy_loss * sample_weight

            # Reduce to scalar
            if sample_weight is not None:
                policy_loss = per_sample_policy_loss.sum() / (sample_weight.sum() + 1e-8)
            else:
                policy_loss = per_sample_policy_loss.mean()

            # Compute KL loss if KL regularization is enabled
            if all_reference_velocities is not None:
                num_mc_samples = all_t_samples.shape[1]

                velocity_indices = []
                for idx in indices:
                    for mc_idx in range(num_mc_samples):
                        velocity_indices.append(idx * num_mc_samples + mc_idx)
                velocity_indices = torch.tensor(velocity_indices, device=self.device)

                batch_reference_velocity = all_reference_velocities[velocity_indices]

                per_sample_kl = ((batch_velocity_pred.view(total_samples, num_mc_samples, -1) -
                                  batch_reference_velocity.view(total_samples, num_mc_samples, -1)) ** 2).mean(dim=[1, 2])

                if sample_weight is not None:
                    per_sample_kl = per_sample_kl * sample_weight

                if sample_weight is not None:
                    kl_loss = per_sample_kl.sum() / (sample_weight.sum() + 1e-8)
                else:
                    kl_loss = per_sample_kl.mean()
            else:
                kl_loss = torch.tensor(0.0, device=self.device)

            # Combine policy loss and KL loss
            combined_loss = policy_loss + self.kl_beta * kl_loss

            # Backward pass
            combined_loss.backward()

            # Track metrics
            accumulated_loss = policy_loss.item() * actual_batch_size
            samples_processed = actual_batch_size

            if all_reference_velocities is not None:
                kl_losses.append(kl_loss.item())
                combined_losses.append(combined_loss.item())

            fpo_ratios_mean.append(fpo_ratio.mean().item())
            fpo_ratios_max.append(fpo_ratio.max().item())
            fpo_ratios_min.append(fpo_ratio.min().item())
            advantages_list_mean.append(clipped_advantages.mean().item())
            advantages_list_max.append(clipped_advantages.max().item())
            advantages_list_min.append(clipped_advantages.min().item())
            clipped_ratios.append((torch.abs(fpo_ratio - 1.0) > self.clip_range).float().mean().item())

            # Gradient clipping and optimizer step
            if samples_processed > 0:
                grad_norm = nn.utils.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.optimizer.zero_grad()
            else:
                # If no samples were processed, we still need to zero gradients to clean up
                self.optimizer.zero_grad()
                grad_norm = torch.tensor(0.0, device=self.device)

            # Store epoch-level metrics
            if samples_processed > 0:
                policy_losses.append(accumulated_loss / (samples_processed))

            dist.barrier()

        # Gather metrics from all GPUs
        local_metrics = torch.tensor([
            np.mean(policy_losses) if policy_losses else 0.0,
            np.mean(kl_losses) if kl_losses else 0.0,
            np.mean(combined_losses) if combined_losses else 0.0,
            np.mean(fpo_ratios_mean) if fpo_ratios_mean else 1.0,
            np.max(fpo_ratios_max) if fpo_ratios_max else 1.0,
            np.min(fpo_ratios_min) if fpo_ratios_min else 1.0,
            np.mean(advantages_list_mean) if advantages_list_mean else 0.0,
            np.max(advantages_list_max) if advantages_list_max else 0.0,
            np.min(advantages_list_min) if advantages_list_min else 0.0,
            np.mean(clipped_ratios) if clipped_ratios else 0.0,
            grad_norm.item() if 'grad_norm' in locals() else 0.0,
        ], device=self.device)

        gathered_metrics = gather_tensor(local_metrics)

        # Compute global metrics (only used on rank 0)
        if dist.get_rank() == 0:
            # Reshape to [world_size, 11] for proper indexing (added kl_loss and combined_loss)
            world_size = dist.get_world_size()
            gathered_metrics = gathered_metrics.view(world_size, 11)

            global_policy_loss = gathered_metrics[:, 0].mean().item()
            global_kl_loss = gathered_metrics[:, 1].mean().item()
            global_combined_loss = gathered_metrics[:, 2].mean().item()
            global_fpo_ratio_mean = gathered_metrics[:, 3].mean().item()
            global_fpo_ratio_max = gathered_metrics[:, 4].max().item()
            global_fpo_ratio_min = gathered_metrics[:, 5].min().item()
            global_advantage_mean = gathered_metrics[:, 6].mean().item()
            global_advantage_max = gathered_metrics[:, 7].max().item()
            global_advantage_min = gathered_metrics[:, 8].min().item()
            global_clipped_ratio = gathered_metrics[:, 9].mean().item()
            global_grad_norm = gathered_metrics[:, 10].mean().item()
        else:
            global_policy_loss = 0.0
            global_kl_loss = 0.0
            global_combined_loss = 0.0
            global_fpo_ratio_mean = 1.0
            global_fpo_ratio_max = 1.0
            global_fpo_ratio_min = 1.0
            global_advantage_mean = 0.0
            global_advantage_max = 0.0
            global_advantage_min = 0.0
            global_clipped_ratio = 0.0
            global_grad_norm = 0.0

        update_metrics = {
            "update/policy_loss": global_policy_loss,
            "update/kl_loss": global_kl_loss,
            "update/combined_loss": global_combined_loss,
            "update/fpo_ratio_mean": global_fpo_ratio_mean,
            "update/fpo_ratio_max": global_fpo_ratio_max,
            "update/fpo_ratio_min": global_fpo_ratio_min,
            "update/advantage_mean": global_advantage_mean,
            "update/advantage_max": global_advantage_max,
            "update/advantage_min": global_advantage_min,
            "update/clipped_ratio_fraction": global_clipped_ratio,
            "update/grad_norm": global_grad_norm,
        }

        return update_metrics

    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        checkpoint = {
            'transformer_state_dict': self.transformer.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'step_count': self.step_count,
            'args': self.args,
        }
        torch.save(checkpoint, path)
        main_print(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.transformer.load_state_dict(checkpoint['transformer_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step_count = checkpoint['step_count']
        main_print(f"Checkpoint loaded from {path}, step: {self.step_count}")

        # Update wandb's step counter if wandb is being used
        try:
            import wandb
            if wandb.run is not None:
                wandb.run._step = self.step_count
                main_print(f"Updated wandb step counter to: {self.step_count}")
        except ImportError:
            pass

    def get_training_state(self) -> Dict[str, Any]:
        """Get current training state."""
        return {
            'step_count': self.step_count,
            'clip_range': self.clip_range,
            'adv_clip_max': self.adv_clip_max,
            'max_grad_norm': self.max_grad_norm,
            'num_epochs': self.num_epochs,
        }
