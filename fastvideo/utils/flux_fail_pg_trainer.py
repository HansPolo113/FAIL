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
from typing import List, Dict, Any, Tuple
import numpy as np
import time
import logging
from contextlib import contextmanager

from ..models.fpo_utils import FpoBatch, FpoUtils, FpoSampleInfo
from diffusers.image_processor import VaeImageProcessor
from .fpo_trainer import FpoTrainer, gather_tensor

from .logging_ import main_print

logger = logging.getLogger(__name__)


class FluxFailPGTrainer(FpoTrainer):
    """
    FAIL-PG (Policy Gradient) trainer for FLUX that extends the base FpoTrainer with discriminator-based rewards.
    Keeps operations simple and modular.
    """

    def __init__(
            self,
            transformer,
            vae,
            flux_pipeline,
            discriminator,
            discriminator_optimizer,
            discriminator_scheduler,
            reward_model,  # Keep for monitoring purposes
            optimizer,
            device: torch.device,
            args,
            clip_range: float = 0.2,
            adv_clip_max: float = 5.0,
            max_grad_norm: float = 1.0,
            num_epochs: int = 4,
            discriminator_updates_per_step: int = 1,
            include_teacher_in_policy: bool = True,
            discriminator_grad_clip: float = 1.0,
            discriminator_warmup_steps: int = 0,
            reward_batch_size: int = None,
            reference_transformer=None,
            kl_beta: float = 0.0,
            image_log_interval: int = 50,  # Log images every N steps (0 to disable)
    ):
        """
        Initialize GAIL FPO trainer for FLUX.
        """
        super().__init__(
            transformer, vae, flux_pipeline, reward_model, optimizer, device, args,
            clip_range, adv_clip_max, max_grad_norm, num_epochs, reward_batch_size,
            reference_transformer, kl_beta
        )

        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler

        self.discriminator_updates_per_step = discriminator_updates_per_step
        self.include_teacher_in_policy = include_teacher_in_policy
        self.discriminator_grad_clip = discriminator_grad_clip
        self.discriminator_warmup_steps = discriminator_warmup_steps
        self.image_log_interval = image_log_interval

    @contextmanager
    def _use_unwrapped_transformer_for_inference(self):
        """Context manager to use unwrapped transformer for faster inference during rollout.

        Uses FSDP.summon_full_params to temporarily gather all sharded parameters,
        allowing direct access to the underlying module without FSDP overhead.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        is_fsdp = isinstance(self.transformer, FSDP)

        if is_fsdp:
            original_transformer = self.flux_pipeline.transformer

            with FSDP.summon_full_params(self.transformer, writeback=False, rank0_only=False):
                self.flux_pipeline.transformer = self.transformer.module
                try:
                    yield
                finally:
                    self.flux_pipeline.transformer = original_transformer
        else:
            yield

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

        Overrides parent method to use FSDP unwrapping context manager for faster inference.

        Args:
            encoder_hidden_states: Text embeddings [batch, seq_len, C]
            pooled_prompt_embeds: Pooled embeddings [batch, C]
            text_ids: Text IDs [batch, seq_len]
            captions: List of prompt strings
            rollout_n: Number of samples per prompt
            mini_batch_size: Mini-batch size for rollout
            guidance_scale: Guidance scale for generation
            rollout_negative_promots: List of negative prompts
            rollout_true_cfg: True CFG scale

        Returns:
            Tuple of (final_latents, expanded_captions)
        """
        SPATIAL_DOWNSAMPLE = 8
        IN_CHANNELS = 16
        w, h = self.args.width, self.args.height
        latent_w, latent_h = w // SPATIAL_DOWNSAMPLE, h // SPATIAL_DOWNSAMPLE

        batch_size = encoder_hidden_states.shape[0]
        total_samples = batch_size * rollout_n
        assert total_samples % mini_batch_size == 0, \
            f"total_samples ({total_samples}) must be divisible by mini_batch_size ({mini_batch_size})"

        expanded_encoder_hidden_states = encoder_hidden_states.repeat_interleave(rollout_n, dim=0)
        expanded_pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(rollout_n, dim=0)
        expanded_text_ids = text_ids.repeat_interleave(rollout_n, dim=0)
        expanded_captions = [caption for caption in captions for _ in range(rollout_n)]
        if rollout_negative_promots:
            expanded_rollout_negative_promots = [p for p in rollout_negative_promots for _ in range(rollout_n)]
        else:
            expanded_rollout_negative_promots = ['' for _ in captions for _ in range(rollout_n)]

        all_final_latents = []

        with torch.no_grad(), \
             torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for mini_start in range(0, total_samples, mini_batch_size):
                mini_end = min(mini_start + mini_batch_size, total_samples)
                mini_batch_actual_size = mini_end - mini_start

                mini_encoder_hidden_states = expanded_encoder_hidden_states[mini_start:mini_end]
                mini_pooled_prompt_embeds = expanded_pooled_prompt_embeds[mini_start:mini_end]
                mini_text_ids = expanded_text_ids[mini_start:mini_end]
                mini_captions = expanded_captions[mini_start:mini_end]
                mini_rollout_negative_promots = expanded_rollout_negative_promots[mini_start:mini_end]

                if self.args.init_same_noise:
                    base_noise = torch.randn(
                        (1, IN_CHANNELS, latent_h, latent_w),
                        device=self.device,
                        dtype=torch.bfloat16
                    )
                    mini_initial_noise = base_noise.repeat(mini_batch_actual_size, 1, 1, 1)
                else:
                    mini_initial_noise = torch.randn(
                        (mini_batch_actual_size, IN_CHANNELS, latent_h, latent_w),
                        device=self.device,
                        dtype=torch.bfloat16
                    )

                pipeline_output = self.flux_pipeline(
                    height=h,
                    width=w,
                    num_inference_steps=self.args.num_sampling_steps,
                    guidance_scale=guidance_scale,
                    output_type="latent",
                    return_dict=False,
                    prompt_embeds=mini_encoder_hidden_states,
                    pooled_prompt_embeds=mini_pooled_prompt_embeds,
                    latents=self.flux_pipeline._pack_latents(
                        mini_initial_noise, mini_batch_actual_size, IN_CHANNELS, latent_h, latent_w
                    ),
                    negative_prompt=mini_rollout_negative_promots,
                    true_cfg_scale=rollout_true_cfg,
                )

                final_latents_packed = pipeline_output[0]
                final_latents_unpacked = self.flux_pipeline._unpack_latents(
                    final_latents_packed, h, w, self.flux_pipeline.vae_scale_factor
                )

                all_final_latents.append(final_latents_unpacked.to(dtype=mini_encoder_hidden_states.dtype))

        all_final_latents = torch.cat(all_final_latents, dim=0)

        return all_final_latents, expanded_captions

    def encode_teacher_images_to_latents(self, teacher_images: torch.Tensor) -> torch.Tensor:
        """
        Encode teacher images to latents using VAE.

        Args:
            teacher_images: Teacher images [batch_size, teacher_n, 3, H, W]

        Returns:
            Teacher latents [batch_size*teacher_n, C, latent_H, latent_W]
        """
        batch_size, teacher_n = teacher_images.shape[:2]
        teacher_images_flat = teacher_images.view(-1, *teacher_images.shape[2:])
        teacher_images_device = teacher_images_flat.to(self.device)

        with torch.no_grad():
            teacher_latents = self.flux_pipeline.vae.encode(
                teacher_images_device.to(self.flux_pipeline.vae.dtype)).latent_dist.sample()
            teacher_latents = (
                                          teacher_latents - self.flux_pipeline.vae.config.shift_factor) * self.flux_pipeline.vae.config.scaling_factor

        return teacher_latents

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images using VAE.

        Args:
            latents: Latents [batch_size, C, latent_H, latent_W]

        Returns:
            Images [batch_size, 3, H, W]
        """
        with torch.no_grad():
            latents_scaled = (
                                         latents / self.flux_pipeline.vae.config.scaling_factor) + self.flux_pipeline.vae.config.shift_factor
            images = self.flux_pipeline.vae.decode(latents_scaled.to(self.flux_pipeline.vae.dtype), return_dict=False)[
                0]

        return images

    def update_discriminator(
            self,
            rollout_images: torch.Tensor,
            teacher_images: torch.Tensor,
            rollout_prompts: List[str],
            teacher_prompts: List[str],
    ) -> Dict[str, float]:
        """
        Update discriminator using rollout and teacher images with QwenVL discriminator.

        Args:
            rollout_images: Generated images [batch*rollout_n, 3, H, W]
            teacher_images: Teacher images [batch*teacher_n, 3, H, W]
            rollout_prompts: Prompts for rollout images [batch*rollout_n]
            teacher_prompts: Prompts for teacher images [batch*teacher_n]

        Returns:
            Dictionary of discriminator training metrics
        """
        self.discriminator.train()
        self.discriminator_optimizer.zero_grad()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rollout_batch_size = rollout_images.shape[0]
            teacher_batch_size = teacher_images.shape[0]

            all_images = torch.cat([rollout_images, teacher_images], dim=0)
            all_prompts = rollout_prompts + teacher_prompts
            all_logits = self.discriminator(all_images, all_prompts)

        gen_logits = all_logits[:rollout_batch_size]
        teacher_logits = all_logits[rollout_batch_size:]

        gen_targets = torch.ones_like(gen_logits)
        teacher_targets = torch.zeros_like(teacher_logits)

        gen_loss = torch.nn.functional.binary_cross_entropy_with_logits(gen_logits, gen_targets)
        teacher_loss = torch.nn.functional.binary_cross_entropy_with_logits(teacher_logits, teacher_targets)
        total_loss = gen_loss + teacher_loss

        total_loss.backward()

        if isinstance(self.discriminator, torch.distributed.fsdp.FullyShardedDataParallel):
            grad_norm = self.discriminator.clip_grad_norm_(self.discriminator_grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.discriminator.parameters(),
                max_norm=self.discriminator_grad_clip
            )

        if not torch.isfinite(grad_norm):
            logger.warning(f"Discriminator grad_norm is not finite: {grad_norm}")
            self.discriminator_optimizer.zero_grad()
        else:
            self.discriminator_optimizer.step()
            if self.discriminator_scheduler is not None:
                self.discriminator_scheduler.step()

        self.discriminator_optimizer.zero_grad()

        with torch.no_grad():
            gen_acc = ((torch.sigmoid(gen_logits) > 0.5) == gen_targets).float().mean()
            teacher_acc = ((torch.sigmoid(teacher_logits) > 0.5) == teacher_targets).float().mean()
            overall_acc = (gen_acc + teacher_acc) / 2

        metrics = {
            "discriminator/loss": total_loss.detach().item(),
            "discriminator/rollout_loss": gen_loss.detach().item(),
            "discriminator/teacher_loss": teacher_loss.detach().item(),
            "discriminator/rollout_acc": gen_acc.detach().item(),
            "discriminator/teacher_acc": teacher_acc.detach().item(),
            "discriminator/overall_acc": overall_acc.detach().item(),
            "discriminator/rollout_logits_mean": gen_logits.mean().detach().item(),
            "discriminator/teacher_logits_mean": teacher_logits.mean().detach().item(),
            "discriminator/grad_norm": grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm,
        }

        return metrics

    def compute_gail_rewards(
            self,
            rollout_images: torch.Tensor,
            teacher_images: torch.Tensor,
            rollout_prompts: List[str],
            teacher_prompts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
        """
        Compute GAIL rewards using the QwenVL discriminator.

        Args:
            rollout_images: Generated images [batch*rollout_n, 3, H, W]
            teacher_images: Teacher images [batch*teacher_n, 3, H, W]
            rollout_prompts: Prompts for rollout images [batch*rollout_n]
            teacher_prompts: Prompts for teacher images [batch*teacher_n]

        Returns:
            Tuple of (gen_rewards, teacher_rewards, metrics_dict)
        """
        self.discriminator.eval()

        rollout_batch_size = rollout_images.shape[0]
        teacher_batch_size = teacher_images.shape[0]

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                all_images = torch.cat([rollout_images, teacher_images], dim=0)
                all_prompts = rollout_prompts + teacher_prompts
                all_logits = self.discriminator(all_images, all_prompts)

        gen_logits = all_logits[:rollout_batch_size]
        teacher_logits = all_logits[rollout_batch_size:]

        gen_rewards = -torch.nn.functional.logsigmoid(gen_logits.mean(-1))
        teacher_rewards = -torch.nn.functional.logsigmoid(teacher_logits.mean(-1))

        with torch.no_grad():
            gen_targets = torch.ones_like(gen_logits)
            teacher_targets = torch.zeros_like(teacher_logits)

            gen_acc = ((torch.sigmoid(gen_logits) > 0.5) == gen_targets).float().mean()
            teacher_acc = ((torch.sigmoid(teacher_logits) > 0.5) == teacher_targets).float().mean()
            overall_acc = (gen_acc + teacher_acc) / 2

            metrics = {
                "discriminator_eval/rollout_logits_mean": gen_logits.mean().item(),
                "discriminator_eval/teacher_logits_mean": teacher_logits.mean().item(),
                "discriminator_eval/rollout_acc": gen_acc.item(),
                "discriminator_eval/teacher_acc": teacher_acc.item(),
                "discriminator_eval/overall_acc": overall_acc.item(),
            }

        return gen_rewards, teacher_rewards, metrics

    def compute_monitoring_rewards(self, images: torch.Tensor, captions: List[str]) -> torch.Tensor:
        """
        Compute HPS rewards for monitoring purposes.

        Args:
            images: Images [batch_size, 3, H, W]
            captions: List of captions

        Returns:
            HPS rewards
        """
        image_pil = self.flux_pipeline.image_processor.postprocess(images, output_type="pil")

        if hasattr(self.args, 'hpsv3_offload') and self.args.hpsv3_offload:
            self.reward_model.model = self.reward_model.model.to(self.device)
            self.reward_model.device = self.device
        with torch.no_grad():
            all_rewards = self.reward_model.reward_pil(image_pil, captions)
        all_rewards = [all_reward[0] for all_reward in all_rewards]

        if hasattr(self.args, 'hpsv3_offload') and self.args.hpsv3_offload:
            self.reward_model.model = self.reward_model.model.to('cpu')
            self.reward_model.device = 'cpu'

        return torch.tensor(all_rewards, device=self.device, dtype=torch.float32)

    def log_images_to_wandb(
            self,
            rollout_images: torch.Tensor,  # [batch*rollout_n, 3, H, W] in [-1, 1]
            teacher_images: torch.Tensor,  # [batch*teacher_n, 3, H, W] in [-1, 1]
            rollout_captions: List[str],
            teacher_captions: List[str],
            step: int,
    ):
        """
        Log images to wandb (only on rank 0).
        Logs all rollout images first, then all teacher images separately.

        Args:
            rollout_images: Generated rollout images [batch*rollout_n, 3, H, W] in [-1, 1]
            teacher_images: Teacher images [batch*teacher_n, 3, H, W] in [-1, 1]
            rollout_captions: Captions for rollout images [batch*rollout_n]
            teacher_captions: Captions for teacher images [batch*teacher_n]
            step: Current training step
        """
        if not dist.is_initialized() or dist.get_rank() != 0:
            return

        try:
            import wandb
        except ImportError:
            main_print("wandb not installed, skipping image logging")
            return

        rollout_pil = self.flux_pipeline.image_processor.postprocess(rollout_images, output_type="pil")
        teacher_pil = self.flux_pipeline.image_processor.postprocess(teacher_images, output_type="pil")

        num_rollout_to_log = min(8, len(rollout_pil))
        num_teacher_to_log = min(8, len(teacher_pil))

        logged_images = []

        for i in range(num_rollout_to_log):
            caption = rollout_captions[i] if i < len(rollout_captions) else ""
            logged_images.append(
                wandb.Image(
                    rollout_pil[i],
                    caption=f"Rollout {i}: {caption}"
                )
            )

        for i in range(num_teacher_to_log):
            caption = teacher_captions[i] if i < len(teacher_captions) else ""
            logged_images.append(
                wandb.Image(
                    teacher_pil[i],
                    caption=f"Teacher {i}: {caption}"
                )
            )

        wandb.log({"images": logged_images}, step=step)


    def interleave_rollout_and_teacher_by_prompt(
            self,
            rollout_data: torch.Tensor,
            teacher_data: torch.Tensor,
            batch_size: int,
            rollout_n: int,
            teacher_n: int,
    ) -> torch.Tensor:
        """
        Interleave rollout and teacher data by prompt to maintain grouping.

        Args:
            rollout_data: Rollout data [batch_size*rollout_n, ...]
            teacher_data: Teacher data [batch_size*teacher_n, ...]
            batch_size: Number of prompts
            rollout_n: Number of rollout samples per prompt
            teacher_n: Number of teacher samples per prompt

        Returns:
            Combined data with proper grouping [batch_size*(rollout_n+teacher_n), ...]
        """
        rollout_reshaped = rollout_data.view(batch_size, rollout_n, *rollout_data.shape[1:])
        teacher_reshaped = teacher_data.view(batch_size, teacher_n, *teacher_data.shape[1:])
        combined_reshaped = torch.cat([rollout_reshaped, teacher_reshaped], dim=1)
        combined_flat = combined_reshaped.view(-1, *rollout_data.shape[1:])

        return combined_flat

    def train_step(
            self,
            encoder_hidden_states: torch.Tensor,
            pooled_prompt_embeds: torch.Tensor,
            text_ids: torch.Tensor,
            teacher_images: torch.Tensor,
            captions: List[str],
            rollout_n: int = 4,
            teacher_n: int = None,
            num_mc_samples: int = 100,
            mini_batch_size: int = 8,
            guidance_scale: float = 1.0,
            rollout_negative_promots: List[str] = None,
            rollout_true_cfg: float = 1.0,
    ) -> Dict[str, float]:
        """
        Perform one GAIL FPO training step.
        """
        batch_size = encoder_hidden_states.shape[0]

        if teacher_n is None:
            teacher_n = teacher_images.shape[1]

        timing_metrics = {}
        in_warmup = self.step_count < self.discriminator_warmup_steps

        rollout_start_time = time.time()
        self.transformer.eval()

        final_latents, expanded_captions = self.rollout_phase(
            encoder_hidden_states=encoder_hidden_states,
            pooled_prompt_embeds=pooled_prompt_embeds,
            text_ids=text_ids,
            captions=captions,
            rollout_n=rollout_n,
            mini_batch_size=mini_batch_size,
            guidance_scale=guidance_scale,
            rollout_negative_promots=rollout_negative_promots,
            rollout_true_cfg=rollout_true_cfg
        )
        rollout_time = time.time() - rollout_start_time
        timing_metrics["time/rollout"] = rollout_time

        torch.cuda.empty_cache()

        teacher_start_time = time.time()

        teacher_latents = self.encode_teacher_images_to_latents(teacher_images)
        teacher_time = time.time() - teacher_start_time
        timing_metrics["time/teacher"] = teacher_time

        decode_start_time = time.time()

        rollout_images = self.decode_latents_to_images(final_latents)
        teacher_images_flat = teacher_images.view(-1, *teacher_images.shape[2:])

        decode_time = time.time() - decode_start_time
        timing_metrics["time/decode"] = decode_time

        teacher_expanded_captions = [caption for caption in captions for _ in range(teacher_n)]

        discriminator_start_time = time.time()

        discriminator_metrics = self.update_discriminator(
            rollout_images=rollout_images,
            teacher_images=teacher_images_flat,
            rollout_prompts=expanded_captions,
            teacher_prompts=teacher_expanded_captions,
        )

        for key, value in discriminator_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    discriminator_metrics[key] = gather_tensor(value.unsqueeze(0)).mean().item()
                else:
                    discriminator_metrics[key] = gather_tensor(value).mean().item()
            elif isinstance(value, (int, float)):
                scalar_tensor = torch.tensor([value], device=self.device)
                discriminator_metrics[key] = gather_tensor(scalar_tensor).mean().item()

        discriminator_time = time.time() - discriminator_start_time
        timing_metrics["time/discriminator"] = discriminator_time

        reward_start_time = time.time()

        gen_rewards, teacher_rewards, discriminator_eval_metrics = self.compute_gail_rewards(
            rollout_images=rollout_images,
            teacher_images=teacher_images_flat,
            rollout_prompts=expanded_captions,
            teacher_prompts=teacher_expanded_captions,
        )

        for key, value in discriminator_eval_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    discriminator_eval_metrics[key] = gather_tensor(value.unsqueeze(0)).mean().item()
                else:
                    discriminator_eval_metrics[key] = gather_tensor(value).mean().item()
            elif isinstance(value, (int, float)):
                scalar_tensor = torch.tensor([value], device=self.device)
                discriminator_eval_metrics[key] = gather_tensor(scalar_tensor).mean().item()

        hps_gen_rewards = None
        hps_teacher_rewards = None
        if self.reward_model is not None:
            hps_gen_rewards = self.compute_monitoring_rewards(rollout_images, expanded_captions)
            hps_teacher_rewards = self.compute_monitoring_rewards(teacher_images_flat, teacher_expanded_captions)
        reward_time = time.time() - reward_start_time
        timing_metrics["time/reward"] = reward_time
        dist.barrier()

        if in_warmup:
            total_time = rollout_time + teacher_time + decode_time + discriminator_time + reward_time
            timing_metrics["time/total"] = total_time

            gathered_gen_rewards_warmup = gather_tensor(gen_rewards)
            gathered_teacher_rewards_warmup = gather_tensor(teacher_rewards)

            metrics = {
                "rollout/mean_gail_rollout_reward": gathered_gen_rewards_warmup.mean().item(),
                "rollout/mean_gail_teacher_reward": gathered_teacher_rewards_warmup.mean().item(),
                "rollout/max_gail_rollout_reward": gathered_gen_rewards_warmup.max().item(),
                "rollout/min_gail_rollout_reward": gathered_gen_rewards_warmup.min().item(),
                "rollout/std_gail_rollout_reward": gathered_gen_rewards_warmup.std().item(),
                "warmup/is_warmup": 1.0,
                "warmup/steps_remaining": max(0, self.discriminator_warmup_steps - self.step_count - 1),
                "rollout/num_rollout_samples": gathered_gen_rewards_warmup.shape[0],
                "rollout/num_teacher_samples": gathered_teacher_rewards_warmup.shape[0],

                **discriminator_metrics,
                **discriminator_eval_metrics,
                **timing_metrics,
            }

            if dist.get_rank() == 0:
                wandb.log(metrics, step=self.step_count)

            if self.image_log_interval > 0 and self.step_count % self.image_log_interval == 0 and self.step_count > 0:
                self.log_images_to_wandb(
                    rollout_images=rollout_images,
                    teacher_images=teacher_images_flat,
                    rollout_captions=expanded_captions,
                    teacher_captions=teacher_expanded_captions,
                    step=self.step_count,
                )

            self.step_count += 1
            dist.barrier()
            return metrics

        if self.include_teacher_in_policy:
            combined_latents = self.interleave_rollout_and_teacher_by_prompt(
                final_latents, teacher_latents, batch_size, rollout_n, teacher_n
            )
            combined_rewards = self.interleave_rollout_and_teacher_by_prompt(
                gen_rewards, teacher_rewards, batch_size, rollout_n, teacher_n
            )

            combined_captions = []
            for caption in captions:
                combined_captions.extend([caption] * (rollout_n + teacher_n))

            effective_rollout_n = rollout_n + teacher_n
            effective_samples = combined_latents
            effective_rewards = combined_rewards
            effective_captions = combined_captions
        else:
            effective_rollout_n = rollout_n
            effective_samples = final_latents
            effective_rewards = gen_rewards
            effective_captions = expanded_captions

        cfm_start_time = time.time()

        t_samples, eps_samples, cfm_losses = self.compute_cfm_losses(
            final_latents=effective_samples,
            encoder_hidden_states=encoder_hidden_states.repeat_interleave(effective_rollout_n, dim=0),
            pooled_prompt_embeds=pooled_prompt_embeds.repeat_interleave(effective_rollout_n, dim=0),
            text_ids=text_ids.repeat_interleave(effective_rollout_n, dim=0),
            num_mc_samples=num_mc_samples,
            guidance_scale=guidance_scale,
        )

        cfm_time = time.time() - cfm_start_time
        timing_metrics["time/cfm"] = cfm_time

        sample_infos = []
        total_samples = effective_samples.shape[0]
        for i in range(total_samples):
            prompt_idx = i // effective_rollout_n
            sample_idx = i % effective_rollout_n

            sample_info = FpoSampleInfo(
                x_1=effective_samples[i].detach().clone(),
                cfm_samples_t=t_samples[i].detach().clone(),
                cfm_samples_eps=eps_samples[i].detach().clone(),
                cfm_samples_loss=cfm_losses[i].detach().clone(),
                initial_noise=None,
                prompt_idx=prompt_idx,
                sample_idx=sample_idx,
            )
            sample_infos.append(sample_info)

        advantages = FpoUtils.compute_advantages_grouped(effective_rewards, sample_infos)

        self.transformer.train()

        fpo_batch = FpoBatch(
            sample_infos=sample_infos,
            rewards=effective_rewards,
            advantages=advantages,
            prompts=captions,
            encoder_hidden_states=encoder_hidden_states.repeat_interleave(effective_rollout_n, dim=0),
            pooled_prompt_embeds=pooled_prompt_embeds.repeat_interleave(effective_rollout_n, dim=0),
            text_ids=text_ids.repeat_interleave(effective_rollout_n, dim=0),
        )

        update_start_time = time.time()

        update_metrics = self._update_policy(fpo_batch, guidance_scale)
        update_time = time.time() - update_start_time
        timing_metrics["time/policy"] = update_time

        gathered_gen_rewards = gather_tensor(gen_rewards)
        gathered_teacher_rewards = gather_tensor(teacher_rewards)

        total_time = rollout_time + teacher_time + decode_time + discriminator_time + reward_time + cfm_time + update_time
        timing_metrics["time/total"] = total_time

        metrics = {
            "rollout/mean_gail_rollout_reward": gathered_gen_rewards.mean().item(),
            "rollout/mean_gail_teacher_reward": gathered_teacher_rewards.mean().item(),
            "rollout/max_gail_rollout_reward": gathered_gen_rewards.max().item(),
            "rollout/min_gail_rollout_reward": gathered_gen_rewards.min().item(),
            "rollout/std_gail_rollout_reward": gathered_gen_rewards.std().item(),
            "rollout/num_rollout_samples": gathered_gen_rewards.shape[0],
            "rollout/num_teacher_samples": gathered_teacher_rewards.shape[0],
            "rollout/effective_rollout_n": effective_rollout_n,

            **discriminator_metrics,
            **discriminator_eval_metrics,
            **update_metrics,
            **timing_metrics,
        }

        if dist.get_rank() == 0:
            wandb.log(metrics, step=self.step_count)

        if self.image_log_interval > 0 and self.step_count % self.image_log_interval == 0 and self.step_count > 0:
            self.log_images_to_wandb(
                rollout_images=rollout_images,
                teacher_images=teacher_images_flat,
                rollout_captions=expanded_captions,
                teacher_captions=teacher_expanded_captions,
                step=self.step_count,
            )

        self.step_count += 1
        dist.barrier()

        return metrics

    def _update_policy(self, fpo_batch: FpoBatch, guidance_scale: float) -> Dict[str, float]:
        """
        Update policy using FPO algorithm.

        Args:
            fpo_batch: Batch of FPO samples
            guidance_scale: Guidance scale for generation

        Returns:
            Update metrics dictionary
        """
        from ..models.fpo_utils import FpoUtils

        total_samples = len(fpo_batch.sample_infos)

        all_x1 = torch.stack([info.x_1 for info in fpo_batch.sample_infos])
        all_t_samples = torch.stack([info.cfm_samples_t for info in fpo_batch.sample_infos])
        all_eps_samples = torch.stack([info.cfm_samples_eps for info in fpo_batch.sample_infos])
        all_old_cfm_losses = torch.stack([info.cfm_samples_loss for info in fpo_batch.sample_infos])

        expanded_encoder_hidden_states = fpo_batch.encoder_hidden_states.repeat_interleave(
            total_samples // fpo_batch.encoder_hidden_states.shape[0], dim=0
        )
        expanded_pooled_prompt_embeds = fpo_batch.pooled_prompt_embeds.repeat_interleave(
            total_samples // fpo_batch.pooled_prompt_embeds.shape[0], dim=0
        )
        expanded_text_ids = fpo_batch.text_ids

        all_reference_velocities = None
        if self.reference_transformer is not None and self.kl_beta > 0:

            all_reference_velocities = self._compute_reference_velocity(
                all_x1, all_t_samples, all_eps_samples,
                expanded_encoder_hidden_states, expanded_pooled_prompt_embeds,
                expanded_text_ids, guidance_scale,
            )

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

        for epoch in range(self.num_epochs):
            indices = torch.randperm(total_samples, device=self.device)
            self.optimizer.zero_grad()

            batch_x1 = all_x1[indices]
            batch_t_samples = all_t_samples[indices]
            batch_eps_samples = all_eps_samples[indices]
            batch_old_cfm_losses = all_old_cfm_losses[indices]
            batch_advantages = fpo_batch.advantages[indices]
            batch_encoder_hidden_states = expanded_encoder_hidden_states[indices]
            batch_pooled_prompt_embeds = expanded_pooled_prompt_embeds[indices]
            batch_text_ids = expanded_text_ids

            if all_reference_velocities is not None:
                batch_new_cfm_losses, batch_velocity_pred = self._compute_cfm_loss_batch(
                    batch_x1,
                    batch_t_samples,
                    batch_eps_samples,
                    batch_encoder_hidden_states,
                    batch_pooled_prompt_embeds,
                    batch_text_ids,
                    guidance_scale,
                    use_grad=True,
                    return_velocity=True,
                )
            else:
                batch_new_cfm_losses = self._compute_cfm_loss_batch(
                    batch_x1,
                    batch_t_samples,
                    batch_eps_samples,
                    batch_encoder_hidden_states,
                    batch_pooled_prompt_embeds,
                    batch_text_ids,
                    guidance_scale,
                    use_grad=True,
                )

            fpo_ratio = FpoUtils.compute_fpo_ratio(batch_old_cfm_losses, batch_new_cfm_losses)
            clipped_advantages = torch.clamp(batch_advantages, -self.adv_clip_max, self.adv_clip_max)

            per_sample_loss = - fpo_ratio * clipped_advantages
            clipped_per_sample_loss = - torch.clamp(fpo_ratio, 1 - self.clip_range,
                                                    1 + self.clip_range) * clipped_advantages
            per_sample_policy_loss = torch.maximum(per_sample_loss, clipped_per_sample_loss)

            policy_loss = per_sample_policy_loss.mean()

            if all_reference_velocities is not None:
                num_mc_samples = all_t_samples.shape[1]

                velocity_indices = []
                for idx in indices:
                    for mc_idx in range(num_mc_samples):
                        velocity_indices.append(idx * num_mc_samples + mc_idx)
                velocity_indices = torch.tensor(velocity_indices, device=self.device)

                batch_reference_velocity = all_reference_velocities[velocity_indices]

                per_sample_kl = ((batch_velocity_pred.view(total_samples, num_mc_samples, -1) -
                                  batch_reference_velocity.view(total_samples, num_mc_samples, -1)) ** 2).mean(
                    dim=[1, 2])

                kl_loss = per_sample_kl.mean()
            else:
                kl_loss = torch.tensor(0.0, device=self.device, dtype=torch.bfloat16)

            combined_loss = policy_loss + self.kl_beta * kl_loss
            combined_loss.backward()

            grad_norm = nn.utils.clip_grad_norm_(self.transformer.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

            policy_losses.append(policy_loss.item())
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

            dist.barrier()

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

        if dist.get_rank() == 0:
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

        metrics = {
            "policy/loss": global_policy_loss,
            "policy/kl_loss": global_kl_loss,
            "policy/combined_loss": global_combined_loss,
            "policy/fpo_ratio_mean": global_fpo_ratio_mean,
            "policy/fpo_ratio_max": global_fpo_ratio_max,
            "policy/fpo_ratio_min": global_fpo_ratio_min,
            "policy/advantage_mean": global_advantage_mean,
            "policy/advantage_max": global_advantage_max,
            "policy/advantage_min": global_advantage_min,
            "policy/clipped_ratio": global_clipped_ratio,
            "policy/grad_norm": global_grad_norm,
        }

        return metrics

    def save_checkpoint(self, output_dir: str, step: int = None):
        """Save training checkpoint including discriminator using FSDP-aware utilities.

        Uses the checkpoint utilities from fastvideo.utils.checkpoint for proper
        FSDP state dict handling.

        Args:
            output_dir: Directory to save checkpoint (will create checkpoint-{step} subdir)
            step: Step number for checkpoint naming (defaults to self.global_step)
        """
        from fastvideo.utils.checkpoint import save_checkpoint_generator_discriminator

        rank = dist.get_rank() if dist.is_initialized() else 0
        step = step if step is not None else self.global_step

        save_checkpoint_generator_discriminator(
            model=self.transformer,
            optimizer=self.optimizer,
            discriminator=self.discriminator,
            discriminator_optimizer=self.discriminator_optimizer,
            rank=rank,
            output_dir=output_dir,
            step=step,
        )

        main_print(f"FAIL checkpoint saved at step {step}")

    def load_checkpoint(self, checkpoint_dir: str):
        """Load training checkpoint including discriminator using FSDP-aware utilities.

        Args:
            checkpoint_dir: Path to checkpoint directory (e.g., output_dir/checkpoint-100)
        """
        from fastvideo.utils.checkpoint import resume_training_generator_discriminator
        import os

        rank = dist.get_rank() if dist.is_initialized() else 0

        self.transformer, self.optimizer, self.discriminator, self.discriminator_optimizer, step = \
            resume_training_generator_discriminator(
                model=self.transformer,
                optimizer=self.optimizer,
                discriminator=self.discriminator,
                discriminator_optimizer=self.discriminator_optimizer,
                checkpoint_dir=checkpoint_dir,
                rank=rank,
            )
