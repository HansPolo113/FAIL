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
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional, Union
import numpy as np
import time
import logging
from copy import deepcopy
from contextlib import contextmanager

from diffusers.image_processor import VaeImageProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from .logging_ import main_print

logger = logging.getLogger(__name__)


def gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather tensor across all processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """Calculate timestep shift for FLUX based on image sequence length."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[torch.device] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Retrieve timesteps from scheduler.
    Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed")
    if timesteps is not None:
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


class FluxFailPDTrainer:
    """
    FAIL-PD (Pathwise Derivative) trainer for FLUX.

    Uses discriminator gradients to directly update the policy through:
    1. Renoise x1 to x_t
    2. Predict velocity v
    3. Integrate to x1' = x_t + v * (1-t)
    4. Maximize discriminator score on x1'
    """

    def __init__(
        self,
        transformer: nn.Module,
        vae: nn.Module,
        flux_pipeline,
        discriminator: nn.Module,
        discriminator_optimizer,
        discriminator_scheduler,
        reward_model,  # HPS reward model for monitoring
        optimizer,
        device: torch.device,
        args,
        discriminator_updates_per_step: int = 1,
        discriminator_grad_clip: float = 1.0,
        discriminator_warmup_steps: int = 0,
        max_grad_norm: float = 1.0,
        include_teacher_in_dis_update: bool = False,
        reference_transformer: Optional[nn.Module] = None,
        kl_beta: float = 0.0,
        reference_update_frequency: int = 100,
        num_mc_samples: int = 8,
        noise_type: str = "random",
        timestep_type: str = "random",
        reward_batch_size: Optional[int] = None,
        image_log_interval: int = 50
    ):
        """
        Initialize FAIL-PD trainer for FLUX with QwenVL discriminator.

        Args:
            transformer: FLUX transformer model
            vae: VAE for encoding/decoding
            flux_pipeline: FLUX pipeline for generation
            discriminator: QwenVL discriminator network
            discriminator_optimizer: Optimizer for discriminator
            discriminator_scheduler: LR scheduler for discriminator
            optimizer: Optimizer for policy
            device: Device to run on
            args: Training arguments
            discriminator_updates_per_step: Number of discriminator updates per training step
            discriminator_grad_clip: Gradient clipping for discriminator
            discriminator_warmup_steps: Number of steps to train discriminator before policy
            max_grad_norm: Gradient clipping for policy
            include_teacher_in_dis_update: Whether to include teacher samples in policy update
            reference_transformer: Frozen reference model for KL regularization
            kl_beta: Weight for KL regularization
            reference_update_frequency: Steps between reference model updates
            num_mc_samples: Number of Monte Carlo samples per image
            noise_type: Type of noise to use ('original' or 'random')
            timestep_type: Type of timestep sampling ('random' or 'scheduler')
        """
        self.transformer = transformer
        self.vae = vae
        self.flux_pipeline = flux_pipeline
        self.discriminator = discriminator
        self.discriminator_optimizer = discriminator_optimizer
        self.discriminator_scheduler = discriminator_scheduler
        self.reward_model = reward_model
        self.optimizer = optimizer
        self.device = device
        self.args = args

        self.discriminator_updates_per_step = discriminator_updates_per_step
        self.discriminator_grad_clip = discriminator_grad_clip
        self.discriminator_warmup_steps = discriminator_warmup_steps
        self.max_grad_norm = max_grad_norm

        self.include_teacher_in_dis_update = include_teacher_in_dis_update

        self.reference_transformer = reference_transformer
        self.kl_beta = kl_beta
        self.reference_update_frequency = reference_update_frequency

        self.num_mc_samples = num_mc_samples
        self.noise_type = noise_type
        self.timestep_type = timestep_type

        self.reward_batch_size = reward_batch_size
        self.image_log_interval = image_log_interval

        self.vae_processor = VaeImageProcessor()

        self.vae.eval()
        for param in self.vae.parameters():
            param.requires_grad = False

        self.global_step = 0

    def update_reference_model(self):
        """Update reference model with current transformer weights."""
        if self.reference_transformer is not None:
            self.reference_transformer.load_state_dict(self.transformer.state_dict())
    

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

    def encode_teacher_images_to_latents(self, teacher_images: torch.Tensor) -> torch.Tensor:
        """
        Encode teacher images to latents using VAE.

        Args:
            teacher_images: [batch, teacher_n, 3, H, W] teacher images in [0, 1]

        Returns:
            teacher_latents: [batch * teacher_n, C, latent_h, latent_w] encoded latents
        """
        batch_size, teacher_n, _, h, w = teacher_images.shape

        teacher_images_flat = teacher_images.view(batch_size * teacher_n, 3, h, w)
        teacher_images_device = teacher_images_flat.to(self.device)

        with torch.no_grad():
            latents = self.vae.encode(teacher_images_device.to(self.flux_pipeline.vae.dtype)).latent_dist.sample()
            latents = (latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        return latents

    def decode_latents_to_images(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latents to images for discriminator input.

        Args:
            latents: [batch, C, latent_h, latent_w] latents

        Returns:
            images: [batch, 3, H, W] decoded images in [0, 1]
        """
        latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor

        with torch.no_grad():
            images = self.vae.decode(latents, return_dict=False)[0]

        return images

    def update_discriminator(
        self,
        rollout_latents: torch.Tensor,
        teacher_latents: torch.Tensor,
        rollout_prompts: List[str],
        teacher_prompts: List[str],
    ) -> Dict[str, float]:
        """
        Update QwenVL discriminator with rollout and teacher samples.

        Args:
            rollout_latents: [batch * rollout_n, C, H, W] generated latents
            teacher_latents: [batch * teacher_n, C, H, W] teacher latents
            rollout_prompts: Prompts for rollout images [batch*rollout_n]
            teacher_prompts: Prompts for teacher images [batch*teacher_n]

        Returns:
            metrics: Dictionary of training metrics
        """
        self.discriminator.train()
        metrics = {}

        for _ in range(self.discriminator_updates_per_step):
            self.discriminator_optimizer.zero_grad()

            rollout_batch_size = rollout_latents.shape[0]

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                rollout_images = self.decode_latents_to_images(rollout_latents)
                teacher_images = self.decode_latents_to_images(teacher_latents)
                all_images = torch.cat([rollout_images, teacher_images], dim=0)

                all_prompts = rollout_prompts + teacher_prompts
                all_logits = self.discriminator(all_images, all_prompts)

            rollout_logits = all_logits[:rollout_batch_size]
            teacher_logits = all_logits[rollout_batch_size:]

            rollout_targets = torch.zeros_like(rollout_logits)
            teacher_targets = torch.ones_like(teacher_logits)

            rollout_loss = F.binary_cross_entropy_with_logits(rollout_logits, rollout_targets)
            teacher_loss = F.binary_cross_entropy_with_logits(teacher_logits, teacher_targets)
            discriminator_loss = rollout_loss + teacher_loss

            discriminator_loss.backward()

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

            self.discriminator_optimizer.zero_grad()

            with torch.no_grad():
                rollout_probs = torch.sigmoid(rollout_logits)
                teacher_probs = torch.sigmoid(teacher_logits)
                rollout_acc = (rollout_probs < 0.5).float().mean()
                teacher_acc = (teacher_probs > 0.5).float().mean()
                overall_acc = (rollout_acc + teacher_acc) / 2

                metrics['discriminator/loss'] = discriminator_loss.detach().item()
                metrics['discriminator/rollout_loss'] = rollout_loss.detach().item()
                metrics['discriminator/teacher_loss'] = teacher_loss.detach().item()
                metrics['discriminator/rollout_acc'] = rollout_acc.detach().item()
                metrics['discriminator/teacher_acc'] = teacher_acc.detach().item()
                metrics['discriminator/overall_acc'] = overall_acc.detach().item()
                metrics['discriminator/rollout_logits_mean'] = rollout_logits.mean().detach().item()
                metrics['discriminator/teacher_logits_mean'] = teacher_logits.mean().detach().item()
                metrics['discriminator/grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

        if self.discriminator_scheduler is not None:
            self.discriminator_scheduler.step()

        return metrics

    def sample_timesteps_and_noise(
        self,
        batch_size: int,
        latent_shape: Tuple[int, int, int],  # (C, H, W)
        original_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample timesteps and noise for renoising.

        Args:
            batch_size: Number of samples
            latent_shape: (C, H, W) shape of latents
            original_noise: Original rollout noise if noise_type="original"

        Returns:
            t_samples: [batch_size, num_mc_samples] timesteps in [0, 1]
            eps_samples: [batch_size, num_mc_samples, C, H, W] noise
            dt_samples: [batch_size, num_mc_samples] timestep spacings
        """
        C, latent_h, latent_w = latent_shape

        if self.timestep_type == "random":
            t = torch.rand(batch_size, self.num_mc_samples, device=self.device)
            dt = -torch.full_like(t, 0.01)

        elif self.timestep_type == "scheduler":
            num_inference_steps = self.args.num_sampling_steps
            image_seq_len = (latent_h // 2) * (latent_w // 2)

            mu = calculate_shift(
                image_seq_len,
                base_seq_len=256,
                max_seq_len=4096,
                base_shift=0.5,
                max_shift=1.15
            )

            sigmas = np.linspace(1.0, 1/num_inference_steps, num_inference_steps)
            timesteps, _ = retrieve_timesteps(
                self.flux_pipeline.scheduler,
                num_inference_steps,
                self.device,
                sigmas=sigmas,
                mu=mu
            )

            timesteps_normalized = timesteps.float() / 1000.0

            indices = torch.randint(
                0, len(timesteps_normalized) - 1,  # Exclude last timestep
                (batch_size, self.num_mc_samples),
                device=self.device
            )
            t = timesteps_normalized[indices]

            dt = torch.zeros_like(t)
            for i in range(batch_size):
                for j in range(self.num_mc_samples):
                    idx = indices[i, j].item()
                    dt[i, j] = timesteps_normalized[idx+1] - timesteps_normalized[idx]

        else:
            raise ValueError(f"Unknown timestep_type: {self.timestep_type}")

        if self.noise_type == "original" and original_noise is not None:
            eps = original_noise.unsqueeze(1).repeat(1, self.num_mc_samples, 1, 1, 1)

        elif self.noise_type == "random":
            eps = torch.randn(
                batch_size, self.num_mc_samples, C, latent_h, latent_w,
                device=self.device,
                dtype=torch.bfloat16
            )

        else:
            raise ValueError(f"Unknown noise_type: {self.noise_type}")

        return t, eps, dt

    def renoise_predict_integrate(
        self,
        x_1: torch.Tensor,  # [batch, C, H, W]
        t_samples: torch.Tensor,  # [batch, num_mc]
        eps_samples: torch.Tensor,  # [batch, num_mc, C, H, W]
        dt_samples: torch.Tensor,  # [batch, num_mc]
        encoder_hidden_states: torch.Tensor,  # [batch, seq_len, dim]
        pooled_prompt_embeds: torch.Tensor,  # [batch, dim]
        text_ids: torch.Tensor,  # [1, seq_len, 3]
        guidance_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Core FAIL computation: renoise → predict velocity → integrate to x1'.

        Args:
            x_1: Final denoised latents [batch, C, H, W]
            t_samples: Timestep samples [batch, num_mc]
            eps_samples: Noise samples [batch, num_mc, C, H, W]
            dt_samples: Timestep spacings [batch, num_mc]
            encoder_hidden_states: Text embeddings [batch, seq_len, dim]
            pooled_prompt_embeds: Pooled text embeddings [batch, dim]
            text_ids: Text token IDs [1, seq_len, 3]
            guidance_scale: CFG scale

        Returns:
            x1_prime: Integrated latents [batch * num_mc, C, H, W]
            velocity_pred: Predicted velocities [batch * num_mc, seq_len, channels] (packed)
        """
        batch_size, num_mc = t_samples.shape
        _, C, latent_h, latent_w = x_1.shape

        x_1_packed = self.flux_pipeline._pack_latents(
            x_1, batch_size, C, latent_h, latent_w
        )

        t_flat = t_samples.flatten()
        dt_flat = dt_samples.flatten()
        eps_flat = eps_samples.view(batch_size * num_mc, C, latent_h, latent_w)

        eps_packed = self.flux_pipeline._pack_latents(
            eps_flat, batch_size * num_mc, C, latent_h, latent_w
        )

        x_1_packed_flat = x_1_packed.repeat_interleave(num_mc, dim=0)

        t_expanded = t_flat.view(-1, 1, 1)
        dt_expanded = dt_flat.view(-1, 1, 1)
        x_t_packed = (1 - t_expanded) * x_1_packed_flat + t_expanded * eps_packed

        image_ids = self.flux_pipeline._prepare_latent_image_ids(
            batch_size * num_mc, latent_h // 2, latent_w // 2, self.device, torch.bfloat16
        )

        encoder_hidden_states_flat = encoder_hidden_states.repeat_interleave(num_mc, dim=0)
        pooled_prompt_embeds_flat = pooled_prompt_embeds.repeat_interleave(num_mc, dim=0)
        text_ids_flat = text_ids[0]

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            velocity_pred = self.transformer(
                hidden_states=x_t_packed,
                encoder_hidden_states=encoder_hidden_states_flat,
                pooled_projections=pooled_prompt_embeds_flat,
                timestep=t_flat,
                img_ids=image_ids,
                txt_ids=text_ids_flat,
                guidance=torch.full(
                    (batch_size * num_mc,),
                    guidance_scale,
                    device=self.device,
                    dtype=torch.bfloat16
                ),
                return_dict=False,
            )[0]

        t_next = t_expanded + dt_expanded
        t_next = torch.clamp(t_next, max=0.999)

        x_t_next_packed = x_t_packed + dt_expanded * velocity_pred

        x1_prime_packed = (x_t_next_packed - t_next * eps_packed) / (1 - t_next)

        x1_prime = self.flux_pipeline._unpack_latents(
            x1_prime_packed,
            self.args.height,
            self.args.width,
            self.flux_pipeline.vae_scale_factor
        )

        return x1_prime, velocity_pred

    def compute_discriminator_gradient_loss(
        self,
        x1_prime: torch.Tensor,  # [batch, C, H, W]
        prompts: List[str],
        return_logits: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compute policy loss from QwenVL discriminator gradients.

        Goal: Maximize discriminator score = minimize BCE with target=1

        Args:
            x1_prime: Integrated latents [batch, C, H, W]
            prompts: Prompts for QwenVL discriminator
            return_logits: If True, also return discriminator logits for logging

        Returns:
            loss: Discriminator gradient loss (scalar)
            logits: (optional) Discriminator logits if return_logits=True
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latents_scaled = x1_prime / self.vae.config.scaling_factor + self.vae.config.shift_factor
            latents_scaled = latents_scaled.to(dtype=self.vae.dtype)
            images = self.vae.decode(latents_scaled, return_dict=False)[0]

            logits = self.discriminator(images, prompts)

        target = torch.ones_like(logits)
        loss = F.binary_cross_entropy_with_logits(logits, target)

        if return_logits:
            return loss, logits
        return loss

    def compute_monitoring_rewards(self, images: torch.Tensor, captions: List[str]) -> torch.Tensor:
        """
        Compute HPS rewards for monitoring purposes.

        Args:
            images: [batch, 3, H, W] images in [0, 1]
            captions: List of text prompts

        Returns:
            rewards: [batch] HPS rewards
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

    def compute_kl_loss(
        self,
        velocity_current: torch.Tensor,  # [batch * num_mc, seq_len, channels]
        x_t_packed: torch.Tensor,        # [batch * num_mc, seq_len, channels]
        t_flat: torch.Tensor,            # [batch * num_mc]
        encoder_hidden_states_flat: torch.Tensor,
        pooled_prompt_embeds_flat: torch.Tensor,
        text_ids_flat: torch.Tensor,
        image_ids: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        KL regularization with frozen reference model.

        Args:
            velocity_current: Current model's velocity prediction
            x_t_packed: Noised latents (packed)
            t_flat: Timesteps
            encoder_hidden_states_flat: Text embeddings
            pooled_prompt_embeds_flat: Pooled text embeddings
            text_ids_flat: Text token IDs
            image_ids: Image position IDs
            guidance_scale: CFG scale

        Returns:
            kl_loss: MSE loss between current and reference velocities (scalar)
        """
        if self.reference_transformer is None:
            return torch.tensor(0.0, device=self.device)

        with torch.no_grad(), \
            torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            velocity_ref = self.reference_transformer(
                hidden_states=x_t_packed,
                encoder_hidden_states=encoder_hidden_states_flat,
                pooled_projections=pooled_prompt_embeds_flat,
                timestep=t_flat,
                img_ids=image_ids,
                txt_ids=text_ids_flat,
                guidance=torch.full(
                    (len(t_flat),),
                    guidance_scale,
                    device=self.device,
                    dtype=torch.bfloat16
                ),
                return_dict=False,
            )[0]

        kl_loss = F.mse_loss(velocity_current, velocity_ref)
        return kl_loss

    def _update_policy(
        self,
        all_latents: torch.Tensor,  # [num_samples, C, H, W]
        all_encoder_hidden_states: torch.Tensor,  # [num_samples, seq_len, dim]
        all_pooled_prompt_embeds: torch.Tensor,  # [num_samples, dim]
        all_text_ids: torch.Tensor,  # [num_samples, seq_len, 3]
        all_captions: List[str],
        all_initial_noise: torch.Tensor,  # [num_samples, C, H, W]
        is_teacher_mask: Optional[torch.Tensor],  # [num_samples] bool
        guidance_scale: float = 1.0,
    ) -> Tuple[Dict[str, float], Optional[torch.Tensor]]:
        """
        Update policy using discriminator gradients with gradient accumulation.

        Args:
            all_latents: All x1 latents (rollout + optionally teacher)
            all_encoder_hidden_states: Text embeddings for all samples
            all_pooled_prompt_embeds: Pooled text embeddings for all samples
            all_text_ids: Text token IDs for all samples
            all_captions: Captions for all samples
            all_initial_noise: Initial noise for all samples
            is_teacher_mask: Boolean mask indicating teacher samples (if include_teacher_in_dis_update)
            guidance_scale: CFG scale

        Returns:
            Tuple of (policy update metrics, teacher_x1_prime if teacher samples included else None)
        """
        num_samples = len(all_latents)
        metrics = {}

        t_samples, eps_samples, dt_samples = self.sample_timesteps_and_noise(
            batch_size=num_samples,
            latent_shape=all_latents.shape[1:],
            original_noise=all_initial_noise if self.noise_type == "original" else None,
        )

        self.discriminator.eval()

        self.transformer.train()

        self.optimizer.zero_grad()

        total_loss = 0.0
        total_dis_loss = 0.0
        total_kl_loss = 0.0

        all_logits = []
        rollout_logits_list = []
        teacher_logits_list = []

        teacher_x1_prime_list = []

        x1_prime, velocity_pred = self.renoise_predict_integrate(
            all_latents,
            t_samples,
            eps_samples,
            dt_samples,
            all_encoder_hidden_states,
            all_pooled_prompt_embeds,
            all_text_ids,
            guidance_scale=guidance_scale,
        )

        all_captions_repeated = []
        for caption in all_captions:
            all_captions_repeated.extend([caption] * self.num_mc_samples)

        dis_grad_loss, batch_logits = self.compute_discriminator_gradient_loss(
            x1_prime,
            prompts=all_captions_repeated,
            return_logits=True,
        )

        policy_loss = dis_grad_loss
        total_dis_loss = dis_grad_loss.item()

        with torch.no_grad():
            all_logits.append(batch_logits.detach())

            if is_teacher_mask is not None:
                is_teacher_expanded = is_teacher_mask.unsqueeze(1).repeat(1, self.num_mc_samples).flatten()

                rollout_mask = ~is_teacher_expanded
                teacher_mask = is_teacher_expanded

                if rollout_mask.any():
                    rollout_logits_list.append(batch_logits[rollout_mask].detach())
                if teacher_mask.any():
                    teacher_logits_list.append(batch_logits[teacher_mask].detach())
                    teacher_x1_prime_list.append(x1_prime[teacher_mask].detach())

        if self.kl_beta > 0 and self.reference_transformer is not None:
            C, latent_h, latent_w = all_latents.shape[1:]

            x_1_packed = self.flux_pipeline._pack_latents(all_latents, num_samples, C, latent_h, latent_w)
            eps_flat = eps_samples.view(num_samples * self.num_mc_samples, C, latent_h, latent_w)
            eps_packed = self.flux_pipeline._pack_latents(eps_flat, num_samples * self.num_mc_samples, C, latent_h, latent_w)

            x_1_packed = x_1_packed.to(dtype=torch.bfloat16)
            eps_packed = eps_packed.to(dtype=torch.bfloat16)

            x_1_packed_flat = x_1_packed.repeat_interleave(self.num_mc_samples, dim=0)
            t_flat = t_samples.flatten()
            t_expanded = t_flat.view(-1, 1, 1)
            x_t_packed = (1 - t_expanded) * x_1_packed_flat + t_expanded * eps_packed

            image_ids = self.flux_pipeline._prepare_latent_image_ids(
                num_samples * self.num_mc_samples, latent_h // 2, latent_w // 2, self.device, torch.bfloat16
            )

            encoder_hidden_states_flat = all_encoder_hidden_states.repeat_interleave(self.num_mc_samples, dim=0)
            pooled_prompt_embeds_flat = all_pooled_prompt_embeds.repeat_interleave(self.num_mc_samples, dim=0)
            text_ids_flat = all_text_ids[0]

            velocity_pred_kl = velocity_pred.to(dtype=torch.bfloat16)

            kl_loss = self.compute_kl_loss(
                velocity_pred_kl,
                x_t_packed,
                t_flat,
                encoder_hidden_states_flat,
                pooled_prompt_embeds_flat,
                text_ids_flat,
                image_ids,
                guidance_scale=guidance_scale,
            )

            policy_loss = policy_loss + self.kl_beta * kl_loss
            total_kl_loss = kl_loss.item()

        policy_loss.backward()

        total_loss = policy_loss.item()

        if isinstance(self.transformer, torch.distributed.fsdp.FullyShardedDataParallel):
            grad_norm = self.transformer.clip_grad_norm_(self.max_grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.transformer.parameters(),
                max_norm=self.max_grad_norm
            )

        if not torch.isfinite(grad_norm):
            logger.warning(f"Policy grad_norm is not finite: {grad_norm}")
            self.optimizer.zero_grad()
        else:
            self.optimizer.step()

        metrics['policy/total_loss'] = total_loss
        metrics['policy/dis_grad_loss'] = total_dis_loss
        metrics['policy/grad_norm'] = grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm

        if all_logits:
            all_logits_tensor = torch.cat(all_logits)
            metrics['policy/dis_logits_mean'] = all_logits_tensor.mean().item()

            if rollout_logits_list:
                rollout_logits_tensor = torch.cat(rollout_logits_list)
                metrics['policy/dis_logits_rollout_mean'] = rollout_logits_tensor.mean().item()

            if teacher_logits_list:
                teacher_logits_tensor = torch.cat(teacher_logits_list)
                metrics['policy/dis_logits_teacher_mean'] = teacher_logits_tensor.mean().item()

        if self.kl_beta > 0 and total_kl_loss > 0:
            metrics['policy/kl_loss'] = total_kl_loss

        teacher_x1_prime = None
        if teacher_x1_prime_list:
            teacher_x1_prime = torch.cat(teacher_x1_prime_list, dim=0)
    

        return metrics, teacher_x1_prime

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

        Args:
            rollout_images: Generated rollout images [batch*rollout_n, 3, H, W] in [-1, 1]
            teacher_images: Teacher images [batch*teacher_n, 3, H, W] in [-1, 1]
            rollout_captions: Captions for rollout images [batch*rollout_n]
            teacher_captions: Captions for teacher images [batch*teacher_n]
            step: Current training step
        """
        if not dist.is_initialized() or dist.get_rank() != 0:
            return  # Only log on rank 0

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
                    caption=f"Rollout: {caption}"
                )
            )

        for i in range(num_teacher_to_log):
            caption = teacher_captions[i] if i < len(teacher_captions) else ""
            logged_images.append(
                wandb.Image(
                    teacher_pil[i],
                    caption=f"Teacher: {caption}"
                )
            )

        wandb.log({"images": logged_images}, step=step)


    def rollout_phase(
        self,
        encoder_hidden_states: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        captions: List[str],
        rollout_n: int = 4,
        mini_batch_size: int = 8,
        guidance_scale: float = 1.0,
    ) -> Tuple[torch.Tensor, List[str], torch.Tensor]:
        """
        Rollout phase: Generate images and store initial noise.

        Args:
            encoder_hidden_states: Text embeddings [batch, seq_len, C]
            pooled_prompt_embeds: Pooled embeddings [batch, C]
            text_ids: Text IDs [batch, seq_len, 3]
            captions: List of prompt strings
            rollout_n: Number of samples per prompt
            mini_batch_size: Mini-batch size for rollout
            guidance_scale: Guidance scale for generation

        Returns:
            Tuple of (final_latents, expanded_captions, initial_noise)
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

        all_final_latents = []
        all_initial_noise = []

        with torch.no_grad(), \
            torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for mini_start in range(0, total_samples, mini_batch_size):
                mini_end = min(mini_start + mini_batch_size, total_samples)
                mini_batch_actual_size = mini_end - mini_start

                mini_encoder_hidden_states = expanded_encoder_hidden_states[mini_start:mini_end]
                mini_pooled_prompt_embeds = expanded_pooled_prompt_embeds[mini_start:mini_end]
                mini_text_ids = expanded_text_ids[mini_start:mini_end]

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

                all_initial_noise.append(mini_initial_noise)

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
                )

                final_latents_packed = pipeline_output[0]
                final_latents_unpacked = self.flux_pipeline._unpack_latents(
                    final_latents_packed, h, w, self.flux_pipeline.vae_scale_factor
                )

                all_final_latents.append(final_latents_unpacked.to(dtype=mini_encoder_hidden_states.dtype))

        all_final_latents = torch.cat(all_final_latents, dim=0)
        all_initial_noise = torch.cat(all_initial_noise, dim=0)

        return all_final_latents, expanded_captions, all_initial_noise

    def train_step(
        self,
        batch: Dict[str, Any],
        rollout_n: int = 4,
        teacher_n: int = None,
        guidance_scale: float = 1.0,
    ) -> Dict[str, float]:
        """
        Main FAIL training step.

        Pipeline:
        1. Rollout policy images (fake samples)
        2. Encode teacher images (real samples)
        3. Policy update with discriminator gradients (skip during warmup):
           - For rollout samples (+ optionally teacher):
             a. Sample (t, eps) pairs
             b. Renoise → predict → integrate
             c. Discriminator gradient loss
           - Optionally add teacher SFT loss
           - Optionally add KL loss
        4. Update discriminator on real/fake pairs
        5. Compute HPS rewards for monitoring
        6. Log images if needed

        Args:
            batch: Dictionary containing:
                - 'encoder_hidden_states': [batch, seq_len, dim]
                - 'pooled_prompt_embeds': [batch, dim]
                - 'text_ids': [batch, seq_len, 3]
                - 'captions': List[str]
                - 'teacher_images': [batch, teacher_n, 3, H, W]
            rollout_n: Number of rollout samples per prompt
            teacher_n: Number of teacher images per prompt (defaults to teacher_images.shape[1] if None)
            guidance_scale: CFG scale

        Returns:
            metrics: Dictionary of training metrics
        """
        metrics = {}
        step_start_time = time.time()

        encoder_hidden_states = batch['encoder_hidden_states']
        pooled_prompt_embeds = batch['pooled_prompt_embeds']
        text_ids = batch['text_ids']
        captions = batch['captions']
        teacher_images = batch['teacher_images']

        batch_size = encoder_hidden_states.shape[0]

        if teacher_n is None:
            teacher_n = teacher_images.shape[1]

        rollout_start = time.time()
        self.transformer.eval()

        rollout_latents, expanded_captions, initial_noise = self.rollout_phase(
            encoder_hidden_states,
            pooled_prompt_embeds,
            text_ids,
            captions,
            rollout_n=rollout_n,
            mini_batch_size=self.args.rollout_mini_batch_size,
            guidance_scale=guidance_scale,
        )

        rollout_time = time.time() - rollout_start
        metrics['time/rollout'] = rollout_time

        torch.cuda.empty_cache()

        teacher_latents = self.encode_teacher_images_to_latents(teacher_images)

        teacher_x1_prime = None
        if self.global_step < self.discriminator_warmup_steps:
            metrics['warmup/is_warmup'] = 1.0
            metrics['warmup/steps_remaining'] = max(0, self.discriminator_warmup_steps - self.global_step - 1)
        else:
            policy_start = time.time()

            if self.include_teacher_in_dis_update:
                all_latents = torch.cat([rollout_latents, teacher_latents], dim=0)
                is_teacher_mask = torch.cat([
                    torch.zeros(len(rollout_latents), dtype=torch.bool, device=self.device),
                    torch.ones(len(teacher_latents), dtype=torch.bool, device=self.device)
                ])
                teacher_captions = [caption for caption in captions for _ in range(teacher_n)]
                all_captions = expanded_captions + teacher_captions
                teacher_initial_noise = torch.randn_like(teacher_latents)
                all_initial_noise = torch.cat([initial_noise, teacher_initial_noise], dim=0)
            else:
                all_latents = rollout_latents
                is_teacher_mask = None
                all_captions = expanded_captions
                all_initial_noise = initial_noise

            num_samples = len(all_latents)
            samples_per_prompt = num_samples // batch_size

            all_encoder_hidden_states = encoder_hidden_states.repeat_interleave(samples_per_prompt, dim=0)
            all_pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(samples_per_prompt, dim=0)
            all_text_ids = text_ids.repeat_interleave(samples_per_prompt, dim=0)

            policy_metrics, teacher_x1_prime = self._update_policy(
                all_latents,
                all_encoder_hidden_states,
                all_pooled_prompt_embeds,
                all_text_ids,
                all_captions,
                all_initial_noise,
                is_teacher_mask,
                guidance_scale=guidance_scale,
            )

            for key, value in policy_metrics.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() == 0:
                        policy_metrics[key] = gather_tensor(value.unsqueeze(0)).mean().item()
                    else:
                        policy_metrics[key] = gather_tensor(value).mean().item()
                elif isinstance(value, (int, float)):
                    scalar_tensor = torch.tensor([value], device=self.device)
                    policy_metrics[key] = gather_tensor(scalar_tensor).mean().item()

            metrics.update(policy_metrics)

            policy_time = time.time() - policy_start
            metrics['time/policy'] = policy_time

            if self.reference_update_frequency > 0 and \
               self.global_step % self.reference_update_frequency == 0:
                self.update_reference_model()

        discriminator_rollout_latents = rollout_latents
        discriminator_rollout_prompts = expanded_captions

        teacher_expanded_captions_for_disc = [caption for caption in captions for _ in range(teacher_n)]
        disc_start = time.time()

        disc_metrics = self.update_discriminator(
            discriminator_rollout_latents.detach(),
            teacher_latents.detach(),
            rollout_prompts=discriminator_rollout_prompts,
            teacher_prompts=teacher_expanded_captions_for_disc,
        )

        for key, value in disc_metrics.items():
            if isinstance(value, torch.Tensor):
                if value.dim() == 0:
                    disc_metrics[key] = gather_tensor(value.unsqueeze(0)).mean().item()
                else:
                    disc_metrics[key] = gather_tensor(value).mean().item()
            elif isinstance(value, (int, float)):
                scalar_tensor = torch.tensor([value], device=self.device)
                disc_metrics[key] = gather_tensor(scalar_tensor).mean().item()

        metrics.update(disc_metrics)

        disc_time = time.time() - disc_start
        metrics['time/discriminator'] = disc_time

        if self.reward_model is not None:
            reward_start = time.time()

            rollout_images = self.decode_latents_to_images(rollout_latents)
            teacher_images_flat = teacher_images.view(-1, *teacher_images.shape[2:])

            teacher_expanded_captions = [caption for caption in captions for _ in range(teacher_n)]

            hps_gen_rewards = self.compute_monitoring_rewards(rollout_images, expanded_captions)
            hps_teacher_rewards = self.compute_monitoring_rewards(teacher_images_flat, teacher_expanded_captions)

            reward_time = time.time() - reward_start
            metrics['time/reward_computation'] = reward_time
            gathered_hps_gen_rewards = gather_tensor(hps_gen_rewards)
            gathered_hps_teacher_rewards = gather_tensor(hps_teacher_rewards)

            metrics['monitoring/mean_hps_gen_reward'] = gathered_hps_gen_rewards.mean().item()
            metrics['monitoring/max_hps_gen_reward'] = gathered_hps_gen_rewards.max().item()
            metrics['monitoring/min_hps_gen_reward'] = gathered_hps_gen_rewards.min().item()
            metrics['monitoring/std_hps_gen_reward'] = gathered_hps_gen_rewards.std().item()
            metrics['monitoring/mean_hps_teacher_reward'] = gathered_hps_teacher_rewards.mean().item()
            metrics['monitoring/max_hps_teacher_reward'] = gathered_hps_teacher_rewards.max().item()
            metrics['monitoring/min_hps_teacher_reward'] = gathered_hps_teacher_rewards.min().item()
            metrics['monitoring/std_hps_teacher_reward'] = gathered_hps_teacher_rewards.std().item()

        if self.image_log_interval > 0 and self.global_step % self.image_log_interval == 0 and self.global_step > 0:
            self.log_images_to_wandb(
                rollout_images=rollout_images,
                teacher_images=teacher_images_flat,
                rollout_captions=expanded_captions,
                teacher_captions=teacher_expanded_captions,
                step=self.global_step,
            )

        metrics['time/total'] = time.time() - step_start_time

        self.global_step += 1

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

        if rank == 0:
            import os
            step_file = os.path.join(output_dir, f"checkpoint-{step}", "training_state.pt")
            torch.save({'global_step': self.global_step}, step_file)

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

        step_file = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(step_file):
            state = torch.load(step_file, map_location='cpu', weights_only=False)
            self.global_step = state.get('global_step', step)
        else:
            self.global_step = step

        main_print(f"FAIL checkpoint loaded from {checkpoint_dir}, step: {self.global_step}")
