"""
FPO (Flow Policy Optimization) utilities for FAIL training.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch


@dataclass
class FpoSampleInfo:
    """Information stored for each sample during FPO rollout phase."""
    x_1: torch.Tensor           # Final denoised image [C, H, W]
    cfm_samples_t: torch.Tensor # Sampled timesteps [N_mc]
    cfm_samples_eps: torch.Tensor  # Sampled noise [N_mc, C, H, W]
    cfm_samples_loss: torch.Tensor # Precomputed CFM loss (scalar)
    initial_noise: torch.Tensor    # Initial noise used [C, H, W]
    prompt_idx: int                # Which prompt this sample belongs to
    sample_idx: int                # Sample index within prompt group


@dataclass
class FpoBatch:
    """Batch-level container for FPO training data."""
    sample_infos: List[FpoSampleInfo]
    rewards: torch.Tensor                # [B*n]
    advantages: torch.Tensor             # [B*n]
    prompts: List[str]                   # [B]
    encoder_hidden_states: torch.Tensor  # [B, seq_len, dim]
    pooled_prompt_embeds: torch.Tensor   # [B, dim]
    text_ids: torch.Tensor               # [B, seq_len]
    sample_mask: Optional[torch.Tensor] = None
    is_teacher_mask: Optional[torch.Tensor] = None

    def get_prompt_groups(self) -> List[List[int]]:
        """Get indices grouped by prompt."""
        groups = {}
        for i, info in enumerate(self.sample_infos):
            prompt_idx = info.prompt_idx
            if prompt_idx not in groups:
                groups[prompt_idx] = []
            groups[prompt_idx].append(i)
        return list(groups.values())


@dataclass
class DpoBatch:
    """Batch-level container for DPO training data."""
    sample_infos: List[FpoSampleInfo]
    prompts: List[str]
    encoder_hidden_states: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    text_ids: torch.Tensor

    def get_prompt_groups(self) -> List[List[int]]:
        """Get indices grouped by prompt."""
        groups = {}
        for i, info in enumerate(self.sample_infos):
            prompt_idx = info.prompt_idx
            if prompt_idx not in groups:
                groups[prompt_idx] = []
            groups[prompt_idx].append(i)
        return list(groups.values())


@dataclass
class SD3FpoBatch(FpoBatch):
    """SD3-specific batch container with negative embeddings for CFG."""
    negative_encoder_hidden_states: Optional[torch.Tensor] = None
    negative_pooled_prompt_embeds: Optional[torch.Tensor] = None


class FpoUtils:
    """Utility functions for FPO training."""

    @staticmethod
    def sample_cfm_pairs(
        num_samples: int,
        action_dim: Tuple[int, ...],
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample (t, ε) pairs for CFM loss computation.

        Args:
            num_samples: Number of MC samples
            action_dim: Dimensions of the action space (C, H, W)
            device: Target device

        Returns:
            t: Sampled timesteps [num_samples]
            eps: Sampled noise [num_samples, *action_dim]
        """
        t = torch.rand(num_samples, device=device)
        eps = torch.randn(num_samples, *action_dim, device=device)
        return t, eps

    @staticmethod
    def compute_advantages_grouped(
        rewards: torch.Tensor,
        sample_infos: List[FpoSampleInfo]
    ) -> torch.Tensor:
        """
        Compute advantages grouped by prompt (GRPO-style).

        Args:
            rewards: Reward values [B*n]
            sample_infos: Sample information for grouping

        Returns:
            advantages: Computed advantages [B*n]
        """
        advantages = torch.zeros_like(rewards)

        # Group samples by prompt
        prompt_groups = {}
        for i, info in enumerate(sample_infos):
            prompt_idx = info.prompt_idx
            if prompt_idx not in prompt_groups:
                prompt_groups[prompt_idx] = []
            prompt_groups[prompt_idx].append(i)

        # Compute advantages within each group
        for group_indices in prompt_groups.values():
            group_rewards = rewards[group_indices]
            group_mean = group_rewards.mean()
            group_std = group_rewards.std() + 1e-8

            group_advantages = (group_rewards - group_mean) / group_std
            for i, idx in enumerate(group_indices):
                advantages[idx] = group_advantages[i]

        return advantages

    @staticmethod
    def create_minibatches(batch_size: int, minibatch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Create shuffled minibatch indices."""
        indices = torch.randperm(batch_size, device=device)
        minibatches = []
        for i in range(0, batch_size, minibatch_size):
            end_idx = min(i + minibatch_size, batch_size)
            minibatches.append(indices[i:end_idx])
        return minibatches

    @staticmethod
    def pack_latents_for_cfm(latents: torch.Tensor, batch_size: int, in_channels: int,
                           latent_h: int, latent_w: int) -> torch.Tensor:
        """
        Pack latents for CFM computation, similar to existing pack_latents function.

        Args:
            latents: Input latents [batch_size, in_channels, latent_h, latent_w]
            batch_size: Batch size
            in_channels: Number of input channels
            latent_h: Latent height
            latent_w: Latent width

        Returns:
            packed_latents: Packed latents for transformer input
        """
        # This should match the pack_latents function used in FLUX training
        # Implementation depends on specific FLUX requirements
        latents = latents.view(batch_size, in_channels, latent_h // 2, 2, latent_w // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, (latent_h // 2) * (latent_w // 2), in_channels * 4)
        return latents

    @staticmethod
    def compute_fpo_ratio(old_loss: torch.Tensor, new_loss: torch.Tensor,
                         clip_max: float = 3.0) -> torch.Tensor:
        """
        Compute FPO ratio from CFM loss differences.

        Args:
            old_loss: CFM loss from old policy [batch_size]
            new_loss: CFM loss from current policy [batch_size]
            clip_max: Maximum clipping value for stability

        Returns:
            ratio: FPO ratio exp(L_old - L_new) [batch_size]
        """
        loss_diff = old_loss - new_loss
        loss_diff = torch.clamp(loss_diff, -clip_max, clip_max)
        ratio = torch.exp(loss_diff)
        return ratio

    @staticmethod
    def compute_fpo_ratio_online(old_loss: torch.Tensor, new_loss: torch.Tensor,
                          clip_max: float = 3.0) -> torch.Tensor:
        """
        Compute FPO ratio from CFM loss differences.

        Args:
            old_loss: CFM loss from old policy [batch_size]
            new_loss: CFM loss from current policy [batch_size]
            clip_max: Maximum clipping value for stability

        Returns:
            ratio: FPO ratio exp(L_old - L_new) [batch_size]
        """
        loss_diff = new_loss.detach() - new_loss
        loss_diff = torch.clamp(loss_diff, -clip_max, clip_max)
        ratio = torch.exp(loss_diff)
        return ratio

    @staticmethod
    def clipped_surrogate_objective(ratio: torch.Tensor, advantages: torch.Tensor,
                                  clip_range: float) -> torch.Tensor:
        """
        Compute PPO-style clipped surrogate objective.

        Args:
            ratio: Policy ratio [batch_size]
            advantages: Advantage values [batch_size]
            clip_range: Clipping range (e.g., 0.2)

        Returns:
            loss: Clipped surrogate loss (negative for minimization)
        """
        unclipped_loss = - ratio * advantages
        clipped_loss = - torch.clamp(ratio, 1 - clip_range, 1 + clip_range) * advantages
        return torch.mean(torch.maximum(unclipped_loss, clipped_loss))