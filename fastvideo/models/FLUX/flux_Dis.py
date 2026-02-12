"""
FLUX-based discriminator for image quality assessment.
"""
from typing import Optional, Dict, Any, List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import SpectralNorm, spectral_norm
from diffusers.utils import logging, USE_PEFT_BACKEND, scale_lora_layers, unscale_lora_layers
from diffusers.models.normalization import LayerNorm

logger = logging.get_logger(__name__)


class ResidualBlock(nn.Module):
    """Residual block with scaling for stable training."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x) + x) / np.sqrt(2)


class SpectralConv1d(nn.Conv1d):
    """Conv1d with spectral normalization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal(nn.Module):
    """Local batch normalization."""
    def __init__(self, num_features: int, affine: bool = True, virtual_bs: int = 8, eps: float = 1e-5):
        super().__init__()
        self.virtual_bs = virtual_bs
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()

        # Reshape batch into groups
        G = np.ceil(x.size(0) / self.virtual_bs).astype(int)
        x = x.view(G, -1, x.size(-2), x.size(-1))

        # Calculate stats
        mean = x.mean([1, 3], keepdim=True)
        var = x.var([1, 3], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None] + self.bias[None, :, None]

        return x.view(shape)


def make_block(channels: int, kernel_size: int) -> nn.Module:
    """Create a 1D convolutional block."""
    return nn.Sequential(
        SpectralConv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            padding_mode='circular',
        ),
        BatchNormLocal(channels),
        nn.LeakyReLU(0.2, True),
    )


class SimpleDiscriminatorHead(nn.Module):
    """Simple single linear layer discriminator head with layer norm."""
    def __init__(
        self,
        hidden_size: int,
        elementwise_affine: bool = True,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        self.norm = LayerNorm(hidden_size, eps, elementwise_affine, bias)
        self.head = spectral_norm(nn.Linear(hidden_size, 1))
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to produce outputs close to 0."""
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        return self.head(x)

def flux_transformer_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
        block_hooks: List = None,
):
    """
            The [`FluxTransformer2DModel`] forward method.

            Args:
                hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
                    Input `hidden_states`.
                encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
                    Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
                pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                    from the embeddings of input conditions.
                timestep ( `torch.LongTensor`):
                    Used to indicate denoising step.
                block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                    A list of tensors that if specified are added to the residuals of transformer blocks.
                joint_attention_kwargs (`dict`, *optional*):
                    A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                    `self.processor` in
                    [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                    tuple.

            Returns:
                If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
                `tuple` where the first element is the sample tensor.
            """
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
        ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
        ip_hidden_states = self.encoder_hid_proj(ip_adapter_image_embeds)
        joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            # For Xlabs ControlNet.
            if controlnet_blocks_repeat:
                hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                )
            else:
                hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    feat_list = []
    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

        # controlnet residual
        if controlnet_single_block_samples is not None:
            interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_single_block_samples[index_block // interval_control]

        if index_block in block_hooks:
            feat_list.append(hidden_states)

    return feat_list

class FluxTransformer2DModelDiscriminator(nn.Module):
    def __init__(
        self,
        pretrained_path: str,
        checkpoint_path: Optional[str] = None,
        is_multiscale: bool = False,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize FLUX discriminator.

        Args:
            pretrained_path: Path to pretrained FLUX model (HuggingFace format) - required for architecture
            checkpoint_path: Optional path to checkpoint file (.pt, .safetensors, or directory)
                            to override weights from a trained policy model
            is_multiscale: If True, extract features from multiple blocks [6, 13, 20, 27, 37]
                          If False, only extract from final block
            torch_dtype: Data type for model parameters
        """
        super().__init__()

        # Always load transformer architecture from pretrained path
        logger.info(f"Loading FLUX discriminator architecture from {pretrained_path}")
        from fastvideo.models.flux_hf.transformer_flux import FluxTransformer2DModel
        self.transformer = FluxTransformer2DModel.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype
        )
        self.transformer.set_attention_backend("flash")

        # Replace forward method with custom one that extracts features
        self.transformer.forward = flux_transformer_forward

        # Optionally load checkpoint weights BEFORE removing norm_out/proj_out
        # (checkpoint contains these layers, so loading must happen while they exist)
        if checkpoint_path is not None:
            self.load_checkpoint_weights(checkpoint_path)

        # Configure which blocks to extract features from
        self.is_multiscale = is_multiscale
        if is_multiscale:
            # Multi-scale: extract from multiple blocks
            self.block_hooks = set([6, 13, 20, 27, 37])
        else:
            # Single-scale: extract from final block only
            final_block_idx = len(self.transformer.single_transformer_blocks) - 1
            self.block_hooks = set([final_block_idx])

        # Remove unused output layers AFTER checkpoint loading
        # (discriminator doesn't need image reconstruction)
        self.transformer.norm_out = None
        self.transformer.proj_out = None

        # Create discriminator heads for each extracted feature
        heads = []
        for _ in range(len(self.block_hooks)):
            heads.append(SimpleDiscriminatorHead(self.transformer.inner_dim, eps=1e-6, elementwise_affine=False))
        self.heads = nn.ModuleList(heads)

    def load_checkpoint_weights(self, checkpoint_path: str) -> None:
        """
        Load weights from a policy model checkpoint into the discriminator backbone.

        Supports multiple checkpoint formats:
        - .pt files (PyTorch checkpoints)
        - .safetensors files
        - Directories containing diffusion_pytorch_model.safetensors

        Args:
            checkpoint_path: Path to checkpoint file or directory
        """
        from safetensors.torch import load_file

        checkpoint_path = Path(checkpoint_path)

        logger.info(f"Loading discriminator backbone weights from {checkpoint_path}")

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

        # Clean state dict (remove common prefixes)
        cleaned_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("_fsdp_wrapped_module.", "")
            new_key = new_key.replace("module.", "")
            cleaned_state_dict[new_key] = value

        # Load into transformer (strict=False because we don't have norm_out/proj_out)
        missing_keys, unexpected_keys = self.transformer.load_state_dict(cleaned_state_dict, strict=False)

        # Filter expected missing keys (norm_out, proj_out which we intentionally removed)
        expected_missing = {"norm_out.linear.weight", "norm_out.linear.bias", "proj_out.weight", "proj_out.bias"}
        unexpected_missing = [k for k in missing_keys if k not in expected_missing]

        if unexpected_missing:
            logger.warning(f"Unexpected missing keys: {unexpected_missing}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys in checkpoint: {len(unexpected_keys)}")

        logger.info("Discriminator backbone weights loaded successfully")

    def freeze_backbone(self) -> None:
        """
        Freeze the transformer backbone parameters, keeping only discriminator heads trainable.
        """
        for param in self.transformer.parameters():
            param.requires_grad = False
        logger.info("FLUX discriminator backbone frozen (only heads are trainable)")

    def enable_gradient_checkpointing(self) -> None:
        """
        Enable gradient checkpointing for the transformer backbone to save memory.
        """
        if hasattr(self.transformer, 'enable_gradient_checkpointing'):
            self.transformer.enable_gradient_checkpointing()
            logger.info("FLUX discriminator gradient checkpointing enabled")
        else:
            logger.warning("Gradient checkpointing not available for this transformer")

    @property
    def model(self):
        return self.transformer

    def forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: torch.Tensor = None,
            pooled_projections: torch.Tensor = None,
            timestep: torch.LongTensor = None,
            img_ids: torch.Tensor = None,
            txt_ids: torch.Tensor = None,
            guidance: torch.Tensor = None,
            joint_attention_kwargs: Optional[Dict[str, Any]] = None,
            controlnet_block_samples=None,
            controlnet_single_block_samples=None,
            return_dict: bool = True,
            controlnet_blocks_repeat: bool = False,
    ):
        feat_list = self.transformer.forward(
            self.transformer,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            pooled_projections=pooled_projections,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
            guidance=guidance,
            joint_attention_kwargs=joint_attention_kwargs,
            controlnet_block_samples=controlnet_block_samples,
            controlnet_single_block_samples=controlnet_single_block_samples,
            return_dict=return_dict,
            controlnet_blocks_repeat=controlnet_blocks_repeat,
            block_hooks=self.block_hooks
        )

        res_list = []
        for feat, head in zip(feat_list, self.heads):
            res_list.append(head(feat).reshape(feat.shape[0], -1))
        concat_res = torch.cat(res_list, dim=1)

        return concat_res

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path)