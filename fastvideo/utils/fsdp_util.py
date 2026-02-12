# ruff: noqa: E731
#This code file is from [https://github.com/hao-ai-lab/FastVideo], which is licensed under Apache License 2.0.


import functools
from functools import partial

import torch
from peft.utils.other import fsdp_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper)
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from fastvideo.models.mochi_hf.modeling_mochi import MochiTransformerBlock
from fastvideo.utils.load import get_no_split_modules

non_reentrant_wrapper = partial(
    checkpoint_wrapper,
    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
)

check_fn = lambda submodule: isinstance(submodule, MochiTransformerBlock)


def apply_fsdp_checkpointing(model, no_split_modules, p=1):
    # https://github.com/foundation-model-stack/fms-fsdp/blob/408c7516d69ea9b6bcd4c0f5efab26c0f64b3c2d/fms_fsdp/policies/ac_handler.py#L16
    """apply activation checkpointing to model
    returns None as model is updated directly
    """
    print("--> applying fdsp activation checkpointing...")
    block_idx = 0
    cut_off = 1 / 2
    # when passing p as a fraction number (e.g. 1/3), it will be interpreted
    # as a string in argv, thus we need eval("1/3") here for fractions.
    p = eval(p) if isinstance(p, str) else p

    def selective_checkpointing(submodule):
        nonlocal block_idx
        nonlocal cut_off

        if isinstance(submodule, no_split_modules):
            block_idx += 1
            if block_idx * p >= cut_off:
                cut_off += 1
                return True
        return False

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=non_reentrant_wrapper,
        check_fn=selective_checkpointing,
    )


def get_mixed_precision(master_weight_type="fp32"):
    weight_type = torch.float32 if master_weight_type == "fp32" else torch.bfloat16
    mixed_precision = MixedPrecision(
        param_dtype=weight_type,
        # Gradient communication precision.
        reduce_dtype=weight_type,
        # Buffer precision.
        buffer_dtype=weight_type,
        cast_forward_inputs=False,
    )
    return mixed_precision


def get_dit_fsdp_kwargs(
    transformer,
    sharding_strategy,
    use_lora=False,
    cpu_offload=False,
    master_weight_type="fp32",
):
    no_split_modules = get_no_split_modules(transformer)
    if use_lora:
        auto_wrap_policy = fsdp_auto_wrap_policy
    else:
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=no_split_modules,
        )

    # we use float32 for fsdp but autocast during training
    mixed_precision = get_mixed_precision(master_weight_type)

    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    device_id = torch.cuda.current_device()
    cpu_offload = (torch.distributed.fsdp.CPUOffload(
        offload_params=True) if cpu_offload else None)
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "cpu_offload": cpu_offload,
    }

    # Add LoRA-specific settings when LoRA is enabled
    if use_lora:
        fsdp_kwargs.update({
            "use_orig_params": False,  # Required for LoRA memory savings
            "sync_module_states": True,
        })

    return fsdp_kwargs, no_split_modules


def get_discriminator_fsdp_kwargs(master_weight_type="fp32"):
    auto_wrap_policy = None

    # Use existing mixed precision settings

    mixed_precision = get_mixed_precision(master_weight_type)
    sharding_strategy = ShardingStrategy.NO_SHARD
    device_id = torch.cuda.current_device()
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
    }

    return fsdp_kwargs


def get_qwen3vl_discriminator_fsdp_kwargs(
    discriminator,
    sharding_strategy="full",
    master_weight_type="fp32",
    cpu_offload=False,
):
    """
    Get FSDP configuration for Qwen3VL-based discriminator.

    The discriminator has:
    - Frozen vision encoder (Qwen3VLVisionBlock) - excluded from FSDP
    - Trainable language model (Qwen3VLTextDecoderLayer)
    - Trainable discriminator head

    Args:
        discriminator: The discriminator model instance
        sharding_strategy: FSDP sharding strategy ('full', 'hybrid_full', 'none')
        master_weight_type: Weight precision ('fp32' or 'bf16')

    Returns:
        fsdp_kwargs: Dictionary of FSDP configuration
        no_split_modules: Tuple of module classes that should not be split
    """

    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextDecoderLayer, Qwen3VLVisionBlock

    # No-split modules: wrap at language model decoder layer boundaries
    # Vision blocks are frozen so we don't need to wrap them
    no_split_modules = (Qwen3VLTextDecoderLayer,)

    # Auto-wrap policy: wrap at transformer decoder layer boundaries
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=no_split_modules,
    )

    # Mixed precision: use bf16 for params and buffers, fp32 for gradients
    mixed_precision = get_mixed_precision(master_weight_type)

    # Sharding strategy
    if sharding_strategy == "full":
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "hybrid_full":
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "none":
        sharding_strategy = ShardingStrategy.NO_SHARD
        auto_wrap_policy = None
    elif sharding_strategy == "hybrid_zero2":
        sharding_strategy = ShardingStrategy._HYBRID_SHARD_ZERO2

    device_id = torch.cuda.current_device()

    # Collect frozen modules to ignore
    ignored_modules = []
    if hasattr(discriminator, 'model') and hasattr(discriminator.model, 'visual'):
        # Check if visual encoder is frozen by checking if any parameter requires grad
        visual_params = list(discriminator.model.visual.parameters())
        if visual_params and not any(p.requires_grad for p in visual_params):
            ignored_modules.append(discriminator.model.visual)
            print(f"FSDP: Ignoring frozen visual encoder module")

    cpu_offload = (torch.distributed.fsdp.CPUOffload(
        offload_params=True) if cpu_offload else None)
    fsdp_kwargs = {
        "auto_wrap_policy": auto_wrap_policy,
        "mixed_precision": mixed_precision,
        "sharding_strategy": sharding_strategy,
        "device_id": device_id,
        "limit_all_gathers": True,
        "sync_module_states": True,  # Important for multi-GPU consistency
        "forward_prefetch": True,  # Improve throughput
        "ignored_modules": ignored_modules if ignored_modules else None,  # Ignore frozen vision encoder
        "cpu_offload": cpu_offload,
    }

    return fsdp_kwargs, no_split_modules
