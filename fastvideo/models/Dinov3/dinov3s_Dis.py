"""
DINOv3-based discriminator for image quality assessment.
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.spectral_norm import SpectralNorm, spectral_norm
import torchvision.transforms as transforms


class ResidualBlock(nn.Module):
    """Residual block with scaling for stable training."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.ratio = 1 / np.sqrt(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.fn(x).add(x)).mul_(self.ratio)


class SpectralConv2d(nn.Conv2d):
    """Conv2d with spectral normalization."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        SpectralNorm.apply(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12)


class BatchNormLocal2d(nn.Module):
    """Local batch normalization for 2D inputs."""
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
        x = x.view(G, -1, x.size(-3), x.size(-2), x.size(-1))

        # Calculate stats over batch, height, width dimensions
        mean = x.mean([1, 3, 4], keepdim=True)
        var = x.var([1, 3, 4], keepdim=True, unbiased=False)
        x = (x - mean) / (torch.sqrt(var + self.eps))

        if self.affine:
            x = x * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return x.view(shape)


def make_block2d(channels: int, kernel_size: int, norm_type: str, norm_eps: float, using_spec_norm: bool) -> nn.Module:
    """Create a 2D convolutional block with normalization."""
    if norm_type == 'bn':
        norm = BatchNormLocal2d(channels, eps=norm_eps)
    elif norm_type == 'sbn':
        norm = nn.SyncBatchNorm(channels, eps=norm_eps, process_group=None)
    elif norm_type == 'gn':
        norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=norm_eps, affine=True)
    elif norm_type == 'ln':
        norm = nn.LayerNorm(channels)
    else:
        raise NotImplementedError(f"Normalization type {norm_type} not implemented")

    return nn.Sequential(
        (SpectralConv2d if using_spec_norm else nn.Conv2d)(
            channels, channels, kernel_size=kernel_size,
            padding=kernel_size // 2, padding_mode='circular'
        ),
        norm,
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
    )


class SimpleDiscriminatorHead(nn.Module):
    """Simple single linear layer discriminator head with spectral normalization."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.head = spectral_norm(nn.Linear(hidden_size, 1))
        self._init_weights()

    def _init_weights(self):
        """Initialize weights to produce outputs close to 0."""
        nn.init.normal_(self.head.weight, mean=0.0, std=0.01)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def make_transform(resize_size: int = 224):
    """Create image transform for DINOv3 input."""
    resize = transforms.Resize((resize_size, resize_size), antialias=True)
    
    def denormalize_and_normalize(x):
        # Convert from [-1, 1] to [0, 1]
        x = (x + 1.0) / 2.0
        # Apply ImageNet normalization
        normalize = transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        return normalize(x)
    
    return transforms.Compose([resize, denormalize_and_normalize])


class Dinov3sDisc(nn.Module):
    """DINOv3-S based discriminator with multi-scale feature extraction."""
    def __init__(
            self,
            repo_dir: str,
            weight_dir: str,
            kernel_size: int = 9,
            key_depths: Tuple[int, ...] = (2, 5, 8, 11),
            norm_type: str = 'bn',
            using_spec_norm: bool = True,
            norm_eps: float = 1e-6,
            num_residual_blocks: int = 1,
    ):
        super().__init__()
        encoder = torch.hub.load(repo_dir, 'dinov3_vits16', source='local', weights=weight_dir)

        self.key_depths = key_depths
        self.num_residual_blocks = num_residual_blocks
        encoder_C = encoder.embed_dim
        self.encoder = encoder

        heads = []
        for _ in range(len(self.key_depths)):
            heads.append(
                nn.Sequential(
                    make_block2d(encoder_C, kernel_size=1, norm_type=norm_type, norm_eps=norm_eps,
                               using_spec_norm=using_spec_norm),
                    *[ResidualBlock(make_block2d(encoder_C, kernel_size=kernel_size, norm_type=norm_type,
                                               norm_eps=norm_eps, using_spec_norm=using_spec_norm))
                      for _ in range(self.num_residual_blocks)],
                    (SpectralConv2d if using_spec_norm else nn.Conv2d)(encoder_C, 1, kernel_size=1, padding=0)
                )
            )

        self.heads = nn.ModuleList(heads)

    def forward(self, x, return_features=False):
        b, c, h, w = x.shape
        x_in = make_transform(resize_size=h)(x)
        feat_list = self.encoder.get_intermediate_layers(
            x_in,
            n=self.key_depths,
            reshape=True,
            norm=True,
        )

        res_list = []
        for feat, head in zip(feat_list, self.heads):
            res_list.append(head(feat).reshape(feat.shape[0], -1))

        concat_res = torch.cat(res_list, dim=1)
        if return_features:
            return concat_res, feat_list
        return concat_res

    def save_pretrained(self, path):
        """Save discriminator state dict."""
        torch.save(self.state_dict(), path)


class Dinov3bDisc(nn.Module):
    """DINOv3-B based discriminator with simple heads."""
    def __init__(
            self,
            repo_dir: str,
            weight_dir: str,
            key_depths: Tuple[int, ...] = (2, 5, 8, 11),
    ):
        super().__init__()
        encoder = torch.hub.load(repo_dir, 'dinov3_vitb16', source='local', weights=weight_dir)

        self.key_depths = key_depths
        encoder_C = encoder.embed_dim
        self.encoder = encoder

        heads = []
        for _ in range(len(self.key_depths)):
            heads.append(SimpleDiscriminatorHead(encoder_C))

        self.heads = nn.ModuleList(heads)

    def forward(self, x, return_features=False):
        b, c, h, w = x.shape
        x_in = make_transform(resize_size=h)(x)
        feat_list = self.encoder.get_intermediate_layers(
            x_in,
            n=self.key_depths,
            norm=True,
        )

        res_list = []
        for feat, head in zip(feat_list, self.heads):
            res_list.append(head(feat).reshape(feat.shape[0], -1))

        concat_res = torch.cat(res_list, dim=1)
        if return_features:
            return concat_res, feat_list
        return concat_res

    def save_pretrained(self, path):
        """Save discriminator state dict."""
        torch.save(self.state_dict(), path)
