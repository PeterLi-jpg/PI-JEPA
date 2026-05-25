"""
U-FNO3D baseline — FNO3D with appended mini-UNet path.

U-FNO (Wen et al. 2022) is the canonical CO2-storage baseline. It adds a
small U-Net branch alongside each Fourier block to better capture sharp
saturation fronts that pure FNO smooths over. This is the baseline
reviewers in the CCS subfield will explicitly look for.

This minimal 3D version: after each FNO block, add a parallel 3D U-Net
(2 down + 2 up levels) of the same channel width, sum with the FNO output,
then activate.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..models.fourier_encoder import SpectralConv3d, _safe_num_groups
except (ImportError, ValueError):
    from models.fourier_encoder import SpectralConv3d, _safe_num_groups


class _MiniUNet3D(nn.Module):
    """2-level 3D U-Net path with channel-preserving in/out."""

    def __init__(self, channels: int):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_safe_num_groups(channels), channels),
            nn.GELU(),
        )
        self.down2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_safe_num_groups(channels), channels),
            nn.GELU(),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2),
            nn.GroupNorm(_safe_num_groups(channels), channels),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2),
            nn.GroupNorm(_safe_num_groups(channels), channels),
            nn.GELU(),
        )

    def forward(self, x):
        D_in, H_in, W_in = x.shape[-3:]
        d1 = self.down1(x)
        d2 = self.down2(d1)
        u1 = self.up1(d2)
        # u1 may have a different shape than d1 if D/H/W are odd
        if u1.shape[-3:] != d1.shape[-3:]:
            u1 = F.interpolate(u1, size=d1.shape[-3:], mode="trilinear", align_corners=False)
        u1 = u1 + d1
        u2 = self.up2(u1)
        if u2.shape[-3:] != x.shape[-3:]:
            u2 = F.interpolate(u2, size=x.shape[-3:], mode="trilinear", align_corners=False)
        return u2


class UFNOBlock3D(nn.Module):
    """FNO block + parallel mini-UNet, summed and activated."""

    def __init__(self, channels: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, *modes)
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.unet = _MiniUNet3D(channels)
        self.norm = nn.GroupNorm(_safe_num_groups(channels), channels)
        self.act = nn.GELU()

    def forward(self, x):
        y = self.spectral(x) + self.local(x) + self.unet(x)
        return self.act(self.norm(y))


class UFNO3D(nn.Module):
    """3D U-FNO: stack of UFNOBlock3D between lift and project."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_blocks: int = 4,
        modes: Tuple[int, int, int] = (8, 8, 8),
    ):
        super().__init__()
        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            UFNOBlock3D(hidden_channels, tuple(modes)) for _ in range(n_blocks)
        ])
        self.project = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(f"UFNO3D expects 5D input; got {tuple(x.shape)}")
        x = self.lift(x)
        for blk in self.blocks:
            x = blk(x)
        return self.project(x)


def build_ufno3d_from_config(config: dict, in_channels: int, out_channels: int) -> UFNO3D:
    bb = config.get("baselines", {}).get("ufno3d", {})
    return UFNO3D(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=bb.get("hidden_channels", 32),
        n_blocks=bb.get("n_blocks", 4),
        modes=tuple(bb.get("modes", [8, 8, 8])),
    )
