"""
FNO3D baseline (vanilla Fourier Neural Operator, 3D).

This is the comparison baseline for PI-JEPA on 3D tasks (CCSNet, FNO4CO2,
synthetic 3D Darcy, SPE10). It is intentionally a MINIMAL vanilla FNO:

  - Lift to hidden channels
  - L Fourier blocks (SpectralConv3d + 1x1 conv + GELU)
  - Project to output channels

No JEPA, no masking, no attention, no per-sub-operator structure. This
keeps the comparison apples-to-apples for the methodological claim.

For rectangular 3D inputs (B, C_in, D, H, W), it predicts (B, C_out, D, H, W).
Trained supervised on (x, y) pairs.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse the project's SpectralConv3d so the spectral operator is identical
# to PI-JEPA's. The DELTA is that FNO3D has no attention / no JEPA / no
# operator-split structure.
try:
    from ..models.fourier_encoder_3d import SpectralConv3d
except (ImportError, ValueError):
    # Allow flat-script invocation: `python scripts/...` adds PI-JEPA/ to
    # sys.path so we can also resolve via top-level `models`.
    from models.fourier_encoder_3d import SpectralConv3d


def _safe_num_groups(channels: int, max_groups: int = 8) -> int:
    """Largest divisor of `channels` <= max_groups (for GroupNorm safety)."""
    if channels < 1:
        return 1
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class FNOBlock3D(nn.Module):
    """One FNO3D block: spectral conv + skip 1x1 conv + GELU + norm."""

    def __init__(self, channels: int, modes: Tuple[int, int, int]):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, *modes)
        self.local = nn.Conv3d(channels, channels, kernel_size=1)
        self.norm = nn.GroupNorm(_safe_num_groups(channels), channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.spectral(x) + self.local(x)
        y = self.norm(y)
        return self.act(y)


class FNO3D(nn.Module):
    """Vanilla 3D Fourier Neural Operator.

    Args:
        in_channels: input channel count (e.g., 1 for permeability-only,
            12 for FNO4CO2's multi-channel inputs)
        out_channels: output channel count (typically 1 for pressure or
            saturation; can be higher for multi-output predictions)
        hidden_channels: width of internal FNO blocks
        n_blocks: number of FNO blocks
        modes: (k_d, k_h, k_w) Fourier modes retained
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_blocks: int = 4,
        modes: Tuple[int, int, int] = (8, 8, 8),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.modes = tuple(modes)

        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            FNOBlock3D(hidden_channels, self.modes) for _ in range(n_blocks)
        ])
        self.project = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W)
        Returns:
            (B, C_out, D, H, W)
        """
        if x.dim() != 5:
            raise ValueError(f"FNO3D expects 5D input (B,C,D,H,W); got shape {tuple(x.shape)}")
        x = self.lift(x)
        for block in self.blocks:
            x = block(x)
        return self.project(x)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Inference helper (alias for eval-time forward)."""
        return self.forward(x)


def build_fno3d_from_config(config: dict, in_channels: int, out_channels: int) -> FNO3D:
    """Convenience builder that pulls FNO3D hyperparams from a config dict."""
    bb = config.get("baselines", {}).get("fno3d", {})
    return FNO3D(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=bb.get("hidden_channels", 32),
        n_blocks=bb.get("n_blocks", 4),
        modes=tuple(bb.get("modes", [8, 8, 8])),
    )
