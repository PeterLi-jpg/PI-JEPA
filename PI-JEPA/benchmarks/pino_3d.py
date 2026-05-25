"""
PINO3D baseline — FNO3D + physics-residual regularization during supervised training.

This is the direct head-to-head test of the paper's most important
ablation question: does PI-JEPA's win come from the OPERATOR-SPLIT
structure, or just from "supervised learning + physics residual"?

If PINO3D (supervised + physics) underperforms PI-JEPA (self-supervised
pretrain + operator-split + physics + fine-tune), the operator-split
structure is doing real work. If PINO3D matches, the structure is
incidental and the paper's central claim weakens.

PINO3D = FNO3D forward + supervised MSE loss + physics-residual loss term
(either FD or spectral, controlled by physics.residual_type config).
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .fno_3d import FNO3D
    from ..physics.darcy import (
        darcy_residual_3d, spectral_darcy_residual_3d,
    )
except (ImportError, ValueError):
    from benchmarks.fno_3d import FNO3D
    from physics.darcy import darcy_residual_3d, spectral_darcy_residual_3d


class PINO3D(nn.Module):
    """FNO3D + Darcy physics-residual head.

    forward(x): returns (B, C_out, D, H, W) prediction
    physics_loss(x, pred): returns scalar physics residual loss
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        hidden_channels: int = 32,
        n_blocks: int = 4,
        modes: Tuple[int, int, int] = (8, 8, 8),
        physics_weight: float = 0.1,
        residual_type: str = "fd",
        dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
    ):
        super().__init__()
        self.backbone = FNO3D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
            modes=modes,
        )
        self.physics_weight = physics_weight
        self.residual_type = residual_type
        self.dx, self.dy, self.dz = dx, dy, dz

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def physics_loss(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Compute Darcy residual loss using `pred` as the candidate pressure
        and the FIRST channel of `x` as the permeability conditioning."""
        # Use the first channel of x as K, first channel of pred as p
        p = pred[:, 0:1]
        K = x[:, 0:1]
        if self.residual_type == "spectral":
            res = spectral_darcy_residual_3d(p, K, q=None, dx=self.dx, dy=self.dy, dz=self.dz)
        else:
            res = darcy_residual_3d(p, K, q=None, dx=self.dx, dy=self.dy, dz=self.dz)
        return (res ** 2).mean()


def build_pino3d_from_config(config: dict, in_channels: int, out_channels: int) -> PINO3D:
    bb = config.get("baselines", {}).get("pino3d", {})
    return PINO3D(
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=bb.get("hidden_channels", 32),
        n_blocks=bb.get("n_blocks", 4),
        modes=tuple(bb.get("modes", [8, 8, 8])),
        physics_weight=bb.get("physics_weight", 0.1),
        residual_type=bb.get("residual_type", "fd"),
        dx=bb.get("dx", 1.0),
        dy=bb.get("dy", 1.0),
        dz=bb.get("dz", 1.0),
    )
