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
except (ImportError, ValueError):
    from benchmarks.fno_3d import FNO3D


# --------------------------------------------------------------------------
# Self-contained 3D Darcy residuals so PINO3D doesn't depend on a specific
# physics-module API. Two variants: finite-difference (FD) and spectral (FFT).
# --------------------------------------------------------------------------

def _grad_3d_fd(u: torch.Tensor, dx: float, dy: float, dz: float):
    """Central differences with reflective padding on (B, C, D, H, W)."""
    u_x = F.pad(u, (1, 1, 0, 0, 0, 0), mode="reflect")
    u_y = F.pad(u, (0, 0, 1, 1, 0, 0), mode="reflect")
    u_z = F.pad(u, (0, 0, 0, 0, 1, 1), mode="reflect")
    dx_u = (u_x[:, :, :, :, 2:] - u_x[:, :, :, :, :-2]) / (2 * dx + 1e-6)
    dy_u = (u_y[:, :, :, 2:, :] - u_y[:, :, :, :-2, :]) / (2 * dy + 1e-6)
    dz_u = (u_z[:, :, 2:, :, :] - u_z[:, :, :-2, :, :]) / (2 * dz + 1e-6)
    return dx_u, dy_u, dz_u


def _div_3d_fd(fx, fy, fz, dx, dy, dz):
    fx_p = F.pad(fx, (1, 1, 0, 0, 0, 0), mode="reflect")
    fy_p = F.pad(fy, (0, 0, 1, 1, 0, 0), mode="reflect")
    fz_p = F.pad(fz, (0, 0, 0, 0, 1, 1), mode="reflect")
    return (
        (fx_p[:, :, :, :, 2:] - fx_p[:, :, :, :, :-2]) / (2 * dx + 1e-6)
        + (fy_p[:, :, :, 2:, :] - fy_p[:, :, :, :-2, :]) / (2 * dy + 1e-6)
        + (fz_p[:, :, 2:, :, :] - fz_p[:, :, :-2, :, :]) / (2 * dz + 1e-6)
    )


def _darcy_residual_3d_fd(p, K, dx, dy, dz):
    dpx, dpy, dpz = _grad_3d_fd(p, dx, dy, dz)
    return _div_3d_fd(-K * dpx, -K * dpy, -K * dpz, dx, dy, dz)


def _grad_3d_spectral(u: torch.Tensor, dx: float, dy: float, dz: float):
    B, C, D, H, W = u.shape
    u_ft = torch.fft.rfftn(u, dim=(-3, -2, -1))
    kz = torch.fft.fftfreq(D, d=dz, device=u.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=dy, device=u.device) * 2 * torch.pi
    kx = torch.fft.rfftfreq(W, d=dx, device=u.device) * 2 * torch.pi
    KZ = kz.view(D, 1, 1)
    KY = ky.view(1, H, 1)
    KX = kx.view(1, 1, -1)
    return (
        torch.fft.irfftn(1j * KX * u_ft, s=(D, H, W), dim=(-3, -2, -1)),
        torch.fft.irfftn(1j * KY * u_ft, s=(D, H, W), dim=(-3, -2, -1)),
        torch.fft.irfftn(1j * KZ * u_ft, s=(D, H, W), dim=(-3, -2, -1)),
    )


def _div_3d_spectral(fx, fy, fz, dx, dy, dz):
    B, C, D, H, W = fx.shape
    fx_ft = torch.fft.rfftn(fx, dim=(-3, -2, -1))
    fy_ft = torch.fft.rfftn(fy, dim=(-3, -2, -1))
    fz_ft = torch.fft.rfftn(fz, dim=(-3, -2, -1))
    kz = torch.fft.fftfreq(D, d=dz, device=fx.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=dy, device=fx.device) * 2 * torch.pi
    kx = torch.fft.rfftfreq(W, d=dx, device=fx.device) * 2 * torch.pi
    div_ft = 1j * (
        kx.view(1, 1, -1) * fx_ft
        + ky.view(1, H, 1) * fy_ft
        + kz.view(D, 1, 1) * fz_ft
    )
    return torch.fft.irfftn(div_ft, s=(D, H, W), dim=(-3, -2, -1))


def _darcy_residual_3d_spectral(p, K, dx, dy, dz):
    dpx, dpy, dpz = _grad_3d_spectral(p, dx, dy, dz)
    return _div_3d_spectral(-K * dpx, -K * dpy, -K * dpz, dx, dy, dz)


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
        p = pred[:, 0:1]
        K = x[:, 0:1]
        if self.residual_type == "spectral":
            res = _darcy_residual_3d_spectral(p, K, self.dx, self.dy, self.dz)
        else:
            res = _darcy_residual_3d_fd(p, K, self.dx, self.dy, self.dz)
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
