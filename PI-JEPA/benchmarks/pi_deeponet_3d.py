"""PI-DeepONet 3D — the missing reviewer-requested baseline (qZsm M3).

Reviewer qZsm specifically called out the absence of PI-DeepONet as the
most notable missing baseline, citing Wang/Wang/Perdikaris 2021
(arXiv:2103.10974) and the more recent PI-Latent-NO paper
(Karumuri et al., arXiv:2501.08428).

PI-DeepONet = DeepONet (branch-trunk operator architecture) + physics
residual loss term during supervised training (analogous to how PINO =
FNO + physics residual). Provides a head-to-head test of "operator
learning + physics-informed objective" that doesn't use FNO's spectral
inductive bias — answers whether PI-JEPA's wins are about the
self-supervised operator-split structure or just about physics-informed
operator learning.

3D variant: branch ingests a flattened (C, D, H, W) input field, trunk
ingests (z, y, x) coordinates, output is reconstructed as (B, C, D, H, W).
Mirrors PINO3D's Darcy physics residual for the regularization term.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Self-contained 3D Darcy residuals (same FD/spectral pair as PINO3D so
# PI-DeepONet doesn't depend on PINO3D's internals).
# --------------------------------------------------------------------------

def _grad_3d_fd(u: torch.Tensor, dx: float, dy: float, dz: float):
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


# --------------------------------------------------------------------------
# DeepONet3D — 3D extension of Brandon's 2D DeepONet
# --------------------------------------------------------------------------

class DeepONet3D(nn.Module):
    """3D DeepONet: flattened-input branch + 3D-coordinate trunk.

    Mirrors Brandon's 2D `DeepONet` in `benchmarks/deeponet.py` but
    operates on (B, C, D, H, W) inputs/outputs.

    Branch: flattened (C, D, H, W) field → MLP → (B, latent_dim).
    Trunk:  (z, y, x) coordinates on the grid → MLP → (DHW, latent_dim).
    Output: einsum gives (B, DHW) → reshape to (B, 1, D, H, W).

    We cache the trunk grid once per spatial shape to avoid rebuilding
    every forward; the cached grid moves with the model device.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        volume_shape: Tuple[int, int, int] = (24, 96, 96),
        hidden_dim: int = 256,
        latent_dim: int = 128,
    ):
        super().__init__()
        D, H, W = volume_shape
        branch_input_dim = int(in_channels) * D * H * W
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.volume_shape = (int(D), int(H), int(W))

        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.trunk = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Per-output-channel bias term; tiny but standard DeepONet practice.
        self.bias = nn.Parameter(torch.zeros(out_channels))
        # Cached trunk evaluation per (device, dtype). Built lazily.
        self._trunk_cache = None

    def _trunk_features(self, device, dtype) -> torch.Tensor:
        """Return (DHW, latent_dim) trunk features for the configured volume."""
        if (
            self._trunk_cache is not None
            and self._trunk_cache[0].device == device
            and self._trunk_cache[0].dtype == dtype
        ):
            return self._trunk_cache[0]
        D, H, W = self.volume_shape
        z = torch.linspace(0.0, 1.0, D, device=device, dtype=dtype)
        y = torch.linspace(0.0, 1.0, H, device=device, dtype=dtype)
        x = torch.linspace(0.0, 1.0, W, device=device, dtype=dtype)
        gz, gy, gx = torch.meshgrid(z, y, x, indexing="ij")
        coords = torch.stack([gz, gy, gx], dim=-1).reshape(-1, 3)  # (DHW, 3)
        feats = self.trunk(coords)  # (DHW, latent_dim)
        self._trunk_cache = (feats,)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 5:
            raise ValueError(
                f"DeepONet3D expects 5D input (B,C,D,H,W); got {tuple(x.shape)}"
            )
        B = x.shape[0]
        D, H, W = self.volume_shape
        # Resize input to the cached volume shape if it differs (DeepONet
        # branch dim is fixed at construction time).
        if x.shape[-3:] != (D, H, W):
            x = F.interpolate(x, size=(D, H, W), mode="trilinear",
                              align_corners=False)
        flat = x.reshape(B, -1)
        b = self.branch(flat)                            # (B, latent_dim)
        t = self._trunk_features(x.device, x.dtype)      # (DHW, latent_dim)
        out = torch.einsum("bi,ni->bn", b, t)            # (B, DHW)
        out = out.view(B, 1, D, H, W) + self.bias.view(1, -1, 1, 1, 1)[:, :1]
        # If out_channels > 1, broadcast (rare; DeepONet typically scalar).
        if self.out_channels > 1:
            out = out.expand(B, self.out_channels, D, H, W).clone()
        return out


# --------------------------------------------------------------------------
# PI-DeepONet 3D = DeepONet3D + Darcy physics residual loss
# --------------------------------------------------------------------------

class PIDeepONet3D(nn.Module):
    """PI-DeepONet 3D: DeepONet3D backbone with Darcy physics-residual loss.

    Mirrors PINO3D's API: `forward(x)` returns predictions; `physics_loss(x, pred)`
    returns a scalar Darcy residual term that the BaselineAdapter3D mixes
    into the supervised MSE with `physics_weight`.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        volume_shape: Tuple[int, int, int] = (24, 96, 96),
        hidden_dim: int = 256,
        latent_dim: int = 128,
        physics_weight: float = 0.1,
        residual_type: str = "fd",
        dx: float = 1.0, dy: float = 1.0, dz: float = 1.0,
    ):
        super().__init__()
        self.backbone = DeepONet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            volume_shape=volume_shape,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )
        self.physics_weight = float(physics_weight)
        self.residual_type = str(residual_type)
        self.dx, self.dy, self.dz = float(dx), float(dy), float(dz)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def physics_loss(self, x: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        """Darcy residual on the predicted pressure (channel 0) using the
        first input channel as permeability."""
        p = pred[:, 0:1]
        K = x[:, 0:1]
        if self.residual_type == "spectral":
            res = _darcy_residual_3d_spectral(p, K, self.dx, self.dy, self.dz)
        else:
            res = _darcy_residual_3d_fd(p, K, self.dx, self.dy, self.dz)
        return (res ** 2).mean()


def build_pi_deeponet_3d_from_config(config: dict, in_channels: int,
                                     out_channels: int) -> PIDeepONet3D:
    bb = config.get("baselines", {}).get("pi_deeponet_3d", {})
    return PIDeepONet3D(
        in_channels=in_channels,
        out_channels=out_channels,
        volume_shape=tuple(bb.get("volume_shape", [24, 96, 96])),
        hidden_dim=bb.get("hidden_dim", 256),
        latent_dim=bb.get("latent_dim", 128),
        physics_weight=bb.get("physics_weight", 0.1),
        residual_type=bb.get("residual_type", "fd"),
        dx=bb.get("dx", 1.0),
        dy=bb.get("dy", 1.0),
        dz=bb.get("dz", 1.0),
    )
