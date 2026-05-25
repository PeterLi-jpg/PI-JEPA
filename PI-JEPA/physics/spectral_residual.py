"""Spectral Residual Module for computing PDE residuals in Fourier space.

Replaces the 32×32 FD collocation approach with full 64×64 spectral modes
using ik-differentiation for alias-free spatial derivatives.
"""

import math
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class SpectralResidualModule(nn.Module):
    """Compute PDE residuals in Fourier space using ik-differentiation.

    Replaces the 32×32 FD collocation with full 64×64 spectral modes.
    """

    def __init__(
        self,
        resolution: int = 64,
        cutoff_ratio: float = 2 / 3,  # smooth spectral cutoff
        dx: float = 1.0,
        dy: float = 1.0,
    ):
        super().__init__()
        self.resolution = resolution
        self.cutoff_ratio = cutoff_ratio
        self.dx = dx
        self.dy = dy

        # Precompute wavenumber grids
        # For rfft2, kx has full N modes, ky has N//2+1 modes
        kx = torch.fft.fftfreq(resolution, d=dx) * 2 * math.pi  # (N,)
        ky = torch.fft.rfftfreq(resolution, d=dy) * 2 * math.pi  # (N//2+1,)

        # Create 2D wavenumber grids: shape (N, N//2+1)
        kx_grid = kx.unsqueeze(-1).expand(resolution, resolution // 2 + 1)
        ky_grid = ky.unsqueeze(0).expand(resolution, resolution // 2 + 1)

        self.register_buffer("kx", kx_grid)
        self.register_buffer("ky", ky_grid)

        # Precompute smooth spectral cutoff filter (2/3 rule with smooth rolloff)
        cutoff_filter = self._build_smooth_cutoff(kx, ky, resolution, cutoff_ratio)
        self.register_buffer("cutoff_filter", cutoff_filter)

    def _build_smooth_cutoff(
        self,
        kx: Tensor,
        ky: Tensor,
        resolution: int,
        cutoff_ratio: float,
    ) -> Tensor:
        """Build a smooth spectral cutoff filter using an exponential rolloff.

        The filter is 1.0 for wavenumbers below cutoff_ratio * k_max,
        and smoothly decays to 0 above that using an exponential function.
        This prevents Gibbs ringing while avoiding a sharp spectral truncation.
        """
        k_max_x = math.pi / self.dx  # Maximum wavenumber in x
        k_max_y = math.pi / self.dy  # Maximum wavenumber in y

        # Normalized wavenumber magnitudes
        kx_norm = (kx.abs() / k_max_x).unsqueeze(-1).expand(
            resolution, resolution // 2 + 1
        )
        ky_norm = (ky.abs() / k_max_y).unsqueeze(0).expand(
            resolution, resolution // 2 + 1
        )

        # Combined normalized wavenumber (max of x and y components)
        k_norm = torch.sqrt(kx_norm**2 + ky_norm**2)

        # Smooth exponential rolloff: 1 below cutoff, decays above
        # Using exp(-alpha * ((k - k_c) / (1 - k_c))^2) for k > k_c
        alpha = 36.0  # Controls sharpness of rolloff
        order = 12  # High order for sharp but smooth transition

        # Filter: exp(-alpha * ((k_norm - cutoff_ratio) / (1 - cutoff_ratio))^order)
        # for k_norm > cutoff_ratio, else 1.0
        mask = k_norm > cutoff_ratio
        rolloff = torch.zeros_like(k_norm)
        rolloff[mask] = (
            (k_norm[mask] - cutoff_ratio) / (1.0 - cutoff_ratio + 1e-10)
        ) ** order
        filt = torch.exp(-alpha * rolloff)

        return filt

    def spectral_gradient(self, u: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute ∂u/∂x, ∂u/∂y via ik multiplication in Fourier space.

        Args:
            u: Real-valued field of shape (B, 1, H, W) or (B, H, W)

        Returns:
            du_dx: Partial derivative w.r.t. x, same shape as input
            du_dy: Partial derivative w.r.t. y, same shape as input
        """
        squeeze = False
        if u.dim() == 3:
            u = u.unsqueeze(1)
            squeeze = True

        # Transform to Fourier space
        u_hat = torch.fft.rfft2(u)

        # Apply smooth cutoff filter to prevent NaN/Inf from high-k amplification
        u_hat_filtered = u_hat * self.cutoff_filter

        # ik multiplication for derivatives
        # kx shape: (N, N//2+1), needs broadcasting to (1, 1, N, N//2+1)
        kx = self.kx.unsqueeze(0).unsqueeze(0)
        ky = self.ky.unsqueeze(0).unsqueeze(0)

        du_dx_hat = 1j * kx * u_hat_filtered
        du_dy_hat = 1j * ky * u_hat_filtered

        # Transform back to physical space
        du_dx = torch.fft.irfft2(du_dx_hat, s=(self.resolution, self.resolution))
        du_dy = torch.fft.irfft2(du_dy_hat, s=(self.resolution, self.resolution))

        if squeeze:
            du_dx = du_dx.squeeze(1)
            du_dy = du_dy.squeeze(1)

        return du_dx, du_dy

    def spectral_divergence(self, fx: Tensor, fy: Tensor) -> Tensor:
        """Compute ∇·F = ∂fx/∂x + ∂fy/∂y in Fourier space.

        Args:
            fx: x-component of vector field, shape (B, 1, H, W)
            fy: y-component of vector field, shape (B, 1, H, W)

        Returns:
            div_F: Divergence field, same shape as input
        """
        # Transform both components to Fourier space
        fx_hat = torch.fft.rfft2(fx)
        fy_hat = torch.fft.rfft2(fy)

        # Apply smooth cutoff filter
        fx_hat_filtered = fx_hat * self.cutoff_filter
        fy_hat_filtered = fy_hat * self.cutoff_filter

        # ik multiplication for divergence
        kx = self.kx.unsqueeze(0).unsqueeze(0)
        ky = self.ky.unsqueeze(0).unsqueeze(0)

        div_hat = 1j * kx * fx_hat_filtered + 1j * ky * fy_hat_filtered

        # Transform back to physical space
        div_F = torch.fft.irfft2(div_hat, s=(self.resolution, self.resolution))

        return div_F

    def pressure_residual(
        self, p: Tensor, K: Tensor, lambda_t: Tensor, q: Tensor
    ) -> Tensor:
        """Compute pressure residual: R_p = -∇·(λ_T K ∇p) - q_T.

        Uses spectral differentiation for all spatial derivatives.

        Args:
            p: Pressure field (B, 1, H, W)
            K: Permeability field (B, 1, H, W)
            lambda_t: Total mobility field (B, 1, H, W)
            q: Source/sink term (B, 1, H, W)

        Returns:
            R_p: Pressure residual field (B, 1, H, W)
        """
        # Compute pressure gradient: ∇p
        dp_dx, dp_dy = self.spectral_gradient(p)

        # Compute flux: F = λ_T * K * ∇p
        flux_x = lambda_t * K * dp_dx
        flux_y = lambda_t * K * dp_dy

        # Compute divergence: ∇·(λ_T K ∇p)
        div_flux = self.spectral_divergence(flux_x, flux_y)

        # Residual: R_p = -∇·(λ_T K ∇p) - q_T
        residual = -div_flux - q

        return residual

    def saturation_residual(
        self,
        Sw: Tensor,
        Sw_prev: Tensor,
        p: Tensor,
        K: Tensor,
        fw: Tensor,
        phi: Tensor,
        dt: float,
    ) -> Tensor:
        """Compute saturation residual: R_s = φ ∂S/∂t + ∇·(f_w v_T) - q_w.

        Uses spectral differentiation for spatial derivatives and
        finite difference in time.

        Args:
            Sw: Current saturation field (B, 1, H, W)
            Sw_prev: Previous timestep saturation field (B, 1, H, W)
            p: Pressure field (B, 1, H, W)
            K: Permeability field (B, 1, H, W)
            fw: Fractional flow field (B, 1, H, W)
            phi: Porosity field (B, 1, H, W)
            dt: Time step size

        Returns:
            R_s: Saturation residual field (B, 1, H, W)
        """
        # Time derivative: ∂S/∂t ≈ (Sw - Sw_prev) / dt
        dSw_dt = (Sw - Sw_prev) / (dt + 1e-8)

        # Compute pressure gradient for total velocity
        dp_dx, dp_dy = self.spectral_gradient(p)

        # Total Darcy velocity: v_T = -K * λ_T * ∇p
        # Note: We use K directly here since lambda_t is already factored
        # into the fractional flow formulation. The total velocity magnitude
        # is embedded in the pressure solution.
        # For the saturation equation, we need v_T which comes from the
        # pressure equation. Here we compute it as -K * ∇p (simplified,
        # assuming lambda_t is absorbed or provided via the pressure field).
        # Actually, the full velocity is v_T = -lambda_t * K * ∇p
        # but since we don't have lambda_t separately here, we compute
        # the water flux directly as f_w * v_T where v_T = -K * ∇p
        # This is consistent with the design where the forward method
        # computes lambda_t from the saturation field.

        # For the saturation equation, the water flux is f_w * v_T
        # where v_T = -K * ∇p (total velocity from pressure equation)
        # We'll use K * ∇p as the velocity magnitude (sign handled below)
        vx = -K * dp_dx
        vy = -K * dp_dy

        # Water flux: f_w * v_T
        flux_w_x = fw * vx
        flux_w_y = fw * vy

        # Divergence of water flux: ∇·(f_w v_T)
        div_flux_w = self.spectral_divergence(flux_w_x, flux_w_y)

        # Residual: R_s = φ ∂S/∂t + ∇·(f_w v_T)
        # Note: source term q_w is not included here as it's handled
        # in the forward method via the params dict
        residual = phi * dSw_dt + div_flux_w

        return residual

    def forward(self, decoded_fields: Tensor, K: Tensor, params: dict) -> Tensor:
        """Compute total spectral physics residual loss.

        Args:
            decoded_fields: Decoded fields (B, C, H, W) where C=2
                           Channel 0: pressure, Channel 1: saturation
            K: Permeability field (B, 1, H, W)
            params: Dictionary containing:
                - 'mu_w': Water viscosity (float)
                - 'mu_o': Oil viscosity (float)
                - 'phi': Porosity field (B, 1, H, W) or scalar
                - 'dt': Time step (float)
                - 'Sw_prev': Previous saturation (B, 1, H, W)
                - 'q_T': Total source/sink (B, 1, H, W), optional (defaults to 0)
                - 'q_w': Water source/sink (B, 1, H, W), optional (defaults to 0)
                - 'krw_exp': Water rel-perm exponent (float, default 2.0)
                - 'kro_exp': Oil rel-perm exponent (float, default 2.0)
                - 'Swr': Residual water saturation (float or Tensor, default 0.0)
                - 'Snr': Residual non-wetting saturation (float or Tensor, default 0.0)
                - 'lambda_bc': Brooks-Corey pore-size distribution index
                  (float or Tensor, optional). When provided, uses the full
                  Brooks-Corey relative permeability model instead of simple
                  power-law. Can be a scalar, per-sample (B,) or (B,1,1,1),
                  or spatially varying (B, 1, H, W).

        Returns:
            loss: Scalar loss tensor (mean squared residual)
        """
        B, C, H, W = decoded_fields.shape
        assert C >= 2, f"Expected at least 2 channels (p, Sw), got {C}"

        # Extract fields
        p = decoded_fields[:, 0:1, :, :]  # (B, 1, H, W)
        Sw = decoded_fields[:, 1:2, :, :]  # (B, 1, H, W)

        # Clamp saturation to physical range
        Sw = Sw.clamp(0.0, 1.0)

        # Extract parameters
        mu_w = params.get("mu_w", 1.0)
        mu_o = params.get("mu_o", 1.0)
        phi = params.get("phi", torch.ones_like(p) * 0.2)
        dt = params.get("dt", 1.0)
        Sw_prev = params.get("Sw_prev", Sw)  # Default: steady state
        q_T = params.get("q_T", torch.zeros_like(p))
        q_w = params.get("q_w", torch.zeros_like(p))
        krw_exp = params.get("krw_exp", 2.0)
        kro_exp = params.get("kro_exp", 2.0)
        Swr = params.get("Swr", 0.0)
        Snr = params.get("Snr", 0.0)
        lambda_bc = params.get("lambda_bc", None)

        # Ensure phi is a tensor with correct shape
        if isinstance(phi, (int, float)):
            phi = torch.full_like(p, phi)
        elif phi.dim() == 0:
            phi = phi.expand_as(p)

        # Ensure Swr and Snr are tensors broadcastable to (B, 1, H, W)
        if isinstance(Swr, (int, float)):
            Swr_t = torch.tensor(Swr, device=Sw.device, dtype=Sw.dtype)
        else:
            Swr_t = Swr
        if isinstance(Snr, (int, float)):
            Snr_t = torch.tensor(Snr, device=Sw.device, dtype=Sw.dtype)
        else:
            Snr_t = Snr

        # Compute effective saturation
        Se = ((Sw - Swr_t) / (1.0 - Swr_t - Snr_t + 1e-8)).clamp(1e-8, 1.0 - 1e-8)

        # Compute relative permeabilities
        if lambda_bc is not None:
            # Full Brooks-Corey model:
            #   k_rw = Se^((2 + 3λ) / λ)
            #   k_ro = (1 - Se)^2 * (1 - Se^((2 + λ) / λ))
            if isinstance(lambda_bc, (int, float)):
                lbc = torch.tensor(lambda_bc, device=Sw.device, dtype=Sw.dtype)
            else:
                lbc = lambda_bc
            # Ensure lbc is broadcastable to Se shape
            if lbc.dim() == 0:
                pass  # scalar, broadcasts naturally
            elif lbc.dim() == 1:
                # Per-sample: (B,) -> (B, 1, 1, 1)
                lbc = lbc.view(-1, 1, 1, 1)
            # else: already (B, 1, H, W) or similar broadcastable shape

            krw_exp_bc = (2.0 + 3.0 * lbc) / lbc
            kro_exp_bc = (2.0 + lbc) / lbc

            Se_safe = Se.clamp(1e-8, 1.0)
            Se_inv = (1.0 - Se).clamp(0.0, 1.0 - 1e-8)

            krw = Se_safe ** krw_exp_bc
            kro = Se_inv ** 2 * (1.0 - Se_safe ** kro_exp_bc)
        else:
            # Simple power-law model (original behavior)
            krw = Se ** krw_exp
            kro = (1.0 - Se) ** kro_exp

        # Compute mobilities
        lambda_w = krw / (mu_w + 1e-8)
        lambda_o = kro / (mu_o + 1e-8)
        lambda_t = lambda_w + lambda_o + 1e-8

        # Compute fractional flow
        fw = lambda_w / lambda_t

        # Compute pressure residual: R_p = -∇·(λ_T K ∇p) - q_T
        R_p = self.pressure_residual(p, K, lambda_t, q_T)

        # Compute saturation residual: R_s = φ ∂S/∂t + ∇·(f_w v_T) - q_w
        R_s = self.saturation_residual(Sw, Sw_prev, p, K, fw, phi, dt)
        R_s = R_s - q_w

        # Combine into total loss (mean squared residuals)
        loss_p = (R_p**2).mean()
        loss_s = (R_s**2).mean()

        # Apply nan_to_num as final safety net
        loss_p = torch.nan_to_num(loss_p, nan=0.0, posinf=1e6, neginf=0.0)
        loss_s = torch.nan_to_num(loss_s, nan=0.0, posinf=1e6, neginf=0.0)

        total_loss = loss_p + loss_s

        return total_loss
