"""Two-Point Flux Approximation (TPFA) physics loss.

Uses harmonic-mean interface transmissibilities to compute pressure
equation residuals on structured grids. Handles permeability contrasts
up to 6 orders of magnitude via epsilon-stabilized denominators and
clamped permeability values.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class TPFALoss(nn.Module):
    """Two-Point Flux Approximation physics loss.

    Uses harmonic-mean interface transmissibilities:
    T_{i+1/2} = 2 K_i K_{i+1} / (K_i + K_{i+1})

    Handles permeability contrasts up to 6 orders of magnitude.
    """

    def __init__(self, dx: float = 1.0, dy: float = 1.0):
        """Initialize TPFA loss module.

        Args:
            dx: Grid spacing in x-direction.
            dy: Grid spacing in y-direction.
        """
        super().__init__()
        self.dx = dx
        self.dy = dy
        self.epsilon = 1e-10  # Denominator stabilization
        self.k_min = 1e-8  # Minimum permeability clamp
        self.k_max = 1e8  # Maximum permeability clamp

    def harmonic_mean_transmissibility(self, K: Tensor) -> Tuple[Tensor, Tensor]:
        """Compute T_x and T_y interface transmissibilities.

        Uses the harmonic mean formula:
        T_{i+1/2} = 2 * K_i * K_{i+1} / (K_i + K_{i+1} + epsilon)

        Args:
            K: Permeability field of shape (B, 1, H, W) with positive values.

        Returns:
            T_x: Transmissibility at x-interfaces, shape (B, 1, H, W-1).
                 T_x[:, :, :, j] is the transmissibility between cells j and j+1.
            T_y: Transmissibility at y-interfaces, shape (B, 1, H-1, W).
                 T_y[:, :, i, :] is the transmissibility between cells i and i+1.
        """
        # Clamp K to handle extreme contrasts (up to 6 orders of magnitude)
        K_clamped = K.clamp(self.k_min, self.k_max)

        # x-direction: interface between cell (i, j) and (i, j+1)
        K_left = K_clamped[:, :, :, :-1]  # (B, 1, H, W-1)
        K_right = K_clamped[:, :, :, 1:]  # (B, 1, H, W-1)
        T_x = 2.0 * K_left * K_right / (K_left + K_right + self.epsilon)
        # Scale by grid spacing: transmissibility = T / dx
        T_x = T_x / self.dx

        # y-direction: interface between cell (i, j) and (i+1, j)
        K_top = K_clamped[:, :, :-1, :]  # (B, 1, H-1, W)
        K_bottom = K_clamped[:, :, 1:, :]  # (B, 1, H-1, W)
        T_y = 2.0 * K_top * K_bottom / (K_top + K_bottom + self.epsilon)
        # Scale by grid spacing: transmissibility = T / dy
        T_y = T_y / self.dy

        return T_x, T_y

    def pressure_residual(self, p: Tensor, K: Tensor, q: Tensor) -> Tensor:
        """Compute TPFA pressure equation residual.

        The TPFA pressure equation at cell i is:
            sum_j T_ij (p_j - p_i) - q_i = 0

        For a structured 2D grid, each interior cell has 4 neighbors
        (left, right, top, bottom).

        Args:
            p: Pressure field of shape (B, 1, H, W).
            K: Permeability field of shape (B, 1, H, W).
            q: Source/sink term of shape (B, 1, H, W).

        Returns:
            residual: TPFA residual at each cell, shape (B, 1, H, W).
        """
        B, C, H, W = p.shape

        # Compute interface transmissibilities
        T_x, T_y = self.harmonic_mean_transmissibility(K)

        # Initialize residual accumulator
        residual = torch.zeros_like(p)

        # x-direction fluxes: T_x[j] * (p[j+1] - p[j])
        # Flux from right neighbor (j+1) into cell j
        flux_x = T_x * (p[:, :, :, 1:] - p[:, :, :, :-1])  # (B, 1, H, W-1)

        # Add flux contributions to residual:
        # Cell j receives flux_x[j] from right neighbor
        residual[:, :, :, :-1] += flux_x
        # Cell j+1 receives -flux_x[j] from left neighbor
        residual[:, :, :, 1:] -= flux_x

        # y-direction fluxes: T_y[i] * (p[i+1] - p[i])
        # Flux from bottom neighbor (i+1) into cell i
        flux_y = T_y * (p[:, :, 1:, :] - p[:, :, :-1, :])  # (B, 1, H-1, W)

        # Add flux contributions to residual:
        # Cell i receives flux_y[i] from bottom neighbor
        residual[:, :, :-1, :] += flux_y
        # Cell i+1 receives -flux_y[i] from top neighbor
        residual[:, :, 1:, :] -= flux_y

        # Subtract source term: residual = sum_j T_ij (p_j - p_i) - q_i
        residual = residual - q

        return residual

    def forward(self, fields: Tensor, K: Tensor, params: dict) -> Tensor:
        """Compute total TPFA physics loss.

        Args:
            fields: Decoded fields (B, C, H, W) where C >= 1.
                   Channel 0: pressure field.
            K: Permeability field (B, 1, H, W).
            params: Dictionary containing:
                - 'q': Source/sink term (B, 1, H, W), optional (defaults to 0).

        Returns:
            loss: Scalar loss tensor (mean squared residual).
        """
        # Extract pressure from fields
        p = fields[:, 0:1, :, :]  # (B, 1, H, W)

        # Get source term
        q = params.get("q", torch.zeros_like(p))

        # Compute TPFA pressure residual
        residual = self.pressure_residual(p, K, q)

        # Mean squared residual loss
        loss = (residual ** 2).mean()

        # Safety: replace NaN/Inf with finite values
        loss = torch.nan_to_num(loss, nan=0.0, posinf=1e6, neginf=0.0)

        return loss
