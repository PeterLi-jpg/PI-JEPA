"""Latent Flux Module for enforcing inter-patch flux consistency in latent space.

Operates directly on predictor output embeddings z ∈ R^{B×N×D}
without requiring the decoder. Penalizes flux discontinuities across
patch boundaries to enforce physics consistency in embedding space.
"""

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


class LatentFluxModule(nn.Module):
    """Enforce inter-patch flux consistency in latent space.

    Operates directly on predictor output embeddings z ∈ R^{B×N×D}
    without requiring the decoder.

    For each shared boundary between adjacent patches, the flux predicted
    from one patch must equal the flux predicted from the neighboring patch.
    The module uses learned linear projections to map patch embeddings to
    flux values at each boundary direction.

    The key insight is that each patch embedding is projected to a flux value
    at its boundary. For a horizontal boundary between patches (i,j) and (i,j+1),
    the flux from patch (i,j) at its right edge should match the flux from
    patch (i,j+1) at its left edge. When adjacent patches have identical
    embeddings, the discontinuity is guaranteed to be zero because the same
    projection is applied to both sides.

    Args:
        embed_dim: Dimension of patch embeddings (D).
        grid_size: Number of patches per side (total patches = grid_size²).
        n_flux_heads: Number of flux heads (independent flux components per boundary).
    """

    def __init__(
        self,
        embed_dim: int = 384,
        grid_size: int = 8,  # patches per side
        n_flux_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.grid_size = grid_size
        self.n_flux_heads = n_flux_heads

        # Horizontal flux projection: maps each patch to its flux contribution
        # at horizontal (left-right) boundaries.
        # The discontinuity at boundary (i, j)↔(i, j+1) is:
        #   proj_h(z[i,j]) - proj_h(z[i,j+1])
        # This guarantees zero discontinuity when adjacent patches are identical.
        self.proj_h = nn.Linear(embed_dim, n_flux_heads)

        # Vertical flux projection: maps each patch to its flux contribution
        # at vertical (top-bottom) boundaries.
        # The discontinuity at boundary (i, j)↔(i+1, j) is:
        #   proj_v(z[i,j]) - proj_v(z[i+1,j])
        self.proj_v = nn.Linear(embed_dim, n_flux_heads)

    def compute_boundary_flux(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Project patch embeddings to flux values at inter-patch boundaries.

        For horizontal boundaries between columns j and j+1:
            flux_h[b, i, j, :] = proj_h(z[i,j]) - proj_h(z[i,j+1])

        For vertical boundaries between rows i and i+1:
            flux_v[b, i, j, :] = proj_v(z[i,j]) - proj_v(z[i+1,j])

        Args:
            z: Patch embeddings of shape (B, N, D) where N = grid_size².

        Returns:
            flux_h: (B, grid_size, grid_size-1, n_flux_heads) horizontal flux
                    discontinuities at vertical boundaries.
            flux_v: (B, grid_size-1, grid_size, n_flux_heads) vertical flux
                    discontinuities at horizontal boundaries.
        """
        B, N, D = z.shape
        G = self.grid_size
        assert N == G * G, f"Expected N={G*G} patches, got {N}"

        # Reshape to spatial grid: (B, G, G, D)
        z_grid = z.view(B, G, G, D)

        # Project all patches to horizontal flux values: (B, G, G, n_flux_heads)
        h_flux_all = self.proj_h(z_grid)

        # Horizontal discontinuity: difference between adjacent columns
        # flux from patch at column j minus flux from patch at column j+1
        flux_h = h_flux_all[:, :, :-1, :] - h_flux_all[:, :, 1:, :]
        # Shape: (B, G, G-1, n_flux_heads)

        # Project all patches to vertical flux values: (B, G, G, n_flux_heads)
        v_flux_all = self.proj_v(z_grid)

        # Vertical discontinuity: difference between adjacent rows
        # flux from patch at row i minus flux from patch at row i+1
        flux_v = v_flux_all[:, :-1, :, :] - v_flux_all[:, 1:, :, :]
        # Shape: (B, G-1, G, n_flux_heads)

        return flux_h, flux_v

    def forward(self, z_pred: Tensor) -> Tensor:
        """Compute flux discontinuity penalty across patch boundaries.

        For each shared boundary between adjacent patches, the flux
        predicted from the left patch must equal the flux from the right.
        The loss is the mean squared flux discontinuity across all boundaries.

        Args:
            z_pred: Predictor output embeddings of shape (B, N, D)
                    where N = grid_size².

        Returns:
            loss: Scalar loss tensor (mean squared flux discontinuity).
                  Non-negative by construction. Zero when all adjacent
                  patches have identical embeddings.
        """
        flux_h, flux_v = self.compute_boundary_flux(z_pred)

        # Mean squared discontinuity across all boundaries and flux heads
        loss_h = (flux_h ** 2).mean()
        loss_v = (flux_v ** 2).mean()

        loss = loss_h + loss_v

        return loss
