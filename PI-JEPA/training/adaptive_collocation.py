"""Adaptive collocation point sampling for physics loss evaluation.

Samples collocation points proportional to gradient magnitude, concentrating
physics evaluation near saturation fronts and high-gradient regions while
maintaining minimum density everywhere.
"""

import torch
from torch import Tensor


class AdaptiveCollocationSampler:
    """Sample collocation points proportional to gradient magnitude.

    Concentrates physics evaluation near saturation fronts and
    high-gradient regions while maintaining minimum density everywhere.

    This is a plain Python class (no learnable parameters). The caller is
    responsible for calling ``update_distribution`` every ``update_interval``
    steps; the sampler stores the interval for reference but does not enforce
    it internally.
    """

    def __init__(
        self,
        resolution: int = 64,
        n_points: int = 1024,
        min_density: float = 0.1,
        update_interval: int = 50,
    ):
        if resolution < 1:
            raise ValueError(f"resolution must be >= 1, got {resolution}")
        if n_points < 1:
            raise ValueError(f"n_points must be >= 1, got {n_points}")
        if min_density < 0.0:
            raise ValueError(f"min_density must be >= 0, got {min_density}")
        if update_interval < 1:
            raise ValueError(f"update_interval must be >= 1, got {update_interval}")

        self.resolution = resolution
        self.n_points = n_points
        self.min_density = min_density
        self.update_interval = update_interval

        # Internal probability distribution over spatial locations (H*W,)
        # None means uniform distribution (not yet updated)
        self._distribution: Tensor | None = None

    def update_distribution(self, field: Tensor) -> None:
        """Recompute sampling distribution from field gradients.

        Computes gradient magnitude using finite differences, averages over
        batch and channels, applies a minimum density floor, and normalizes
        to a valid probability distribution.

        Args:
            field: Predicted field tensor of shape (B, C, H, W) or (1, C, H, W).
        """
        if field.ndim != 4:
            raise ValueError(
                f"field must be 4D (B, C, H, W), got shape {field.shape}"
            )

        # Compute spatial gradients via finite differences
        # df/dx: difference along width (last dim)
        grad_x = torch.zeros_like(field)
        grad_x[..., :-1] = field[..., 1:] - field[..., :-1]
        # df/dy: difference along height (second-to-last dim)
        grad_y = torch.zeros_like(field)
        grad_y[..., :-1, :] = field[..., 1:, :] - field[..., :-1, :]

        # Gradient magnitude: sqrt((df/dx)^2 + (df/dy)^2)
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Average over batch and channels to get (H, W) map
        grad_mag_map = grad_mag.mean(dim=(0, 1))  # (H, W)

        H, W = grad_mag_map.shape
        N = H * W

        # Flatten gradient magnitude map
        prob_flat = grad_mag_map.reshape(-1)  # (H*W,)

        # Normalize to get initial probability distribution
        total = prob_flat.sum()
        if total > 0:
            prob_flat = prob_flat / total
        else:
            # Uniform if all gradients are zero
            prob_flat = torch.ones_like(prob_flat) / N

        # Apply minimum density floor and re-normalize.
        # We iteratively clamp and re-normalize to guarantee the floor holds.
        floor = self.min_density / N
        for _ in range(10):  # converges in 1-2 iterations typically
            below_floor = prob_flat < floor
            if not below_floor.any():
                break
            # Set below-floor entries to the floor value
            prob_flat = torch.clamp(prob_flat, min=floor)
            # Re-normalize
            prob_flat = prob_flat / prob_flat.sum()

        self._distribution = prob_flat

    def sample(self, batch_size: int) -> Tensor:
        """Sample collocation point indices.

        Uses the stored distribution to sample n_points indices via
        torch.multinomial. If the distribution hasn't been updated yet,
        uses a uniform distribution.

        Args:
            batch_size: Number of samples in the batch.

        Returns:
            Tensor of shape (batch_size, n_points, 2) where the last
            dimension contains (row, col) indices into the spatial grid.
        """
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        H = self.resolution
        W = self.resolution
        N = H * W

        if self._distribution is None:
            # Uniform distribution
            dist = torch.ones(N) / N
        else:
            dist = self._distribution

        # Sample flat indices for each batch element
        # torch.multinomial requires 2D input for batched sampling
        dist_expanded = dist.unsqueeze(0).expand(batch_size, -1)  # (B, H*W)
        flat_indices = torch.multinomial(
            dist_expanded, num_samples=self.n_points, replacement=True
        )  # (B, n_points)

        # Convert flat indices to (row, col) pairs
        rows = flat_indices // W  # (B, n_points)
        cols = flat_indices % W   # (B, n_points)

        # Stack to get (B, n_points, 2) with last dim = (row, col)
        indices = torch.stack([rows, cols], dim=-1)  # (B, n_points, 2)

        return indices
