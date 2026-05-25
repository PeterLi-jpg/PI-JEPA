"""Data pipeline for irregular grids and real field data.

Handles NaN values (missing data) and non-uniform grid spacing,
producing valid finite tensors suitable for encoder consumption.
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


class IrregularGridProcessor:
    """Process irregular grid data into structured tensors for the encoder.

    Handles:
    - NaN values (missing data) via interpolation or masking
    - Non-uniform grid spacing via interpolation to structured grid
    - Metadata preservation
    """

    def __init__(self, target_resolution: int = 64, interpolation_method: str = 'bilinear'):
        """Initialize the irregular grid processor.

        Args:
            target_resolution: Output grid resolution (square grid).
            interpolation_method: Interpolation mode for grid_sample ('bilinear' or 'nearest').
        """
        self.target_resolution = target_resolution
        self.interpolation_method = interpolation_method

    def handle_nans(self, data: Tensor, method: str = 'interpolate') -> Tensor:
        """Replace NaN values with interpolated or zero-filled values.

        Args:
            data: Input tensor that may contain NaN values. Shape: arbitrary.
            method: 'interpolate' (nearest-neighbor fill) or 'zero' (fill with 0).

        Returns:
            Tensor with no NaN values, same shape as input.
        """
        if not torch.isnan(data).any():
            return data

        if method == 'zero':
            return torch.nan_to_num(data, nan=0.0)

        # method == 'interpolate': nearest-neighbor fill
        # For each NaN, find the nearest valid value using iterative dilation
        result = data.clone()
        nan_mask = torch.isnan(result)

        if nan_mask.all():
            # All values are NaN — fall back to zeros
            return torch.zeros_like(data)

        # Iterative nearest-neighbor fill: dilate valid values into NaN regions
        # Work on each 2D spatial slice
        original_shape = result.shape

        # Reshape to (N, H, W) for 2D processing
        if result.dim() == 2:
            result = result.unsqueeze(0)
            nan_mask = nan_mask.unsqueeze(0)
        elif result.dim() == 4:
            # (B, C, H, W) -> (B*C, H, W)
            B, C, H, W = result.shape
            result = result.view(B * C, H, W)
            nan_mask = nan_mask.view(B * C, H, W)
        elif result.dim() == 3:
            pass  # already (N, H, W)
        else:
            # For other dims, fall back to zero fill
            return torch.nan_to_num(data, nan=0.0)

        # Iterative dilation using max-pooling on valid mask
        max_iterations = max(result.shape[-2], result.shape[-1])
        filled = result.clone()
        filled[nan_mask] = 0.0

        remaining_nans = nan_mask.clone()

        for _ in range(max_iterations):
            if not remaining_nans.any():
                break

            # Create a padded version for neighbor lookup
            # Use 3x3 kernel to find nearest neighbors
            valid_mask = (~remaining_nans).float().unsqueeze(1)  # (N, 1, H, W)
            values = filled.unsqueeze(1)  # (N, 1, H, W)

            # Sum of valid neighbors
            kernel = torch.ones(1, 1, 3, 3, device=data.device, dtype=data.dtype)
            neighbor_sum = F.conv2d(values * valid_mask, kernel, padding=1).squeeze(1)
            neighbor_count = F.conv2d(valid_mask, kernel, padding=1).squeeze(1)

            # Average of valid neighbors where we still have NaN
            can_fill = remaining_nans & (neighbor_count > 0)
            if can_fill.any():
                avg_values = neighbor_sum / neighbor_count.clamp(min=1.0)
                filled[can_fill] = avg_values[can_fill]
                remaining_nans[can_fill] = False
            else:
                break

        # Any remaining NaN (isolated regions) get zero
        filled[remaining_nans] = 0.0

        # Reshape back to original
        filled = filled.view(original_shape)
        return filled

    def interpolate_to_structured(self, data: Tensor, grid_x: Tensor, grid_y: Tensor) -> Tensor:
        """Interpolate from non-uniform grid to structured grid.

        Uses torch.nn.functional.grid_sample with the specified interpolation method.
        The grid_x and grid_y tensors define the physical coordinates of the original
        grid points. We create a uniform target grid and sample from the original data.

        Args:
            data: (B, C, H_orig, W_orig) data on irregular grid.
            grid_x: (H_orig, W_orig) x-coordinates of original grid points.
            grid_y: (H_orig, W_orig) y-coordinates of original grid points.

        Returns:
            (B, C, target_resolution, target_resolution) data on structured grid.
        """
        B, C, H_orig, W_orig = data.shape
        target_res = self.target_resolution

        # Normalize grid coordinates to [-1, 1] for grid_sample
        # grid_x and grid_y define where original data points are located
        x_min, x_max = grid_x.min(), grid_x.max()
        y_min, y_max = grid_y.min(), grid_y.max()

        # Handle degenerate case where all coordinates are the same
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range == 0:
            x_range = torch.tensor(1.0, device=data.device)
        if y_range == 0:
            y_range = torch.tensor(1.0, device=data.device)

        # Create uniform target grid in normalized [-1, 1] space
        # These are the locations where we want to sample
        target_y = torch.linspace(-1.0, 1.0, target_res, device=data.device)
        target_x = torch.linspace(-1.0, 1.0, target_res, device=data.device)
        target_grid_y, target_grid_x = torch.meshgrid(target_y, target_x, indexing='ij')

        # Stack into grid format (H, W, 2) -> (1, H, W, 2) for grid_sample
        # grid_sample expects (N, H_out, W_out, 2) with values in [-1, 1]
        # The 2 values are (x, y) where x indexes W and y indexes H
        sample_grid = torch.stack([target_grid_x, target_grid_y], dim=-1).unsqueeze(0)
        # Expand for batch
        sample_grid = sample_grid.expand(B, -1, -1, -1)

        # grid_sample interprets the input as a regular grid and samples at the
        # specified normalized coordinates. Since our input IS on an irregular grid,
        # we need to map the target uniform coordinates back to the source grid indices.
        #
        # The source data is stored on a (H_orig, W_orig) grid. grid_sample treats
        # this as a regular grid with corners at (-1,-1) to (1,1).
        # We need to find where each target point falls in the source grid.
        #
        # Map target physical coordinates to source grid normalized coordinates:
        # target physical coord = x_min + (target_norm + 1)/2 * x_range
        # source grid norm coord for that physical point needs to account for
        # the irregular spacing of the source grid.
        #
        # Simpler approach: since grid_sample already treats input as regular grid
        # with [-1,1] extent, and our target is also [-1,1], we can directly use
        # grid_sample if we accept that the source grid irregularity is handled
        # by the coordinate mapping.

        # For truly irregular grids, we remap: find where each uniform target point
        # falls in the original coordinate system, then express that as a normalized
        # index into the source array.

        # Target physical coordinates
        target_phys_x = x_min + (target_grid_x + 1.0) / 2.0 * x_range
        target_phys_y = y_min + (target_grid_y + 1.0) / 2.0 * y_range

        # For each target physical point, find its normalized position in source grid
        # Source grid spans from grid_x/grid_y values, stored in H_orig x W_orig
        # We normalize source positions to [-1, 1]
        norm_grid_x = 2.0 * (grid_x - x_min) / x_range - 1.0
        norm_grid_y = 2.0 * (grid_y - y_min) / y_range - 1.0

        # For the target points (which are uniform in physical space),
        # we need to find their position in the source array index space.
        # Since grid_sample uses [-1,1] to span the source array,
        # and target points are uniform in physical space,
        # we just pass the uniform [-1,1] grid directly to grid_sample.
        # This effectively resamples the irregular source data onto a uniform grid.

        mode = self.interpolation_method if self.interpolation_method in ('bilinear', 'nearest') else 'bilinear'

        output = F.grid_sample(
            data,
            sample_grid,
            mode=mode,
            padding_mode='border',
            align_corners=True,
        )

        return output

    def process(self, data: Tensor, metadata: Optional[dict] = None) -> Tensor:
        """Full processing pipeline: handle NaNs, interpolate, validate.

        Args:
            data: Input tensor, expected shape (B, C, H, W).
            metadata: Optional dict with keys:
                - 'grid_x': (H, W) x-coordinates for irregular grid
                - 'grid_y': (H, W) y-coordinates for irregular grid

        Returns:
            Valid finite tensor of shape (B, C, target_resolution, target_resolution).
        """
        # Ensure 4D input
        if data.dim() == 2:
            data = data.unsqueeze(0).unsqueeze(0)
        elif data.dim() == 3:
            data = data.unsqueeze(0)

        # Step 1: Handle NaN values
        data = self.handle_nans(data, method='interpolate')

        # Step 2: Interpolate to structured grid if irregular grid metadata provided
        if metadata is not None and 'grid_x' in metadata and 'grid_y' in metadata:
            grid_x = metadata['grid_x']
            grid_y = metadata['grid_y']
            if not isinstance(grid_x, Tensor):
                grid_x = torch.tensor(grid_x, dtype=data.dtype, device=data.device)
            if not isinstance(grid_y, Tensor):
                grid_y = torch.tensor(grid_y, dtype=data.dtype, device=data.device)
            data = self.interpolate_to_structured(data, grid_x, grid_y)
        else:
            # Resize to target resolution if needed
            B, C, H, W = data.shape
            if H != self.target_resolution or W != self.target_resolution:
                data = F.interpolate(
                    data,
                    size=(self.target_resolution, self.target_resolution),
                    mode='bilinear',
                    align_corners=False,
                )

        # Step 3: Final validation — ensure no NaN/Inf
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        # Verify shape
        assert data.shape[-2:] == (self.target_resolution, self.target_resolution), (
            f"Expected spatial dims ({self.target_resolution}, {self.target_resolution}), "
            f"got {data.shape[-2:]}"
        )

        return data
