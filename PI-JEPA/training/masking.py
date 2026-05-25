"""
Spatial Block Masking for Self-Supervised Pretraining

Implements spatial block masking strategy for JEPA-style self-supervised learning
on coefficient fields.

Validates: Requirement 3 (Spatial Block Masking Strategy)
- AC 3.1: Partition coefficient field into non-overlapping patches of size P×P (default 8×8)
- AC 3.2: Select context patches as contiguous spatial block covering ~65% of domain
- AC 3.3: Select target patches as complement of context patches
- AC 3.4: Augment mask tokens with 2D sinusoidal positional encodings
- AC 3.5: Sample new random mask configurations for all training batches
"""

import math
from typing import Tuple

import torch
import torch.nn as nn


class SpatialBlockMasker:
    """Spatial block masking for self-supervised pretraining.
    
    This class implements the masking strategy described in the JEPA paper,
    where context patches form a contiguous spatial block and target patches
    are the complement.
    
    The masking works by:
    1. Dividing the image into a grid of patches (e.g., 8×8 grid for 64×64 image with 8×8 patches)
    2. Sampling a random rectangular block to be the TARGET region
    3. The CONTEXT region is everything outside the target block
    4. Target patches are predicted from context patches in latent space
    
    Attributes:
        grid_size: Number of patches per side (e.g., 8 for 64×64 image with 8×8 patches)
        context_ratio: Fraction of patches to use as context (default 0.65)
        min_block_size: Minimum target block size in patches
        max_block_size: Maximum target block size in patches
    """
    
    def __init__(
        self,
        grid_size: int = 8,
        context_ratio: float = 0.65,
        min_block_size: int = 2,
        max_block_size: int = 4
    ):
        """
        Initialize spatial block masker.
        
        Args:
            grid_size: Number of patches per side (64/8 = 8 for patch_size=8)
            context_ratio: Fraction of patches to use as context (default 0.65)
            min_block_size: Minimum target block size in patches
            max_block_size: Maximum target block size in patches
            
        Raises:
            ValueError: If parameters are invalid
        """
        if grid_size < 1:
            raise ValueError(f"grid_size must be >= 1, got {grid_size}")
        if not 0 < context_ratio < 1:
            raise ValueError(f"context_ratio must be in (0, 1), got {context_ratio}")
        if min_block_size < 1:
            raise ValueError(f"min_block_size must be >= 1, got {min_block_size}")
        if max_block_size < min_block_size:
            raise ValueError(
                f"max_block_size ({max_block_size}) must be >= min_block_size ({min_block_size})"
            )
        if max_block_size > grid_size:
            raise ValueError(
                f"max_block_size ({max_block_size}) must be <= grid_size ({grid_size})"
            )
        
        self.grid_size = grid_size
        self.context_ratio = context_ratio
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.total_patches = grid_size * grid_size
        
        # Calculate target block dimensions to achieve desired context ratio
        # Target ratio = 1 - context_ratio ≈ 0.35
        # For 8×8 grid with 0.65 context ratio: target ≈ 22 patches
        # This requires blocks of roughly 4-5 patches per side
        target_ratio = 1.0 - context_ratio
        target_patches = self.total_patches * target_ratio
        
        # Calculate ideal block dimension to achieve target ratio
        # For a square block: dim^2 = target_patches
        ideal_dim = math.sqrt(target_patches)
        
        # Compute effective min/max that respect both user constraints and target ratio
        # We'll sample dimensions that average to the ideal
        self._ideal_block_dim = max(min_block_size, min(max_block_size, ideal_dim))
    
    def sample_mask(
        self,
        batch_size: int,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample context and target indices for a batch.
        
        For each sample in the batch, samples a random rectangular block
        as the target region. The context region is the complement.
        
        Args:
            batch_size: Number of samples in batch
            device: Device for tensors
            
        Returns:
            context_idx: (B, N_c) indices of context patches (flattened grid indices)
            target_idx: (B, N_t) indices of target patches (flattened grid indices)
            
        Note:
            Indices are in row-major order for the patch grid.
            For an 8×8 grid, index 0 is top-left, index 63 is bottom-right.
        """
        context_indices_list = []
        target_indices_list = []
        
        # Calculate target block dimensions to achieve desired context ratio
        # Target patches = total_patches * (1 - context_ratio)
        target_ratio = 1.0 - self.context_ratio
        target_patches_desired = self.total_patches * target_ratio
        
        # For a square block: we want h * w ≈ target_patches_desired
        # We'll sample dimensions that center around sqrt(target_patches_desired)
        ideal_dim = math.sqrt(target_patches_desired)
        
        # Clamp to user-specified bounds
        effective_min = max(self.min_block_size, int(ideal_dim * 0.7))
        effective_max = min(self.max_block_size, int(ideal_dim * 1.3) + 1, self.grid_size)
        
        # Ensure valid range
        effective_min = min(effective_min, effective_max)
        
        for _ in range(batch_size):
            # Sample random block dimensions within effective range
            block_h = torch.randint(
                effective_min, 
                effective_max + 1, 
                (1,)
            ).item()
            block_w = torch.randint(
                effective_min, 
                effective_max + 1, 
                (1,)
            ).item()
            
            # Sample random top-left corner for the target block
            max_row = self.grid_size - block_h
            max_col = self.grid_size - block_w
            
            top = torch.randint(0, max_row + 1, (1,)).item()
            left = torch.randint(0, max_col + 1, (1,)).item()
            
            # Create mask grid (True = target, False = context)
            mask_grid = torch.zeros(self.grid_size, self.grid_size, dtype=torch.bool)
            mask_grid[top:top + block_h, left:left + block_w] = True
            
            # Flatten and get indices
            mask_flat = mask_grid.flatten()
            all_indices = torch.arange(self.total_patches)
            
            target_idx = all_indices[mask_flat]
            context_idx = all_indices[~mask_flat]
            
            target_indices_list.append(target_idx)
            context_indices_list.append(context_idx)
        
        # Pad to same length (different samples may have different block sizes)
        max_target_len = max(t.shape[0] for t in target_indices_list)
        max_context_len = max(c.shape[0] for c in context_indices_list)
        
        # Pad with -1 (will be masked out during attention)
        target_padded = torch.full((batch_size, max_target_len), -1, dtype=torch.long)
        context_padded = torch.full((batch_size, max_context_len), -1, dtype=torch.long)
        
        for i, (t_idx, c_idx) in enumerate(zip(target_indices_list, context_indices_list)):
            target_padded[i, :t_idx.shape[0]] = t_idx
            context_padded[i, :c_idx.shape[0]] = c_idx
        
        return context_padded.to(device), target_padded.to(device)
    
    def get_positional_encoding(
        self,
        target_idx: torch.Tensor,
        embed_dim: int
    ) -> torch.Tensor:
        """
        Generate 2D sinusoidal positional encodings for target patches.
        
        Uses separate sinusoidal encodings for row and column positions,
        concatenated to form the full positional encoding.
        
        Args:
            target_idx: (B, N_t) target patch indices (flattened grid indices)
            embed_dim: Embedding dimension (must be divisible by 4)
            
        Returns:
            pos_enc: (B, N_t, embed_dim) positional encodings
            
        Note:
            For padded indices (-1), returns zero encodings.
        """
        if embed_dim % 4 != 0:
            raise ValueError(f"embed_dim must be divisible by 4, got {embed_dim}")
        
        batch_size, n_targets = target_idx.shape
        device = target_idx.device
        
        # Convert flat indices to 2D coordinates
        # Handle padding (-1 indices) by clamping to 0, will be zeroed out later
        valid_mask = target_idx >= 0
        safe_idx = target_idx.clamp(min=0)
        
        row_idx = safe_idx // self.grid_size  # (B, N_t)
        col_idx = safe_idx % self.grid_size   # (B, N_t)
        
        # Normalize to [0, 1]
        row_norm = row_idx.float() / (self.grid_size - 1) if self.grid_size > 1 else row_idx.float()
        col_norm = col_idx.float() / (self.grid_size - 1) if self.grid_size > 1 else col_idx.float()
        
        # Generate sinusoidal encodings
        pos_enc = self._sinusoidal_encoding_2d(row_norm, col_norm, embed_dim, device)
        
        # Zero out padded positions
        pos_enc = pos_enc * valid_mask.unsqueeze(-1).float()
        
        return pos_enc
    
    def _sinusoidal_encoding_2d(
        self,
        row_pos: torch.Tensor,
        col_pos: torch.Tensor,
        embed_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate 2D sinusoidal positional encoding.
        
        The encoding is split into 4 parts:
        - sin(row), cos(row), sin(col), cos(col)
        Each part gets embed_dim/4 dimensions with different frequencies.
        
        Args:
            row_pos: (B, N) normalized row positions in [0, 1]
            col_pos: (B, N) normalized column positions in [0, 1]
            embed_dim: Total embedding dimension
            device: Device for tensors
            
        Returns:
            encoding: (B, N, embed_dim) positional encoding
        """
        dim_per_component = embed_dim // 4
        
        # Frequency bands (geometric progression)
        freq_bands = torch.arange(dim_per_component, device=device, dtype=torch.float32)
        freq_bands = 10000 ** (-freq_bands / dim_per_component)  # (D/4,)
        
        # Scale positions to radians
        # row_pos, col_pos: (B, N) -> (B, N, 1)
        row_pos = row_pos.unsqueeze(-1) * math.pi  # Scale to [0, pi]
        col_pos = col_pos.unsqueeze(-1) * math.pi
        
        # Apply frequencies: (B, N, 1) * (D/4,) -> (B, N, D/4)
        row_enc = row_pos * freq_bands
        col_enc = col_pos * freq_bands
        
        # Concatenate sin and cos for both dimensions
        encoding = torch.cat([
            torch.sin(row_enc),
            torch.cos(row_enc),
            torch.sin(col_enc),
            torch.cos(col_enc)
        ], dim=-1)  # (B, N, embed_dim)
        
        return encoding
    
    def get_context_ratio_actual(self, context_idx: torch.Tensor) -> float:
        """
        Calculate the actual context ratio for a batch.
        
        Args:
            context_idx: (B, N_c) context patch indices
            
        Returns:
            Average context ratio across the batch
        """
        # Count valid (non-padded) context patches
        valid_counts = (context_idx >= 0).sum(dim=1).float()
        return (valid_counts / self.total_patches).mean().item()


def build_spatial_block_masker(config: dict) -> SpatialBlockMasker:
    """
    Build SpatialBlockMasker from configuration.
    
    Args:
        config: Configuration dictionary with masking settings
        
    Returns:
        Configured SpatialBlockMasker instance
    """
    masking_cfg = config.get("pretraining", {}).get("masking", {})
    model_cfg = config.get("model", {}).get("encoder", {})

    image_size = model_cfg.get("image_size", 64)
    patch_size = model_cfg.get("patch_size", 8)

    encoder_type = model_cfg.get("type", "vit").lower()
    if encoder_type in ("fourier_3d", "fourier3d"):
        # Support either scalar (cubic) or 3-list (rectangular) image/patch.
        def _triple(v):
            if isinstance(v, int):
                return (v, v, v)
            return tuple(int(x) for x in v)

        ig = _triple(image_size)
        pg = _triple(patch_size)
        grid_dhw = (ig[0] // pg[0], ig[1] // pg[1], ig[2] // pg[2])
        min_axis = min(grid_dhw)
        return SpatialBlockMasker3D(
            grid_size=grid_dhw,
            context_ratio=masking_cfg.get("context_ratio", 0.65),
            min_block_size=masking_cfg.get("min_block_size", 1),
            max_block_size=masking_cfg.get("max_block_size", max(2, min_axis // 2)),
        )

    # 2D path: image_size and patch_size are scalars
    grid_size_2d = image_size // patch_size if isinstance(image_size, int) else (image_size[0] // patch_size[0])
    return SpatialBlockMasker(
        grid_size=grid_size_2d,
        context_ratio=masking_cfg.get("context_ratio", 0.65),
        min_block_size=masking_cfg.get("min_block_size", 2),
        max_block_size=masking_cfg.get("max_block_size", 4)
    )


# =============================================================================
# 3D spatial block masker — operates on a cubic patch grid (gd, gh, gw).
# Patch indices are flattened in (d, h, w) row-major order, matching
# FourierJEPAEncoder3D's flatten(2).transpose(1, 2) convention.
# =============================================================================


class SpatialBlockMasker3D:
    """Spatial block masking over a 3D (cubic) patch grid.

    Samples a random axis-aligned cuboid as the target region; the context
    region is the complement. Returns padded (B, N_t) and (B, N_c) tensors
    in the same format as SpatialBlockMasker so downstream code is unchanged.
    """

    def __init__(
        self,
        grid_size,
        context_ratio: float = 0.65,
        min_block_size: int = 1,
        max_block_size: int = 4,
    ):
        # grid_size may be a scalar (cubic) or a 3-tuple (rectangular: gd, gh, gw).
        if isinstance(grid_size, int):
            self.grid_size_dhw = (grid_size, grid_size, grid_size)
        elif hasattr(grid_size, "__len__") and len(grid_size) == 3:
            self.grid_size_dhw = tuple(int(g) for g in grid_size)
        else:
            raise ValueError(
                f"grid_size must be int or 3-tuple, got {grid_size!r}"
            )
        gd, gh, gw = self.grid_size_dhw
        if min(gd, gh, gw) < 1:
            raise ValueError(f"all grid_size axes must be >= 1, got {self.grid_size_dhw}")
        if not 0 < context_ratio < 1:
            raise ValueError(f"context_ratio must be in (0, 1), got {context_ratio}")
        if min_block_size < 1:
            raise ValueError(f"min_block_size must be >= 1, got {min_block_size}")
        if max_block_size < min_block_size:
            raise ValueError(
                f"max_block_size ({max_block_size}) must be >= min_block_size ({min_block_size})"
            )
        # max_block_size is checked against the SMALLEST axis to guarantee
        # the cuboid fits in every axis.
        if max_block_size > min(gd, gh, gw):
            raise ValueError(
                f"max_block_size ({max_block_size}) must be <= min(grid axes) "
                f"= {min(gd, gh, gw)}; grid axes={self.grid_size_dhw}"
            )

        # Legacy scalar alias used by some downstream code (largest axis).
        self.grid_size = max(self.grid_size_dhw)
        self.context_ratio = context_ratio
        self.min_block_size = min_block_size
        self.max_block_size = max_block_size
        self.total_patches = gd * gh * gw

        target_ratio = 1.0 - context_ratio
        target_patches_desired = self.total_patches * target_ratio
        ideal_dim = target_patches_desired ** (1.0 / 3.0)
        self._ideal_block_dim = max(min_block_size, min(max_block_size, ideal_dim))

    def sample_mask(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gd, gh, gw = self.grid_size_dhw
        target_ratio = 1.0 - self.context_ratio
        target_patches_desired = self.total_patches * target_ratio
        ideal_dim = target_patches_desired ** (1.0 / 3.0)

        # Compute per-axis effective bounds so we don't draw a block bigger
        # than the smallest axis on that axis.
        def axis_bounds(g_axis):
            emin = max(self.min_block_size, int(ideal_dim * 0.7))
            emax = min(self.max_block_size, int(ideal_dim * 1.3) + 1, g_axis)
            emin = min(emin, emax)
            return emin, emax

        emin_d, emax_d = axis_bounds(gd)
        emin_h, emax_h = axis_bounds(gh)
        emin_w, emax_w = axis_bounds(gw)

        context_indices_list = []
        target_indices_list = []

        for _ in range(batch_size):
            bd = torch.randint(emin_d, emax_d + 1, (1,)).item()
            bh = torch.randint(emin_h, emax_h + 1, (1,)).item()
            bw = torch.randint(emin_w, emax_w + 1, (1,)).item()

            d0 = torch.randint(0, gd - bd + 1, (1,)).item()
            h0 = torch.randint(0, gh - bh + 1, (1,)).item()
            w0 = torch.randint(0, gw - bw + 1, (1,)).item()

            mask_grid = torch.zeros(gd, gh, gw, dtype=torch.bool)
            mask_grid[d0:d0 + bd, h0:h0 + bh, w0:w0 + bw] = True

            mask_flat = mask_grid.flatten()
            all_indices = torch.arange(self.total_patches)

            target_idx = all_indices[mask_flat]
            context_idx = all_indices[~mask_flat]

            target_indices_list.append(target_idx)
            context_indices_list.append(context_idx)

        max_target_len = max(t.shape[0] for t in target_indices_list)
        max_context_len = max(c.shape[0] for c in context_indices_list)

        target_padded = torch.full((batch_size, max_target_len), -1, dtype=torch.long)
        context_padded = torch.full((batch_size, max_context_len), -1, dtype=torch.long)

        for i, (t_idx, c_idx) in enumerate(zip(target_indices_list, context_indices_list)):
            target_padded[i, :t_idx.shape[0]] = t_idx
            context_padded[i, :c_idx.shape[0]] = c_idx

        return context_padded.to(device), target_padded.to(device)

    def get_positional_encoding(
        self,
        target_idx: torch.Tensor,
        embed_dim: int,
    ) -> torch.Tensor:
        """Sinusoidal positional encodings for 3D patch indices.

        Splits embed_dim into 6 chunks: sin/cos for each of d, h, w.
        """
        if embed_dim % 6 != 0:
            raise ValueError(f"embed_dim must be divisible by 6 for 3D pos enc, got {embed_dim}")

        device = target_idx.device
        valid_mask = target_idx >= 0
        safe_idx = target_idx.clamp(min=0)

        # Decompose flat index into (d, h, w) using the actual grid axes.
        gd, gh, gw = self.grid_size_dhw
        d_idx = safe_idx // (gh * gw)
        rem = safe_idx % (gh * gw)
        h_idx = rem // gw
        w_idx = rem % gw

        d_norm = d_idx.float() / max(gd - 1, 1)
        h_norm = h_idx.float() / max(gh - 1, 1)
        w_norm = w_idx.float() / max(gw - 1, 1)

        dim_per = embed_dim // 6
        freq_bands = torch.arange(dim_per, device=device, dtype=torch.float32)
        freq_bands = 10000 ** (-freq_bands / dim_per)

        d_pos = d_norm.unsqueeze(-1) * math.pi * freq_bands
        h_pos = h_norm.unsqueeze(-1) * math.pi * freq_bands
        w_pos = w_norm.unsqueeze(-1) * math.pi * freq_bands

        enc = torch.cat([
            torch.sin(d_pos), torch.cos(d_pos),
            torch.sin(h_pos), torch.cos(h_pos),
            torch.sin(w_pos), torch.cos(w_pos),
        ], dim=-1)

        enc = enc * valid_mask.unsqueeze(-1).float()
        return enc

    def get_context_ratio_actual(self, context_idx: torch.Tensor) -> float:
        valid_counts = (context_idx >= 0).sum(dim=1).float()
        return (valid_counts / self.total_patches).mean().item()
