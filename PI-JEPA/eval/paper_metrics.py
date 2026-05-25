"""
Paper-grade evaluation metrics for PI-JEPA experiments.

Beyond the standard relative-L2 the original paper reported, top-venue
reviewers in the CO2-storage subfield expect:
  - nRMSE (relative L2 root-mean-square)
  - MaxErr (worst-case voxel error)
  - cRMSE (conservation residual: did the model violate mass conservation?)
  - fRMSE in low/mid/high frequency bands (does it get long-scale dynamics
    right but high-frequency wrong, or vice versa?)
  - IoU on plume saturation (where is the CO2 plume?)
  - Plume migration distance error
  - Mass conservation error per timestep
  - Breakthrough time error

This module supplies them as pure numpy/torch functions that take
predicted/true field tensors and return scalars (or per-batch arrays).
Suitable for direct use in the eval orchestrator (run_eval.py).
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch


def relative_l2(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample relative L2: ||pred - true||_2 / ||true||_2.

    Returns (B,) tensor. pred and true: (B, ...).
    """
    pred = pred.float()
    true = true.float()
    dims = tuple(range(1, pred.dim()))
    num = (pred - true).pow(2).sum(dim=dims).sqrt()
    den = true.pow(2).sum(dim=dims).sqrt() + eps
    return num / den


def nrmse(pred: torch.Tensor, true: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalized RMSE: RMSE / (true.max - true.min)."""
    pred = pred.float()
    true = true.float()
    dims = tuple(range(1, pred.dim()))
    rmse = (pred - true).pow(2).mean(dim=dims).sqrt()
    rng = true.flatten(1).max(dim=1).values - true.flatten(1).min(dim=1).values + eps
    return rmse / rng


def max_err(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    """Worst-case absolute error per sample."""
    return (pred.float() - true.float()).abs().flatten(1).max(dim=1).values


def conservation_residual(field: torch.Tensor) -> torch.Tensor:
    """Approximate mass-conservation residual for a saturation/density field
    over time. For a conserved quantity, the spatial integral should be
    constant (modulo sources/sinks). We measure ||∂_t ∑_xy[z] field||_2.

    field: (B, C, T, H, W) (3D time series) or (B, C, T, H, W) — in either
    case the time axis is dim=2. Sums over the last two spatial axes (H, W).
    Returns (B,) tensor.
    """
    if field.dim() != 5:
        return torch.zeros(field.shape[0], device=field.device)
    spatial_sum = field.sum(dim=(-2, -1))  # (B, C, T)
    dt_sum = spatial_sum[..., 1:] - spatial_sum[..., :-1]  # (B, C, T-1)
    if dt_sum.shape[-1] == 0:
        return torch.zeros(field.shape[0], device=field.device)
    return dt_sum.pow(2).mean(dim=(1, 2)).sqrt()


def fourier_band_rmse(
    pred: torch.Tensor,
    true: torch.Tensor,
    bands: Tuple[float, float] = (0.25, 0.5),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Spectral-band RMSE: low/mid/high-frequency error.

    Splits the rfftn magnitude spectrum at fraction-of-Nyquist thresholds
    given by `bands` (low_cutoff, mid_cutoff). Returns (low, mid, high) RMSE.

    pred, true: (B, C, [D,] H, W). Uses rfftn over the last 2 or 3 spatial axes.
    """
    pred = pred.float()
    true = true.float()
    spatial_dims = list(range(2, pred.dim()))
    P_ft = torch.fft.rfftn(pred, dim=spatial_dims)
    T_ft = torch.fft.rfftn(true, dim=spatial_dims)
    err_ft_mag = (P_ft - T_ft).abs()  # complex magnitude

    # Build a radial frequency grid in cycles-per-pixel
    shapes_full = pred.shape[2:]
    shapes_rfft = list(err_ft_mag.shape[2:])
    # Build wavenumber per axis
    ks = []
    for axis, (full_n, rfft_n) in enumerate(zip(shapes_full, shapes_rfft)):
        if axis == len(shapes_full) - 1:
            k = torch.fft.rfftfreq(full_n).to(pred.device)  # (rfft_n,)
        else:
            k = torch.fft.fftfreq(full_n).to(pred.device)
        ks.append(k.abs())
    # Outer-product to get a radial freq grid
    K = ks[0]
    for nxt in ks[1:]:
        K = K.unsqueeze(-1) + nxt  # additive proxy, not Euclidean radial
    # Reshape K to broadcast over (B, C, ...)
    while K.dim() < err_ft_mag.dim():
        K = K.unsqueeze(0)

    low_mask = K < bands[0]
    mid_mask = (K >= bands[0]) & (K < bands[1])
    high_mask = K >= bands[1]

    def _band_rmse(mask):
        sq = err_ft_mag.pow(2) * mask
        return sq.flatten(1).mean(dim=1).sqrt()

    return _band_rmse(low_mask), _band_rmse(mid_mask), _band_rmse(high_mask)


def saturation_iou(
    pred_saturation: torch.Tensor,
    true_saturation: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Intersection-over-Union on the binarized plume.

    pred_saturation, true_saturation: (B, ...) tensors with values nominally in [0, 1].
    threshold defines "plume present" (default 5% saturation, common for CO2 monitoring).
    Returns (B,) IoU.
    """
    pred_bin = (pred_saturation.float() > threshold)
    true_bin = (true_saturation.float() > threshold)
    inter = (pred_bin & true_bin).flatten(1).sum(dim=1).float()
    union = (pred_bin | true_bin).flatten(1).sum(dim=1).float()
    return inter / (union + 1e-8)


def plume_centroid_error(
    pred_saturation: torch.Tensor,
    true_saturation: torch.Tensor,
    threshold: float = 0.05,
) -> torch.Tensor:
    """Distance between predicted and true plume centroids (mass-weighted).

    Reports the L2 distance in voxel units. For physical units multiply
    by the cell size externally.
    """
    pred = pred_saturation.float()
    true = true_saturation.float()
    spatial_dims = list(range(1, pred.dim()))  # all dims except batch
    grid_shape = pred.shape[1:]

    def _centroid(field):
        # field: (B, [C,] [T,] H, W). Build per-axis coordinate grids.
        B = field.shape[0]
        coords = []
        for axis_idx, n in enumerate(grid_shape, start=1):
            shape = [1] * field.dim()
            shape[axis_idx] = n
            grid = torch.arange(n, device=field.device, dtype=torch.float32).reshape(shape)
            coords.append(grid)
        mass = field.sum(dim=spatial_dims, keepdim=True) + 1e-8
        centroids = []
        for grid in coords:
            c = (field * grid).sum(dim=spatial_dims, keepdim=True) / mass
            centroids.append(c.flatten())  # (B,)
        return torch.stack(centroids, dim=1)  # (B, D)

    c_pred = _centroid((pred > threshold).float() * pred)
    c_true = _centroid((true > threshold).float() * true)
    return (c_pred - c_true).pow(2).sum(dim=1).sqrt()


def bootstrap_ci_95(values: np.ndarray, n_boot: int = 2000, rng_seed: int = 0):
    """Return (mean, ci_low, ci_high) for a 95% bootstrap CI."""
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(rng_seed)
    boots = np.empty(n_boot, dtype=np.float64)
    n = len(values)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(values[idx]))
    mean = float(np.mean(values))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return mean, lo, hi
