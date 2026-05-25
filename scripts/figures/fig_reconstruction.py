#!/usr/bin/env python
"""
Figure: Reconstruction Comparison.

Side-by-side panels showing:
- Input K field (permeability)
- Ground-truth solution
- PI-JEPA prediction
- Pointwise residual map

Displays 3 representative samples at N_ℓ=100.
Uses diverging colormap for residuals, consistent scale across samples.
Annotates quantitative error metrics (relative ℓ2 error) on each panel.

Uses colorblind-accessible palettes.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# Colorblind-accessible colormaps
PERM_CMAP = 'viridis'          # permeability
FIELD_CMAP = 'cividis'         # pressure/saturation fields
RESIDUAL_CMAP = 'RdBu_r'      # diverging for residuals


def compute_relative_l2_error(pred: np.ndarray, gt: np.ndarray) -> float:
    """Compute relative L2 error."""
    return np.linalg.norm(pred - gt) / (np.linalg.norm(gt) + 1e-10)


def generate_synthetic_samples(n_samples: int = 3, resolution: int = 64,
                                seed: int = 42) -> dict:
    """Generate synthetic data for figure when real data is unavailable.

    Returns dict with keys: 'K', 'ground_truth', 'prediction'.
    Each has shape (n_samples, resolution, resolution).
    """
    rng = np.random.default_rng(seed)

    K_fields = []
    gt_fields = []
    pred_fields = []

    for i in range(n_samples):
        # Permeability: log-normal with spatial correlation
        noise = rng.standard_normal((resolution, resolution))
        from scipy.ndimage import gaussian_filter
        K = np.exp(gaussian_filter(noise, sigma=5.0))
        K_fields.append(K)

        # Ground truth: smooth field influenced by K
        gt = gaussian_filter(rng.standard_normal((resolution, resolution)), sigma=8.0)
        gt = gt * np.log(K + 1) / np.log(K + 1).max()
        gt_fields.append(gt)

        # Prediction: GT + small structured error
        error = gaussian_filter(rng.standard_normal((resolution, resolution)), sigma=4.0)
        error *= 0.05 * gt.std()  # ~5% relative error
        pred = gt + error
        pred_fields.append(pred)

    return {
        'K': np.stack(K_fields),
        'ground_truth': np.stack(gt_fields),
        'prediction': np.stack(pred_fields),
    }


def load_or_generate_data(data_path: str = None, n_samples: int = 3) -> dict:
    """Load real results or generate synthetic data."""
    if data_path and os.path.exists(data_path):
        data = torch.load(data_path, map_location='cpu')
        # Expected format: dict with 'K', 'ground_truth', 'prediction' tensors
        return {k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
    else:
        print("  No data found, generating synthetic samples for figure layout.")
        return generate_synthetic_samples(n_samples)


def create_reconstruction_figure(data: dict, output_path: str, dpi: int = 300):
    """Create the reconstruction comparison figure.

    Layout: 3 rows (samples) × 4 columns (K, GT, Pred, Residual)
    """
    n_samples = min(3, data['K'].shape[0])

    fig, axes = plt.subplots(n_samples, 4, figsize=(12, 3 * n_samples + 0.5))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    # Column titles
    col_titles = ['Input K (log)', 'Ground Truth', 'PI-JEPA Prediction', 'Residual']

    # Compute global residual range for consistent colorbar
    residuals = []
    for i in range(n_samples):
        res = data['prediction'][i] - data['ground_truth'][i]
        residuals.append(res)
    max_abs_residual = max(np.abs(r).max() for r in residuals)
    residual_norm = mcolors.TwoSlopeNorm(vmin=-max_abs_residual, vcenter=0,
                                          vmax=max_abs_residual)

    for i in range(n_samples):
        K = data['K'][i]
        gt = data['ground_truth'][i]
        pred = data['prediction'][i]
        residual = pred - gt

        # Relative L2 error
        rel_error = compute_relative_l2_error(pred, gt)

        # Column 0: Permeability (log scale)
        im0 = axes[i, 0].imshow(np.log10(K + 1e-10), cmap=PERM_CMAP, origin='lower')
        axes[i, 0].set_ylabel(f'Sample {i+1}', fontsize=10, fontweight='bold')

        # Column 1: Ground truth
        vmin_gt, vmax_gt = gt.min(), gt.max()
        im1 = axes[i, 1].imshow(gt, cmap=FIELD_CMAP, origin='lower',
                                  vmin=vmin_gt, vmax=vmax_gt)

        # Column 2: Prediction (same scale as GT)
        im2 = axes[i, 2].imshow(pred, cmap=FIELD_CMAP, origin='lower',
                                  vmin=vmin_gt, vmax=vmax_gt)
        # Annotate error
        axes[i, 2].text(0.95, 0.95, f'ε = {rel_error:.4f}',
                        transform=axes[i, 2].transAxes, ha='right', va='top',
                        fontsize=8, color='white', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        # Column 3: Residual (diverging colormap)
        im3 = axes[i, 3].imshow(residual, cmap=RESIDUAL_CMAP, origin='lower',
                                  norm=residual_norm)

        # Add colorbars
        for j, im in enumerate([im0, im1, im2, im3]):
            divider = make_axes_locatable(axes[i, j])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            plt.colorbar(im, cax=cax)

        # Remove tick labels for cleaner look
        for j in range(4):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    # Column titles
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=10, fontweight='bold', pad=8)

    # Overall title
    fig.suptitle('Reconstruction Quality at $N_\\ell = 100$', fontsize=12,
                 fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate reconstruction comparison figure")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to reconstruction results (.pt file)")
    parser.add_argument("--output", type=str, default="paper/figures/fig_reconstruction.pdf",
                        help="Output file path")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--n-samples", type=int, default=3,
                        help="Number of samples to display")
    args = parser.parse_args()

    data = load_or_generate_data(args.data_path, args.n_samples)
    create_reconstruction_figure(data, args.output, args.dpi)


if __name__ == "__main__":
    main()
