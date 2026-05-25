#!/usr/bin/env python
"""
Figure: Spectral Analysis.

Shows the distinct frequency content of pressure (smooth/low-frequency)
versus saturation (sharp fronts/high-frequency) fields.

Panels:
- Left: Example pressure and saturation fields
- Center: Radially-averaged power spectra (log-log)
- Right: Cumulative energy vs wavenumber

Justifies the operator-splitting architecture based on spectral character.
Uses colorblind-accessible palette.
"""

import os
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# Colorblind-accessible colors
COLOR_PRESSURE = '#0072B2'     # blue
COLOR_SATURATION = '#D55E00'   # vermillion
COLOR_PERMEABILITY = '#009E73' # green


def compute_radial_power_spectrum(field: np.ndarray) -> tuple:
    """Compute radially-averaged power spectrum of a 2D field.

    Args:
        field: 2D array (H, W).

    Returns:
        (wavenumbers, power): 1D arrays of radial wavenumber and power.
    """
    H, W = field.shape
    # 2D FFT
    fft2 = np.fft.fft2(field - field.mean())
    power_2d = np.abs(fft2) ** 2

    # Shift zero frequency to center
    power_2d = np.fft.fftshift(power_2d)

    # Radial averaging
    cy, cx = H // 2, W // 2
    y, x = np.ogrid[:H, :W]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cx, cy)
    radial_power = np.zeros(max_r)
    counts = np.zeros(max_r)

    for ri in range(max_r):
        mask = r == ri
        radial_power[ri] = power_2d[mask].mean()
        counts[ri] = mask.sum()

    # Normalize
    radial_power /= (H * W)

    wavenumbers = np.arange(max_r)
    return wavenumbers, radial_power


def compute_cumulative_energy(wavenumbers: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Compute cumulative energy fraction vs wavenumber."""
    cumulative = np.cumsum(power * wavenumbers)  # weight by shell area
    total = cumulative[-1] if cumulative[-1] > 0 else 1.0
    return cumulative / total


def generate_synthetic_fields(resolution: int = 64, seed: int = 42) -> dict:
    """Generate synthetic pressure and saturation fields with distinct spectra."""
    rng = np.random.default_rng(seed)

    # Pressure: smooth, low-frequency dominated
    noise = rng.standard_normal((resolution, resolution))
    pressure = gaussian_filter(noise, sigma=10.0)
    pressure = (pressure - pressure.mean()) / (pressure.std() + 1e-8)

    # Saturation: sharp fronts, high-frequency content
    noise = rng.standard_normal((resolution, resolution))
    smooth_base = gaussian_filter(noise, sigma=5.0)
    # Add sharp front
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)
    X, Y = np.meshgrid(x, y)
    front = 1.0 / (1.0 + np.exp(-30 * (X + 0.3 * Y - 0.5)))
    saturation = 0.3 * smooth_base + 0.7 * front
    saturation = (saturation - saturation.mean()) / (saturation.std() + 1e-8)

    # Permeability: intermediate frequency content
    noise = rng.standard_normal((resolution, resolution))
    permeability = gaussian_filter(noise, sigma=4.0)
    permeability = np.exp(permeability)

    return {
        'pressure': pressure,
        'saturation': saturation,
        'permeability': permeability,
    }


def load_or_generate_data(data_path: str = None) -> dict:
    """Load real fields or generate synthetic data."""
    if data_path and os.path.exists(data_path):
        data = torch.load(data_path, map_location='cpu')
        return {k: v.numpy() if isinstance(v, torch.Tensor) else v
                for k, v in data.items()}
    else:
        print("  No data found, generating synthetic fields for figure layout.")
        return generate_synthetic_fields()


def create_spectral_analysis_figure(data: dict, output_path: str, dpi: int = 300):
    """Create the spectral analysis figure."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    pressure = data['pressure']
    saturation = data['saturation']

    # === Left panel: Example fields ===
    ax = axes[0]

    # Show both fields as subplots within the panel
    # Use inset axes
    ax_p = ax.inset_axes([0.0, 0.52, 0.95, 0.45])
    ax_s = ax.inset_axes([0.0, 0.0, 0.95, 0.45])

    im_p = ax_p.imshow(pressure, cmap='cividis', origin='lower')
    ax_p.set_title('Pressure (smooth)', fontsize=8, color=COLOR_PRESSURE)
    ax_p.set_xticks([])
    ax_p.set_yticks([])

    im_s = ax_s.imshow(saturation, cmap='inferno', origin='lower')
    ax_s.set_title('Saturation (sharp fronts)', fontsize=8, color=COLOR_SATURATION)
    ax_s.set_xticks([])
    ax_s.set_yticks([])

    ax.axis('off')
    ax.set_title('(a) Example Fields', fontsize=11, fontweight='bold')

    # === Center panel: Power spectra ===
    ax = axes[1]

    k_p, psd_p = compute_radial_power_spectrum(pressure)
    k_s, psd_s = compute_radial_power_spectrum(saturation)

    # Skip k=0 for log-log plot
    k_p, psd_p = k_p[1:], psd_p[1:]
    k_s, psd_s = k_s[1:], psd_s[1:]

    ax.loglog(k_p, psd_p, color=COLOR_PRESSURE, linewidth=2, label='Pressure')
    ax.loglog(k_s, psd_s, color=COLOR_SATURATION, linewidth=2, label='Saturation')

    # Reference slopes
    k_ref = np.logspace(0.3, 1.3, 20)
    ax.loglog(k_ref, 0.5 * k_ref**(-3), 'k--', alpha=0.4, linewidth=1,
              label='$k^{-3}$ (smooth)')
    ax.loglog(k_ref, 0.1 * k_ref**(-1), 'k:', alpha=0.4, linewidth=1,
              label='$k^{-1}$ (discontinuous)')

    ax.set_xlabel('Wavenumber $k$', fontsize=10)
    ax.set_ylabel('Power Spectral Density', fontsize=10)
    ax.set_title('(b) Radial Power Spectra', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # === Right panel: Cumulative energy ===
    ax = axes[2]

    k_p_full, psd_p_full = compute_radial_power_spectrum(pressure)
    k_s_full, psd_s_full = compute_radial_power_spectrum(saturation)

    cum_p = compute_cumulative_energy(k_p_full, psd_p_full)
    cum_s = compute_cumulative_energy(k_s_full, psd_s_full)

    ax.plot(k_p_full / k_p_full.max(), cum_p, color=COLOR_PRESSURE,
            linewidth=2, label='Pressure')
    ax.plot(k_s_full / k_s_full.max(), cum_s, color=COLOR_SATURATION,
            linewidth=2, label='Saturation')

    # Mark 90% energy threshold
    ax.axhline(y=0.9, color='gray', linestyle='--', alpha=0.5)
    ax.text(0.7, 0.92, '90% energy', fontsize=8, color='gray')

    # Find 90% energy wavenumber for each
    k90_p = k_p_full[np.searchsorted(cum_p, 0.9)] / k_p_full.max()
    k90_s = k_s_full[np.searchsorted(cum_s, 0.9)] / k_s_full.max()
    ax.axvline(x=k90_p, color=COLOR_PRESSURE, linestyle=':', alpha=0.5)
    ax.axvline(x=k90_s, color=COLOR_SATURATION, linestyle=':', alpha=0.5)

    ax.set_xlabel('Normalized Wavenumber $k/k_{max}$', fontsize=10)
    ax.set_ylabel('Cumulative Energy Fraction', fontsize=10)
    ax.set_title('(c) Energy Distribution', fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Annotation
    fig.text(0.5, -0.02,
             'Pressure is dominated by low frequencies (smooth); '
             'saturation has significant high-frequency content (sharp fronts).\n'
             'This spectral separation motivates the operator-split predictor bank architecture.',
             ha='center', fontsize=9, fontstyle='italic', color='gray')

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate spectral analysis figure")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to field data (.pt file)")
    parser.add_argument("--output", type=str,
                        default="paper/figures/fig_spectral_analysis.pdf",
                        help="Output file path")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    args = parser.parse_args()

    data = load_or_generate_data(args.data_path)
    create_spectral_analysis_figure(data, args.output, args.dpi)


if __name__ == "__main__":
    main()
