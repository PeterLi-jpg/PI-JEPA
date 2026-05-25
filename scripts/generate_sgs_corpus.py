#!/usr/bin/env python
"""
Generate SGS (Sequential Gaussian Simulation) permeability corpus.

Uses spectral method (FFT-based) to generate spatially correlated Gaussian
random fields, producing realistic permeability realizations at ~5ms each.

Output: .pt files with shape (N, 1, 64, 64) containing log-normal permeability fields.
"""

import os
import argparse
import time
import numpy as np
import torch


def build_spectral_covariance(resolution: int, correlation_length: float, variance: float = 1.0):
    """Build the spectral covariance (power spectrum) for FFT-based generation.

    Uses an exponential covariance model: C(r) = variance * exp(-r / correlation_length)
    whose power spectrum is analytically known.

    Args:
        resolution: Grid size (square grid resolution x resolution).
        correlation_length: Spatial correlation length in grid units.
        variance: Variance of the Gaussian field.

    Returns:
        sqrt_spectrum: (resolution, resolution//2+1) array of sqrt(power spectrum)
                       for use with rfft2.
    """
    # Frequency grids for rfft2
    kx = np.fft.fftfreq(resolution, d=1.0 / resolution)
    ky = np.fft.rfftfreq(resolution, d=1.0 / resolution)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K_mag = np.sqrt(KX**2 + KY**2)

    # Power spectrum of exponential covariance in 2D:
    # S(k) = (2 * pi * variance * L^2) / (1 + (2*pi*L*|k|/N)^2)^(3/2)
    # where L is correlation length in grid units
    L = correlation_length
    k_scaled = 2.0 * np.pi * L * K_mag / resolution
    spectrum = (2.0 * np.pi * variance * L**2) / (1.0 + k_scaled**2)**1.5

    # Normalize so that the field has the desired variance
    # The total power should equal variance * resolution^2
    total_power = spectrum.sum()
    if total_power > 0:
        spectrum *= (variance * resolution**2) / total_power

    sqrt_spectrum = np.sqrt(np.maximum(spectrum, 0.0)).astype(np.float32)
    return sqrt_spectrum


def generate_sgs_realizations(
    n_realizations: int,
    resolution: int = 64,
    correlation_length: float = 8.0,
    variance: float = 1.0,
    log_perm_mean: float = 0.0,
    log_perm_std: float = 2.0,
    seed: int = 42,
) -> torch.Tensor:
    """Generate SGS permeability realizations using spectral method.

    The spectral method generates correlated Gaussian random fields by:
    1. Drawing white noise in Fourier space
    2. Multiplying by sqrt(power spectrum) to impose correlation structure
    3. Inverse FFT to get spatial field
    4. Exponentiating to get log-normal permeability

    This is extremely fast (~5ms per realization) compared to traditional
    SGS which requires sequential visiting of grid nodes.

    Args:
        n_realizations: Number of realizations to generate.
        resolution: Grid resolution (square).
        correlation_length: Spatial correlation length in grid cells.
        variance: Variance of the underlying Gaussian field.
        log_perm_mean: Mean of log-permeability.
        log_perm_std: Std of log-permeability.
        seed: Random seed.

    Returns:
        Tensor of shape (n_realizations, 1, resolution, resolution) with
        log-normal permeability values.
    """
    rng = np.random.default_rng(seed)

    # Pre-compute spectral covariance (shared across all realizations)
    sqrt_spectrum = build_spectral_covariance(resolution, correlation_length, variance)

    # Generate all realizations at once for efficiency
    # Shape: (n_realizations, resolution, resolution//2+1) complex noise
    n_freq = resolution // 2 + 1
    noise_real = rng.standard_normal((n_realizations, resolution, n_freq)).astype(np.float32)
    noise_imag = rng.standard_normal((n_realizations, resolution, n_freq)).astype(np.float32)
    noise_complex = noise_real + 1j * noise_imag

    # Apply spectral covariance
    filtered = noise_complex * sqrt_spectrum[np.newaxis, :, :]

    # Inverse FFT to get spatial fields
    fields = np.fft.irfft2(filtered, s=(resolution, resolution))

    # Normalize to desired log-perm statistics
    fields = (fields - fields.mean(axis=(1, 2), keepdims=True)) / (
        fields.std(axis=(1, 2), keepdims=True) + 1e-8
    )
    fields = log_perm_mean + log_perm_std * fields

    # Exponentiate to get log-normal permeability
    permeability = np.exp(fields).astype(np.float32)

    # Convert to torch tensor with channel dimension: (N, 1, H, W)
    tensor = torch.from_numpy(permeability).unsqueeze(1)
    return tensor


def main():
    parser = argparse.ArgumentParser(
        description="Generate SGS permeability corpus for PI-JEPA pretraining"
    )
    parser.add_argument("--n-realizations", type=int, default=10000,
                        help="Number of permeability realizations to generate")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Grid resolution (square)")
    parser.add_argument("--correlation-length", type=float, default=8.0,
                        help="Spatial correlation length in grid cells")
    parser.add_argument("--variance", type=float, default=1.0,
                        help="Variance of underlying Gaussian field")
    parser.add_argument("--log-perm-mean", type=float, default=0.0,
                        help="Mean of log-permeability")
    parser.add_argument("--log-perm-std", type=float, default=2.0,
                        help="Std of log-permeability")
    parser.add_argument("--output-dir", type=str, default="data/sgs_corpus",
                        help="Output directory for .pt files")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Number of realizations per .pt file")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run timing benchmark")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.benchmark:
        # Benchmark generation speed
        print("Running timing benchmark...")
        n_bench = 100
        start = time.time()
        _ = generate_sgs_realizations(
            n_bench, args.resolution, args.correlation_length,
            args.variance, args.log_perm_mean, args.log_perm_std, args.seed
        )
        elapsed = time.time() - start
        per_realization_ms = (elapsed / n_bench) * 1000
        print(f"  {n_bench} realizations in {elapsed:.3f}s")
        print(f"  {per_realization_ms:.2f} ms per realization")
        print(f"  Target: ~5ms per realization")
        return

    # Generate corpus in batches
    n_total = args.n_realizations
    batch_size = args.batch_size
    n_batches = (n_total + batch_size - 1) // batch_size

    print(f"Generating {n_total} SGS permeability realizations...")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Correlation length: {args.correlation_length} cells")
    print(f"  Output: {args.output_dir}/")
    print(f"  Batches: {n_batches} x {batch_size}")

    total_start = time.time()

    for batch_idx in range(n_batches):
        n_this_batch = min(batch_size, n_total - batch_idx * batch_size)
        batch_seed = args.seed + batch_idx * 1000

        start = time.time()
        realizations = generate_sgs_realizations(
            n_this_batch,
            args.resolution,
            args.correlation_length,
            args.variance,
            args.log_perm_mean,
            args.log_perm_std,
            batch_seed,
        )
        elapsed = time.time() - start

        # Save as .pt file
        filename = f"sgs_batch_{batch_idx:04d}.pt"
        filepath = os.path.join(args.output_dir, filename)
        torch.save(realizations, filepath)

        per_ms = (elapsed / n_this_batch) * 1000
        print(f"  Batch {batch_idx+1}/{n_batches}: {n_this_batch} realizations "
              f"in {elapsed:.2f}s ({per_ms:.2f} ms/realization) -> {filename}")

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total time: {total_elapsed:.1f}s")
    print(f"  Average: {(total_elapsed / n_total) * 1000:.2f} ms/realization")
    print(f"  Output shape per batch: ({batch_size}, 1, {args.resolution}, {args.resolution})")


if __name__ == "__main__":
    main()
