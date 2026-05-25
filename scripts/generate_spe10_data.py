#!/usr/bin/env python
"""
SPE10 3D data loading and preprocessing.

Loads SPE10 Model 2 (Tarbert formation) permeability data and preprocesses
it into tensors of shape (N, C, 32, 32, 32) suitable for the 3D encoder.

If the actual SPE10 dataset is not available, generates synthetic SPE10-like
data with similar statistical properties (high heterogeneity, channelized
structures, 6 orders of magnitude permeability contrast).

SPE10 Model 2 dimensions: 60 x 220 x 85 cells
- Tarbert formation: layers 1-35 (shallow marine)
- Upper Ness: layers 36-85 (fluvial)

We extract 32x32x32 sub-volumes from the Tarbert formation.
"""

import os
import argparse
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, zoom


def load_spe10_data(data_path: str) -> np.ndarray:
    """Load SPE10 permeability data from file.

    Expected format: binary file with 60*220*85 float64 values
    (standard SPE10 distribution format).

    Args:
        data_path: Path to SPE10 permeability file (e.g., 'spe_perm.dat').

    Returns:
        Permeability array of shape (60, 220, 85).
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"SPE10 data not found at {data_path}. "
            "Use --synthetic flag to generate synthetic data instead."
        )

    # SPE10 standard format: 60*220*85 = 1,122,000 values
    n_cells = 60 * 220 * 85
    perm = np.fromfile(data_path, dtype=np.float64, count=n_cells)
    perm = perm.reshape((60, 220, 85), order='F')  # Fortran ordering
    return perm


def generate_synthetic_spe10(
    nx: int = 60, ny: int = 220, nz: int = 85,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic SPE10-like permeability field.

    Mimics the statistical properties of the Tarbert formation:
    - Log-normal distribution with high variance
    - Layered structure with lateral correlation
    - Permeability contrast of ~6 orders of magnitude
    - Channelized features

    Args:
        nx, ny, nz: Grid dimensions.
        seed: Random seed.

    Returns:
        Permeability array of shape (nx, ny, nz).
    """
    rng = np.random.default_rng(seed)

    # Generate layered structure with different correlation per layer
    perm = np.zeros((nx, ny, nz))

    for k in range(nz):
        # Each layer has different correlation structure
        noise = rng.standard_normal((nx, ny))

        # Lateral correlation (anisotropic: longer in y-direction)
        sigma_x = 2.0 + rng.random() * 3.0
        sigma_y = 5.0 + rng.random() * 8.0
        smoothed = gaussian_filter(noise, sigma=[sigma_x, sigma_y], mode='wrap')

        # Add channelized features (random channels in some layers)
        if rng.random() > 0.5:
            channel = np.zeros((nx, ny))
            n_channels = rng.integers(1, 4)
            for _ in range(n_channels):
                cx = rng.integers(5, nx - 5)
                width = rng.integers(2, 6)
                channel[max(0, cx - width):min(nx, cx + width), :] = (
                    rng.random() * 3.0
                )
            channel = gaussian_filter(channel, sigma=[1.0, 3.0])
            smoothed += channel

        # Normalize and scale to get ~6 orders of magnitude range
        smoothed = smoothed / (smoothed.std() + 1e-8)
        # Log-perm: mean ~1, std ~3 gives range of ~6 orders
        log_perm = 1.0 + 2.5 * smoothed
        perm[:, :, k] = np.exp(log_perm)

    # Add vertical correlation (adjacent layers are correlated)
    for k in range(1, nz):
        alpha = 0.3 + 0.4 * rng.random()  # correlation strength
        perm[:, :, k] = alpha * perm[:, :, k - 1] + (1 - alpha) * perm[:, :, k]

    return perm


def extract_subvolumes(
    perm: np.ndarray,
    target_size: int = 32,
    n_samples: int = 50,
    layer_range: tuple = (0, 35),
    seed: int = 42,
) -> np.ndarray:
    """Extract 3D sub-volumes from the full SPE10 field.

    Extracts sub-volumes of size (target_size, target_size, target_size)
    from the specified layer range (Tarbert formation = layers 0-34).

    If the source grid is smaller than target_size in any dimension,
    we use interpolation to resize.

    Args:
        perm: Full permeability field (nx, ny, nz).
        target_size: Output sub-volume size (cubic).
        n_samples: Number of sub-volumes to extract.
        layer_range: (start, end) layer indices for extraction.
        seed: Random seed for random cropping.

    Returns:
        Array of shape (n_samples, target_size, target_size, target_size).
    """
    rng = np.random.default_rng(seed)
    nx, ny, nz = perm.shape
    z_start, z_end = layer_range
    z_depth = z_end - z_start

    # Work with the specified layer range
    perm_subset = perm[:, :, z_start:z_end]

    subvolumes = []

    for _ in range(n_samples):
        # Random crop or resize depending on available dimensions
        if nx >= target_size and ny >= target_size and z_depth >= target_size:
            # Random crop
            ix = rng.integers(0, nx - target_size + 1)
            iy = rng.integers(0, ny - target_size + 1)
            iz = rng.integers(0, z_depth - target_size + 1)
            subvol = perm_subset[ix:ix + target_size,
                                  iy:iy + target_size,
                                  iz:iz + target_size]
        else:
            # Resize to target using zoom
            # Random crop to largest possible cube first
            crop_size = min(nx, ny, z_depth)
            ix = rng.integers(0, max(1, nx - crop_size + 1))
            iy = rng.integers(0, max(1, ny - crop_size + 1))
            iz = rng.integers(0, max(1, z_depth - crop_size + 1))
            subvol = perm_subset[ix:ix + crop_size,
                                  iy:iy + crop_size,
                                  iz:iz + crop_size]
            # Resize to target
            zoom_factors = [target_size / s for s in subvol.shape]
            subvol = zoom(subvol, zoom_factors, order=1)

        # Apply log transform for better numerical properties
        subvol = np.log(np.clip(subvol, 1e-6, 1e12))

        subvolumes.append(subvol)

    return np.stack(subvolumes)


def generate_pressure_saturation_3d(
    perm_log: np.ndarray, n_steps: int = 5, seed: int = 42,
) -> tuple:
    """Generate simplified 3D pressure and saturation fields.

    Uses a simplified diffusion-based approach for 3D fields
    (full 3D IMPES would be too expensive for data generation).

    Args:
        perm_log: Log-permeability sub-volume (target_size, target_size, target_size).
        n_steps: Number of timesteps.
        seed: Random seed.

    Returns:
        (pressure, saturation): Each of shape (n_steps, target_size, target_size, target_size).
    """
    rng = np.random.default_rng(seed)
    N = perm_log.shape[0]
    K = np.exp(perm_log)

    # Simplified: pressure from Laplacian solve approximation
    # Use iterative smoothing as proxy for pressure solve
    pressures = []
    saturations = []

    # Initial saturation
    Sw = np.ones((N, N, N)) * 0.2

    for t in range(n_steps):
        # Approximate pressure: smooth random field weighted by permeability
        noise = rng.standard_normal((N, N, N))
        p = gaussian_filter(noise * np.sqrt(K), sigma=3.0)
        p = (p - p.mean()) / (p.std() + 1e-8)

        pressures.append(p)
        saturations.append(Sw.copy())

        # Simple saturation evolution (diffusion + advection proxy)
        grad_p = np.gradient(p)
        flux_mag = np.sqrt(sum(g**2 for g in grad_p))
        Sw = Sw + 0.02 * gaussian_filter(flux_mag * K, sigma=1.0)
        Sw = np.clip(Sw, 0.0, 1.0)

    return np.stack(pressures), np.stack(saturations)


def preprocess_to_tensors(
    perm: np.ndarray,
    target_size: int = 32,
    n_samples: int = 50,
    n_steps: int = 5,
    seed: int = 42,
) -> dict:
    """Full preprocessing pipeline: extract sub-volumes and generate fields.

    Args:
        perm: Full permeability field.
        target_size: Output tensor spatial size.
        n_samples: Number of samples.
        n_steps: Timesteps per sample.
        seed: Random seed.

    Returns:
        Dictionary with:
        - 'permeability': (n_samples, 1, D, H, W) log-permeability
        - 'pressure': (n_samples, n_steps, D, H, W)
        - 'saturation': (n_samples, n_steps, D, H, W)
    """
    # Extract sub-volumes
    subvolumes = extract_subvolumes(perm, target_size, n_samples, seed=seed)

    all_pressure = []
    all_saturation = []

    for i in range(n_samples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"    Processing sub-volume {i+1}/{n_samples}")

        p, s = generate_pressure_saturation_3d(
            subvolumes[i], n_steps, seed=seed + i
        )
        all_pressure.append(p)
        all_saturation.append(s)

    return {
        'permeability': subvolumes[:, np.newaxis, :, :, :].astype(np.float32),
        'pressure': np.stack(all_pressure).astype(np.float32),
        'saturation': np.stack(all_saturation).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Load and preprocess SPE10 3D data for PI-JEPA"
    )
    parser.add_argument("--spe10-path", type=str, default=None,
                        help="Path to SPE10 permeability file (spe_perm.dat)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Generate synthetic SPE10-like data (if real data unavailable)")
    parser.add_argument("--target-size", type=int, default=32,
                        help="Output sub-volume size (cubic)")
    parser.add_argument("--n-train", type=int, default=50,
                        help="Number of training sub-volumes")
    parser.add_argument("--n-test", type=int, default=10,
                        help="Number of test sub-volumes")
    parser.add_argument("--n-steps", type=int, default=5,
                        help="Timesteps per sample")
    parser.add_argument("--output-dir", type=str, default="data/spe10",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load or generate permeability field
    if args.spe10_path and not args.synthetic:
        print(f"Loading SPE10 data from {args.spe10_path}...")
        try:
            perm = load_spe10_data(args.spe10_path)
            print(f"  Loaded: shape {perm.shape}, "
                  f"range [{perm.min():.2e}, {perm.max():.2e}]")
        except FileNotFoundError as e:
            print(f"  {e}")
            print("  Falling back to synthetic data generation.")
            perm = generate_synthetic_spe10(seed=args.seed)
    else:
        print("Generating synthetic SPE10-like permeability field...")
        perm = generate_synthetic_spe10(seed=args.seed)
        print(f"  Generated: shape {perm.shape}, "
              f"range [{perm.min():.2e}, {perm.max():.2e}]")

    # Process training data
    print(f"\nPreprocessing training data ({args.n_train} sub-volumes)...")
    print(f"  Target shape: ({args.n_train}, C, {args.target_size}, "
          f"{args.target_size}, {args.target_size})")

    train_data = preprocess_to_tensors(
        perm, args.target_size, args.n_train, args.n_steps, args.seed
    )

    train_path = os.path.join(args.output_dir, "spe10_train.pt")
    torch.save(train_data, train_path)
    print(f"  Saved: {train_path}")
    for key, val in train_data.items():
        print(f"    {key}: {val.shape}")

    # Process test data
    print(f"\nPreprocessing test data ({args.n_test} sub-volumes)...")
    test_data = preprocess_to_tensors(
        perm, args.target_size, args.n_test, args.n_steps, args.seed + 1000
    )

    test_path = os.path.join(args.output_dir, "spe10_test.pt")
    torch.save(test_data, test_path)
    print(f"  Saved: {test_path}")
    for key, val in test_data.items():
        print(f"    {key}: {val.shape}")

    print("\nDone.")


if __name__ == "__main__":
    main()
