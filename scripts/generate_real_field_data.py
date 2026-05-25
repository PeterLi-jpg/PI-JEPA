#!/usr/bin/env python
"""
Real field data pipeline for Sleipner or Norne datasets.

Handles irregular grid geometries and missing data, interpolating to
structured grids with metadata preservation.

Uses the IrregularGridProcessor from PI-JEPA/data/irregular_grid.py
to handle NaN values and non-uniform grid spacing.

Supported datasets:
- Sleipner CO2 storage (North Sea): CO2 plume monitoring data
- Norne field (Norwegian Sea): Production history with pressure/saturation

If real data files are not available, generates synthetic data with
similar characteristics (irregular grids, missing data, realistic statistics).
"""

import os
import sys
import argparse
import numpy as np
import torch

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from data.irregular_grid import IrregularGridProcessor


def generate_synthetic_sleipner(
    n_samples: int = 30,
    nx: int = 50, ny: int = 80,
    missing_fraction: float = 0.1,
    seed: int = 42,
) -> dict:
    """Generate synthetic Sleipner-like CO2 storage data.

    Mimics characteristics of the Sleipner CO2 injection site:
    - Irregular grid (stretched near injection well)
    - Missing data in some regions (seismic shadow zones)
    - CO2 saturation plume spreading laterally
    - Pressure buildup near injector

    Args:
        n_samples: Number of time snapshots.
        nx, ny: Original irregular grid dimensions.
        missing_fraction: Fraction of cells with missing data.
        seed: Random seed.

    Returns:
        Dictionary with:
        - 'data': (n_samples, 2, nx, ny) - [pressure, CO2_saturation]
        - 'grid_x': (nx, ny) - x-coordinates (non-uniform)
        - 'grid_y': (nx, ny) - y-coordinates (non-uniform)
        - 'metadata': dict with field info
    """
    rng = np.random.default_rng(seed)

    # Create non-uniform grid (stretched near center where injector is)
    # x-direction: finer near center
    x_1d = np.linspace(0, 1, nx)
    x_1d = x_1d + 0.1 * np.sin(2 * np.pi * x_1d)  # non-uniform stretching
    x_1d = (x_1d - x_1d.min()) / (x_1d.max() - x_1d.min())

    # y-direction: finer near injection point
    y_1d = np.linspace(0, 1, ny)
    y_1d = y_1d**1.3  # power-law stretching
    y_1d = (y_1d - y_1d.min()) / (y_1d.max() - y_1d.min())

    grid_x, grid_y = np.meshgrid(x_1d, y_1d, indexing='ij')

    # Generate CO2 plume evolution
    data = np.zeros((n_samples, 2, nx, ny), dtype=np.float32)

    # Injection point
    inj_x, inj_y = nx // 3, ny // 2

    for t in range(n_samples):
        time_factor = (t + 1) / n_samples

        # Pressure: radial buildup from injector, decaying with time
        r = np.sqrt((grid_x - grid_x[inj_x, inj_y])**2 +
                    (grid_y - grid_y[inj_x, inj_y])**2)
        pressure = 10.0 + 5.0 * time_factor / (r + 0.05)
        pressure += rng.normal(0, 0.1, (nx, ny))
        data[t, 0] = pressure

        # CO2 saturation: expanding plume
        plume_radius = 0.1 + 0.3 * time_factor
        plume = np.exp(-r**2 / (2 * plume_radius**2)) * 0.8 * time_factor
        # Add lateral spreading (gravity override)
        plume += 0.2 * time_factor * np.exp(
            -(grid_y - grid_y[inj_x, inj_y])**2 / 0.1
        ) * (grid_x > grid_x[inj_x, inj_y]).astype(float)
        plume = np.clip(plume, 0, 1)
        data[t, 1] = plume

    # Introduce missing data (NaN) - simulating seismic shadow zones
    nan_mask = rng.random((nx, ny)) < missing_fraction
    # Make missing data spatially correlated (patches, not random pixels)
    from scipy.ndimage import binary_dilation
    nan_mask = binary_dilation(nan_mask, iterations=2)

    for t in range(n_samples):
        data[t, 0][nan_mask] = np.nan
        data[t, 1][nan_mask] = np.nan

    metadata = {
        'field_name': 'Sleipner_synthetic',
        'n_samples': n_samples,
        'original_grid_size': (nx, ny),
        'grid_type': 'irregular_stretched',
        'missing_fraction': float(nan_mask.mean()),
        'channels': ['pressure_MPa', 'CO2_saturation'],
        'injection_location': (inj_x, inj_y),
    }

    return {
        'data': data,
        'grid_x': grid_x.astype(np.float32),
        'grid_y': grid_y.astype(np.float32),
        'metadata': metadata,
    }


def generate_synthetic_norne(
    n_samples: int = 40,
    nx: int = 46, ny: int = 112,
    missing_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Generate synthetic Norne-like production data.

    Mimics characteristics of the Norne field:
    - Complex fault structure (missing cells at faults)
    - Multiple wells (producers and injectors)
    - Pressure depletion and water breakthrough
    - Irregular corner-point grid

    Args:
        n_samples: Number of time snapshots.
        nx, ny: Grid dimensions.
        missing_fraction: Fraction of inactive/missing cells.
        seed: Random seed.

    Returns:
        Dictionary with same structure as generate_synthetic_sleipner.
    """
    rng = np.random.default_rng(seed)

    # Corner-point-like grid (non-uniform with fault offsets)
    x_1d = np.linspace(0, 1, nx)
    y_1d = np.linspace(0, 1, ny)

    # Add fault-like discontinuities
    fault_y = ny // 3
    y_1d[fault_y:] += 0.02  # offset at fault
    y_1d = (y_1d - y_1d.min()) / (y_1d.max() - y_1d.min())

    # Non-uniform spacing
    x_1d = x_1d + 0.05 * np.sin(4 * np.pi * x_1d)
    x_1d = (x_1d - x_1d.min()) / (x_1d.max() - x_1d.min())

    grid_x, grid_y = np.meshgrid(x_1d, y_1d, indexing='ij')

    # Well locations
    n_producers = 4
    n_injectors = 2
    producers = [(rng.integers(5, nx - 5), rng.integers(5, ny - 5))
                 for _ in range(n_producers)]
    injectors = [(rng.integers(5, nx - 5), rng.integers(5, ny - 5))
                 for _ in range(n_injectors)]

    # Generate production data
    data = np.zeros((n_samples, 2, nx, ny), dtype=np.float32)

    for t in range(n_samples):
        time_factor = (t + 1) / n_samples

        # Pressure: depletion near producers, buildup near injectors
        pressure = np.ones((nx, ny)) * 30.0  # initial reservoir pressure (MPa)

        for px, py in producers:
            r = np.sqrt((grid_x - grid_x[px, py])**2 +
                        (grid_y - grid_y[px, py])**2)
            pressure -= 10.0 * time_factor / (r + 0.05)

        for ix, iy in injectors:
            r = np.sqrt((grid_x - grid_x[ix, iy])**2 +
                        (grid_y - grid_y[ix, iy])**2)
            pressure += 3.0 * time_factor / (r + 0.05)

        pressure += rng.normal(0, 0.2, (nx, ny))
        data[t, 0] = pressure

        # Water saturation: water front advancing from injectors
        Sw = np.ones((nx, ny)) * 0.2  # connate water
        for ix, iy in injectors:
            r = np.sqrt((grid_x - grid_x[ix, iy])**2 +
                        (grid_y - grid_y[ix, iy])**2)
            front_radius = 0.05 + 0.4 * time_factor
            Sw += 0.6 * np.clip(1.0 - r / front_radius, 0, 1)

        Sw = np.clip(Sw + rng.normal(0, 0.02, (nx, ny)), 0, 1)
        data[t, 1] = Sw

    # Missing data: fault zones and inactive cells
    nan_mask = np.zeros((nx, ny), dtype=bool)
    # Fault zone
    fault_width = 2
    nan_mask[:, fault_y - fault_width:fault_y + fault_width] = True
    # Random inactive cells
    random_inactive = rng.random((nx, ny)) < (missing_fraction - nan_mask.mean())
    nan_mask |= random_inactive

    for t in range(n_samples):
        data[t, 0][nan_mask] = np.nan
        data[t, 1][nan_mask] = np.nan

    metadata = {
        'field_name': 'Norne_synthetic',
        'n_samples': n_samples,
        'original_grid_size': (nx, ny),
        'grid_type': 'corner_point_irregular',
        'missing_fraction': float(nan_mask.mean()),
        'channels': ['pressure_MPa', 'water_saturation'],
        'n_producers': n_producers,
        'n_injectors': n_injectors,
        'producer_locations': producers,
        'injector_locations': injectors,
    }

    return {
        'data': data,
        'grid_x': grid_x.astype(np.float32),
        'grid_y': grid_y.astype(np.float32),
        'metadata': metadata,
    }


def process_field_data(
    raw_data: dict,
    target_resolution: int = 64,
    interpolation_method: str = 'bilinear',
) -> dict:
    """Process raw field data through the IrregularGridProcessor pipeline.

    Args:
        raw_data: Dictionary from generate_synthetic_* functions.
        target_resolution: Output structured grid resolution.
        interpolation_method: Interpolation mode.

    Returns:
        Dictionary with:
        - 'tensors': (n_samples, C, target_res, target_res) processed tensors
        - 'metadata': preserved metadata with processing info added
    """
    processor = IrregularGridProcessor(
        target_resolution=target_resolution,
        interpolation_method=interpolation_method,
    )

    data = raw_data['data']  # (n_samples, C, H, W)
    grid_x = raw_data['grid_x']
    grid_y = raw_data['grid_y']

    n_samples = data.shape[0]
    processed_tensors = []

    for i in range(n_samples):
        sample = torch.from_numpy(data[i]).unsqueeze(0)  # (1, C, H, W)

        # Process through pipeline (handles NaN + interpolation)
        grid_metadata = {
            'grid_x': torch.from_numpy(grid_x),
            'grid_y': torch.from_numpy(grid_y),
        }
        processed = processor.process(sample, metadata=grid_metadata)
        processed_tensors.append(processed.squeeze(0))  # (C, target_res, target_res)

    tensors = torch.stack(processed_tensors)  # (n_samples, C, target_res, target_res)

    # Verify output quality
    assert torch.isfinite(tensors).all(), "Output contains non-finite values"

    # Update metadata
    metadata = raw_data['metadata'].copy()
    metadata['processed_resolution'] = target_resolution
    metadata['interpolation_method'] = interpolation_method
    metadata['output_shape'] = list(tensors.shape)

    return {
        'tensors': tensors,
        'metadata': metadata,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Process real field data (Sleipner/Norne) for PI-JEPA"
    )
    parser.add_argument("--field", type=str, choices=["sleipner", "norne", "both"],
                        default="both", help="Which field to process")
    parser.add_argument("--data-path", type=str, default=None,
                        help="Path to real field data (if available)")
    parser.add_argument("--target-resolution", type=int, default=64,
                        help="Output structured grid resolution")
    parser.add_argument("--n-samples", type=int, default=30,
                        help="Number of time snapshots (for synthetic)")
    parser.add_argument("--output-dir", type=str, default="data/real_field",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    fields_to_process = []
    if args.field in ("sleipner", "both"):
        fields_to_process.append("sleipner")
    if args.field in ("norne", "both"):
        fields_to_process.append("norne")

    for field_name in fields_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {field_name.upper()} field data")
        print(f"{'='*60}")

        # Generate synthetic data (real data loading would go here)
        if args.data_path and os.path.exists(args.data_path):
            print(f"  Loading real data from {args.data_path}...")
            # TODO: Implement real data loading for specific formats
            # For now, fall back to synthetic
            print("  Real data loading not yet implemented, using synthetic.")

        if field_name == "sleipner":
            print("  Generating synthetic Sleipner-like data...")
            raw_data = generate_synthetic_sleipner(
                n_samples=args.n_samples, seed=args.seed
            )
        else:
            print("  Generating synthetic Norne-like data...")
            raw_data = generate_synthetic_norne(
                n_samples=args.n_samples, seed=args.seed
            )

        print(f"  Raw data shape: {raw_data['data'].shape}")
        print(f"  Grid: {raw_data['grid_x'].shape} (non-uniform)")
        print(f"  Missing data: {raw_data['metadata']['missing_fraction']:.1%}")

        # Process through IrregularGridProcessor
        print(f"  Processing to {args.target_resolution}x{args.target_resolution} "
              f"structured grid...")
        processed = process_field_data(
            raw_data,
            target_resolution=args.target_resolution,
        )

        print(f"  Output tensor shape: {processed['tensors'].shape}")
        print(f"  Value range: [{processed['tensors'].min():.4f}, "
              f"{processed['tensors'].max():.4f}]")
        print(f"  All finite: {torch.isfinite(processed['tensors']).all()}")

        # Save
        output_path = os.path.join(args.output_dir, f"{field_name}_processed.pt")
        torch.save(processed, output_path)
        print(f"  Saved: {output_path}")

        # Also save raw data for reference
        raw_path = os.path.join(args.output_dir, f"{field_name}_raw.pt")
        # Convert numpy arrays to tensors for saving
        raw_save = {
            'data': torch.from_numpy(raw_data['data']),
            'grid_x': torch.from_numpy(raw_data['grid_x']),
            'grid_y': torch.from_numpy(raw_data['grid_y']),
            'metadata': raw_data['metadata'],
        }
        torch.save(raw_save, raw_path)
        print(f"  Saved raw: {raw_path}")

    print(f"\n{'='*60}")
    print("Done. All field data processed successfully.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
