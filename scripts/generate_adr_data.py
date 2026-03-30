#!/usr/bin/env python
"""
Generate advection-diffusion-reaction (ADR) dataset at 64x64 resolution.

Solves:  ∂c_i/∂t + v·∇c_i = D_i ∇²c_i + R_i(c)

for n_species=2 reacting species on a 64x64 grid, across multiple
Péclet / Damköhler regimes, matching the PDEBench ADR format.
"""

import os
import argparse
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter


def advection_diffusion_reaction_step(
    c, vx, vy, D, Da, dx, dt, n_species=2
):
    """Single explicit timestep for ADR system.

    Args:
        c: (n_species, H, W) concentration fields
        vx, vy: (H, W) velocity components
        D: diffusivity scalar
        Da: Damköhler number (reaction rate)
        dx: grid spacing
        dt: timestep
    """
    c_new = c.copy()
    for s in range(n_species):
        cs = c[s]

        # Advection (upwind)
        dc_dx = np.zeros_like(cs)
        dc_dy = np.zeros_like(cs)
        dc_dx[:, 1:-1] = (cs[:, 2:] - cs[:, :-2]) / (2 * dx)
        dc_dy[1:-1, :] = (cs[2:, :] - cs[:-2, :]) / (2 * dx)
        advection = vx * dc_dx + vy * dc_dy

        # Diffusion (central)
        lap = np.zeros_like(cs)
        lap[:, 1:-1] += (cs[:, 2:] - 2 * cs[:, 1:-1] + cs[:, :-2]) / dx**2
        lap[1:-1, :] += (cs[2:, :] - 2 * cs[1:-1, :] + cs[:-2, :]) / dx**2
        diffusion = D * lap

        # Reaction: simple coupled system  R_0 = -Da*c0*c1,  R_1 = +Da*c0*c1
        reaction = Da * c[0] * c[1] * ((-1.0) ** s)

        c_new[s] = cs + dt * (-advection + diffusion + reaction)
        c_new[s] = np.clip(c_new[s], 0.0, 10.0)

    return c_new


def generate_trajectory(N, n_steps, Pe, Da, rng, n_species=2):
    """Generate one ADR trajectory."""
    dx = 1.0 / (N - 1)
    D = 1.0 / (Pe + 1e-8)
    dt = 0.2 * dx**2 / (D + 1e-8)  # CFL-like stability
    dt = min(dt, 0.001)

    # Random smooth velocity field
    vx_raw = rng.standard_normal((N, N))
    vy_raw = rng.standard_normal((N, N))
    vx = gaussian_filter(vx_raw, sigma=N * 0.1, mode="wrap") * Pe * 0.1
    vy = gaussian_filter(vy_raw, sigma=N * 0.1, mode="wrap") * Pe * 0.1

    # Initial concentrations: smooth random blobs
    c = np.zeros((n_species, N, N), dtype=np.float32)
    for s in range(n_species):
        blob = rng.standard_normal((N, N))
        c[s] = np.clip(gaussian_filter(blob, sigma=N * 0.08, mode="wrap") + 0.5, 0.0, 2.0)

    trajectory = [c.copy()]
    sub_steps = max(1, int(0.05 / dt))  # ~0.05 time units between snapshots

    for t in range(n_steps):
        for _ in range(sub_steps):
            c = advection_diffusion_reaction_step(c, vx, vy, D, Da, dx, dt, n_species)
        trajectory.append(c.copy())

    concentration = np.stack(trajectory)  # (T+1, n_species, H, W)
    # Transpose to (n_species, T+1, H, W) to match PDEBenchADRDataset
    concentration = concentration.transpose(1, 0, 2, 3)

    return concentration, vx, vy, D, Da


def generate_regime(n_samples, N, n_steps, Pe, Da, seed, n_species=2):
    """Generate data for one (Pe, Da) regime."""
    rng = np.random.default_rng(seed)
    all_conc, all_vel, all_diff, all_react = [], [], [], []

    for i in range(n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"    Sample {i+1}/{n_samples}  Pe={Pe} Da={Da}")
        conc, vx, vy, D, Da_val = generate_trajectory(N, n_steps, Pe, Da, rng, n_species)
        all_conc.append(conc)
        vel = np.stack([vx, vy])  # (2, H, W)
        all_vel.append(vel)
        all_diff.append(D)
        all_react.append(Da_val)

    return {
        "concentration": np.stack(all_conc).astype(np.float32),
        "velocity": np.stack(all_vel).astype(np.float32),
        "diffusivity": np.array(all_diff, dtype=np.float32),
        "reaction_rate": np.array(all_react, dtype=np.float32),
    }


def save_hdf5(data, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr, compression="gzip")
    print(f"  Saved {path}  shapes: { {k: v.shape for k, v in data.items()} }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ADR benchmark data")
    parser.add_argument("--n-train", type=int, default=200)
    parser.add_argument("--n-test", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--output", default="data/adr")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Paper specifies Pe ∈ {0.1, 1, 10}, Da ∈ {0.01, 0.1, 1.0}
    regimes = [
        (1.0, 0.1),    # moderate Pe, moderate Da
        (10.0, 1.0),   # high Pe, high Da
        (0.1, 0.01),   # low Pe, low Da
    ]

    for Pe, Da in regimes:
        tag = f"Pe{Pe}_Da{Da}"
        print(f"\nRegime: {tag}")

        print("  Training data...")
        train = generate_regime(args.n_train, args.resolution, args.n_steps, Pe, Da, args.seed)
        save_hdf5(train, os.path.join(args.output, f"adr_train_{tag}.h5"))

        print("  Test data...")
        test = generate_regime(args.n_test, args.resolution, args.n_steps, Pe, Da, args.seed + 1)
        save_hdf5(test, os.path.join(args.output, f"adr_test_{tag}.h5"))

    print("\nDone.")
