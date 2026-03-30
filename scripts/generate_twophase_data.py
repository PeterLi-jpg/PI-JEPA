#!/usr/bin/env python
"""
Generate two-phase Darcy flow dataset (CO2-water style) at 64x64 resolution.

Solves the coupled IMPES system:
  - Pressure equation (elliptic):  -∇·(λ_T K ∇p) = q_T
  - Saturation transport (hyperbolic): φ ∂S_w/∂t + ∇·(f_w v_T) = q_w

Generates heterogeneous permeability fields and evolves pressure + saturation
over multiple timesteps, producing trajectory data similar to the U-FNO CO2
dataset format.
"""

import os
import argparse
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter


def generate_permeability_field(N, length_scale=0.1, variance=1.0, rng=None):
    """Generate log-normal permeability with spatial correlation."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal((N, N))
    sigma = length_scale * N
    smoothed = gaussian_filter(noise, sigma=sigma, mode="wrap")
    smoothed = smoothed / (smoothed.std() + 1e-8) * np.sqrt(variance)
    return np.exp(smoothed)


def solve_pressure(K, S_w, mu_w=1.0, mu_n=5.0, q_T=None, N=64):
    """Solve elliptic pressure equation with IMPES splitting."""
    h = 1.0 / (N - 1)

    # Relative permeabilities (quadratic Brooks-Corey)
    S_e = np.clip(S_w, 0.01, 0.99)
    kr_w = S_e ** 2
    kr_n = (1 - S_e) ** 2
    lam_w = kr_w / mu_w
    lam_n = kr_n / mu_n
    lam_T = lam_w + lam_n

    # Total mobility * permeability
    T = lam_T * K

    # Build sparse system for interior nodes
    n_int = (N - 2) ** 2
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_int)

    if q_T is None:
        q_T = np.zeros((N, N))
        q_T[N // 4, N // 4] = 1.0  # injection
        q_T[3 * N // 4, 3 * N // 4] = -1.0  # production

    def idx(i, j):
        return (i - 1) * (N - 2) + (j - 1)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            k = idx(i, j)
            center = 0.0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                t_face = 2.0 * T[i, j] * T[ni, nj] / (T[i, j] + T[ni, nj] + 1e-12)
                coeff = t_face / h ** 2
                center -= coeff
                if 1 <= ni <= N - 2 and 1 <= nj <= N - 2:
                    rows.append(k)
                    cols.append(idx(ni, nj))
                    vals.append(coeff)
            rows.append(k)
            cols.append(k)
            vals.append(center)
            rhs[k] = q_T[i, j]

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))
    p_int = spsolve(A, rhs)

    p = np.zeros((N, N))
    p[1:-1, 1:-1] = p_int.reshape(N - 2, N - 2)
    return p


def update_saturation(S_w, p, K, phi, mu_w=1.0, mu_n=5.0, dt=0.005, N=64, q_w=None):
    """Explicit saturation transport step."""
    h = 1.0 / (N - 1)
    S_e = np.clip(S_w, 0.01, 0.99)
    kr_w = S_e ** 2
    kr_n = (1 - S_e) ** 2
    lam_w = kr_w / mu_w
    lam_n = kr_n / mu_n
    lam_T = lam_w + lam_n + 1e-12
    f_w = lam_w / lam_T

    # Darcy velocity from pressure gradient
    dp_dx = np.zeros_like(p)
    dp_dy = np.zeros_like(p)
    dp_dx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * h)
    dp_dy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * h)

    v_x = -lam_T * K * dp_dx
    v_y = -lam_T * K * dp_dy

    # Upwind flux for saturation
    flux_x = f_w * v_x
    flux_y = f_w * v_y

    div_flux = np.zeros_like(S_w)
    div_flux[:, 1:-1] += (flux_x[:, 2:] - flux_x[:, :-2]) / (2 * h)
    div_flux[1:-1, :] += (flux_y[2:, :] - flux_y[:-2, :]) / (2 * h)

    if q_w is None:
        q_w = np.zeros((N, N))
        q_w[N // 4, N // 4] = 0.5

    S_new = S_w - dt / (phi + 1e-8) * div_flux + dt / (phi + 1e-8) * q_w
    return np.clip(S_new, 0.0, 1.0)


def generate_trajectory(K, phi, n_steps=10, N=64, seed=None):
    """Generate a full pressure-saturation trajectory."""
    S_w = np.ones((N, N)) * 0.2  # initial water saturation
    pressures = []
    saturations = []

    for t in range(n_steps):
        p = solve_pressure(K, S_w, N=N)
        pressures.append(p.copy())
        saturations.append(S_w.copy())
        S_w = update_saturation(S_w, p, K, phi, N=N)

    return np.stack(pressures), np.stack(saturations)


def generate_dataset(n_samples=500, resolution=64, n_steps=10, seed=42):
    """Generate full two-phase dataset."""
    rng = np.random.default_rng(seed)

    all_pressure = []
    all_saturation = []
    all_perm = []
    all_porosity = []

    for i in range(n_samples):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Generating sample {i+1}/{n_samples}")

        K = generate_permeability_field(resolution, rng=rng)
        phi = 0.15 + 0.1 * rng.random((resolution, resolution))

        p_traj, s_traj = generate_trajectory(K, phi, n_steps=n_steps, N=resolution)

        all_pressure.append(p_traj)
        all_saturation.append(s_traj)
        all_perm.append(K)
        all_porosity.append(phi)

    return {
        "pressure": np.stack(all_pressure).astype(np.float32),
        "saturation": np.stack(all_saturation).astype(np.float32),
        "permeability": np.stack(all_perm).astype(np.float32),
        "porosity": np.stack(all_porosity).astype(np.float32),
    }


def save_hdf5(data, path):
    """Save dataset in HDF5 format compatible with UFNODataset."""
    import h5py
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with h5py.File(path, "w") as f:
        for key, arr in data.items():
            f.create_dataset(key, data=arr, compression="gzip")
    print(f"Saved {path}  shapes: { {k: v.shape for k, v in data.items()} }")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate two-phase Darcy flow data")
    parser.add_argument("--n-train", type=int, default=500)
    parser.add_argument("--n-test", type=int, default=100)
    parser.add_argument("--resolution", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=10)
    parser.add_argument("--output", default="data/twophase")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Generating training data...")
    train_data = generate_dataset(args.n_train, args.resolution, args.n_steps, args.seed)
    save_hdf5(train_data, os.path.join(args.output, "twophase_train.h5"))

    print("Generating test data...")
    test_data = generate_dataset(args.n_test, args.resolution, args.n_steps, args.seed + 1)
    save_hdf5(test_data, os.path.join(args.output, "twophase_test.h5"))

    print("Done.")
