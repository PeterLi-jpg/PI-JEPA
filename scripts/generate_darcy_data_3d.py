#!/usr/bin/env python
"""
Generate a small synthetic 3D Darcy dataset for smoke-testing PI-JEPA 3D.

For each sample:
  K(x, y, z) = exp( GRF(x, y, z) )  with isotropic Gaussian random field
  Solve  -∇·(K ∇p) = q  on the unit cube with Dirichlet p=0 on boundary
  q       = constant volumetric source (centered Gaussian "well")

Solver: matrix-free Jacobi-PCG via SciPy on a flattened 7-point stencil
        (3D analogue of the 5-point stencil used by generate_darcy_data.py).

Output: torch .pt with x = (N, 1, D, H, W) permeability and y = (N, 1, D, H, W) pressure.
"""

import argparse
import math
import os
from typing import Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import cg


def gaussian_random_field_3d(
    resolution: int, length_scale: float, variance: float, rng: np.random.Generator
) -> np.ndarray:
    """Sample a 3D isotropic Gaussian random field via spectral synthesis.

    Returns log K (zero-mean Gaussian) of shape (N, N, N).
    """
    N = resolution
    # Frequency grids
    k1 = np.fft.fftfreq(N) * N
    k2 = np.fft.fftfreq(N) * N
    k3 = np.fft.fftfreq(N) * N
    kx, ky, kz = np.meshgrid(k1, k2, k3, indexing="ij")
    k_sq = kx ** 2 + ky ** 2 + kz ** 2
    # Squared-exponential spectrum (isotropic, length scale ℓ)
    spectrum = np.exp(-(k_sq) * (length_scale ** 2))
    # White noise in Fourier space with proper Hermitian symmetry via real input
    noise = rng.standard_normal((N, N, N))
    noise_ft = np.fft.fftn(noise)
    field_ft = noise_ft * np.sqrt(spectrum + 1e-12)
    field = np.real(np.fft.ifftn(field_ft))
    # Normalize to unit variance, then scale to requested variance
    field = (field - field.mean()) / (field.std() + 1e-12)
    field = field * math.sqrt(variance)
    return field


def build_poisson_matrix_3d(K: np.ndarray) -> sparse.csr_matrix:
    """7-point discretization of -∇·(K∇p) with Dirichlet BC (p=0 on boundary).

    Interior dofs only: (N-2)^3 unknowns. Harmonic-mean K at faces.
    """
    N = K.shape[0]
    n = N - 2  # interior dim
    n_int = n ** 3

    def idx(i, j, k):
        # Map interior coordinates (1..N-2) -> linear index in [0, n^3)
        return ((i - 1) * n + (j - 1)) * n + (k - 1)

    rows, cols, vals = [], [], []

    def harm(a, b):
        return 2.0 * a * b / (a + b + 1e-12)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            for k in range(1, N - 1):
                p = idx(i, j, k)
                diag = 0.0
                # +x neighbor (k+1)
                Kf = harm(K[i, j, k], K[i, j, k + 1])
                if k + 1 < N - 1:
                    rows.append(p); cols.append(idx(i, j, k + 1)); vals.append(-Kf)
                diag += Kf
                # -x neighbor (k-1)
                Kf = harm(K[i, j, k], K[i, j, k - 1])
                if k - 1 > 0:
                    rows.append(p); cols.append(idx(i, j, k - 1)); vals.append(-Kf)
                diag += Kf
                # +y neighbor (j+1)
                Kf = harm(K[i, j, k], K[i, j + 1, k])
                if j + 1 < N - 1:
                    rows.append(p); cols.append(idx(i, j + 1, k)); vals.append(-Kf)
                diag += Kf
                # -y neighbor (j-1)
                Kf = harm(K[i, j, k], K[i, j - 1, k])
                if j - 1 > 0:
                    rows.append(p); cols.append(idx(i, j - 1, k)); vals.append(-Kf)
                diag += Kf
                # +z neighbor (i+1)
                Kf = harm(K[i, j, k], K[i + 1, j, k])
                if i + 1 < N - 1:
                    rows.append(p); cols.append(idx(i + 1, j, k)); vals.append(-Kf)
                diag += Kf
                # -z neighbor (i-1)
                Kf = harm(K[i, j, k], K[i - 1, j, k])
                if i - 1 > 0:
                    rows.append(p); cols.append(idx(i - 1, j, k)); vals.append(-Kf)
                diag += Kf

                rows.append(p); cols.append(p); vals.append(diag)

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))
    return A


def solve_one_sample(
    resolution: int,
    length_scale: float,
    variance: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (K, p) of shape (N, N, N)."""
    N = resolution
    logK = gaussian_random_field_3d(N, length_scale, variance, rng)
    K = np.exp(logK)  # always positive

    A = build_poisson_matrix_3d(K)

    # Volumetric source: smooth Gaussian "injection" centered in the cube.
    coords = (np.arange(N) - (N - 1) / 2.0) / N
    zz, yy, xx = np.meshgrid(coords, coords, coords, indexing="ij")
    width = 0.1
    q = np.exp(-(xx ** 2 + yy ** 2 + zz ** 2) / (2 * width ** 2))
    q = q[1:-1, 1:-1, 1:-1].reshape(-1)

    p_int, info = cg(A, q, rtol=1e-6, maxiter=2000)
    if info != 0:
        # Not fatal for smoke test; still return whatever the solver got.
        pass

    p = np.zeros((N, N, N), dtype=np.float32)
    p[1:-1, 1:-1, 1:-1] = p_int.reshape(N - 2, N - 2, N - 2)
    return K.astype(np.float32), p


def main():
    ap = argparse.ArgumentParser(description="Generate small 3D Darcy dataset for smoke test")
    ap.add_argument("--n-train", type=int, default=64)
    ap.add_argument("--n-test", type=int, default=8)
    ap.add_argument("--resolution", type=int, default=32, help="Cubic grid side length")
    ap.add_argument("--length-scale", type=float, default=0.15)
    ap.add_argument("--variance", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="data/darcy_3d")
    ap.add_argument("--normalize", action="store_true", default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    def make_split(n: int) -> dict:
        Ks, ps = [], []
        for i in range(n):
            K, p = solve_one_sample(args.resolution, args.length_scale, args.variance, rng)
            Ks.append(K)
            ps.append(p)
            if (i + 1) % max(1, n // 10) == 0:
                print(f"  sample {i+1}/{n}")
        K_arr = np.stack(Ks, axis=0)[:, None, ...]  # (N, 1, D, H, W)
        p_arr = np.stack(ps, axis=0)[:, None, ...]
        return {"x": torch.from_numpy(K_arr), "y": torch.from_numpy(p_arr)}

    print(f"Generating train ({args.n_train}) at {args.resolution}^3 ...")
    train = make_split(args.n_train)

    print(f"Generating test ({args.n_test}) at {args.resolution}^3 ...")
    test = make_split(args.n_test)

    if args.normalize:
        x_mean = train["x"].mean()
        x_std = train["x"].std() + 1e-8
        y_mean = train["y"].mean()
        y_std = train["y"].std() + 1e-8
        for split in (train, test):
            split["x"] = (split["x"] - x_mean) / x_std
            split["y"] = (split["y"] - y_mean) / y_std

    train_path = os.path.join(args.out_dir, "darcy3d_train.pt")
    test_path = os.path.join(args.out_dir, "darcy3d_test.pt")
    torch.save(train, train_path)
    torch.save(test, test_path)
    print(f"Saved {train_path} shapes: x={tuple(train['x'].shape)}, y={tuple(train['y'].shape)}")
    print(f"Saved {test_path}  shapes: x={tuple(test['x'].shape)}, y={tuple(test['y'].shape)}")


if __name__ == "__main__":
    main()
