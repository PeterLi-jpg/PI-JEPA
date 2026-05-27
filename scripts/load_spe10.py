#!/usr/bin/env python
"""
Parse SPE10 Model 2 permeability + porosity into compact numpy arrays.

SPE10 ships as two ASCII files:
    spe_perm.dat    — kx, ky, kz triples for 60×220×85 = 1,122,000 cells
                      (3 floats per cell, space-separated, blocks of three)
    spe_phi.dat     — porosity values for the same grid

This loader reads both and writes:
    data/spe10/spe10_arrays.npz with:
        perm_x : (60, 220, 85)  in mD
        perm_y : (60, 220, 85)  in mD
        perm_z : (60, 220, 85)  in mD
        phi    : (60, 220, 85)  porosity ∈ (0, 1)

So downstream code never has to re-parse the slow ASCII files.

Usage:
    python scripts/load_spe10.py \
        --spe10-dir data/spe10 \
        --out data/spe10/spe10_arrays.npz
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np


SPE10_NX, SPE10_NY, SPE10_NZ = 60, 220, 85
SPE10_N = SPE10_NX * SPE10_NY * SPE10_NZ


def _read_floats(path: str, expected_count: int) -> np.ndarray:
    """Read whitespace-separated floats from an ASCII file."""
    print(f"  Reading {path} ({expected_count} values expected)...")
    t0 = time.time()
    vals = np.fromfile(path, sep=" ", dtype=np.float64)
    if vals.size != expected_count:
        raise ValueError(
            f"{path}: expected {expected_count} floats, got {vals.size}"
        )
    print(f"    parsed in {time.time() - t0:.1f}s")
    return vals


def main():
    ap = argparse.ArgumentParser(description="Parse SPE10 .dat files into numpy")
    ap.add_argument("--spe10-dir", default="data/spe10")
    ap.add_argument("--out", default="data/spe10/spe10_arrays.npz")
    args = ap.parse_args()

    perm_path = os.path.join(args.spe10_dir, "spe_perm.dat")
    phi_path = os.path.join(args.spe10_dir, "spe_phi.dat")
    if not os.path.exists(perm_path):
        raise SystemExit(f"Missing {perm_path}. Run download first.")
    if not os.path.exists(phi_path):
        raise SystemExit(f"Missing {phi_path}. Run download first.")

    # Permeability: 3 floats per cell (kx, ky, kz), grouped sequentially.
    perm_flat = _read_floats(perm_path, 3 * SPE10_N)
    perm = perm_flat.reshape(3, SPE10_N).reshape(3, SPE10_NX, SPE10_NY, SPE10_NZ)
    perm_x, perm_y, perm_z = perm[0], perm[1], perm[2]

    phi_flat = _read_floats(phi_path, SPE10_N)
    phi = phi_flat.reshape(SPE10_NX, SPE10_NY, SPE10_NZ)

    print(f"  perm_x range [mD]: {perm_x.min():.3g} .. {perm_x.max():.3g}")
    print(f"  phi range:         {phi.min():.3g} .. {phi.max():.3g}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        perm_x=perm_x.astype(np.float32),
        perm_y=perm_y.astype(np.float32),
        perm_z=perm_z.astype(np.float32),
        phi=phi.astype(np.float32),
    )
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"Wrote {args.out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
