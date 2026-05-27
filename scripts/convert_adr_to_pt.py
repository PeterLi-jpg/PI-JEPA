#!/usr/bin/env python
"""
Convert the HDF5 output of `generate_adr_pe_da_sweep.py` into `.pt`
files in the same `{"x": ..., "y": ...}` layout as the other 3D
datasets (synthetic Darcy, CCSNet-converted, FNO4CO2-converted).

The PI-JEPA encoder + decoder + finetune pipeline expects a 5D
`(N, C, D, H, W)` tensor where the spatial dims feed the 3D Fourier
encoder. ADR is physically 2D + time; we treat time as the depth axis
so the resulting (N, 1, T, H, W) is structurally identical to a 3D
volume and the pipeline ingests it without any code changes.

Layout written:
    {
      "x":  (N, 1, T, H, W)   — initial condition u(x,y,t=0) broadcast
                                  across all T timesteps. Broadcasting
                                  matches the pattern used for static
                                  parameter fields in CCSNet (where x is
                                  permeability replicated across time).
      "y":  (N, 1, T, H, W)   — the full spatiotemporal trajectory.
      "pe": (N,)              — Pe value per sample (for conditioning
                                  studies or sweet-spot analysis).
      "da": (N,)              — Da value per sample.
    }

Usage:
    python scripts/convert_adr_to_pt.py \
        --input  data/pdebench_adr/pe_da_sweep.h5 \
        --out-dir data/pdebench_adr \
        --train-frac 0.8 --seed 42
"""

from __future__ import annotations

import argparse
import os

import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="ADR HDF5 → .pt converter")
    ap.add_argument("--input", required=True,
                    help="HDF5 from generate_adr_pe_da_sweep.py")
    ap.add_argument("--out-dir", required=True,
                    help="Where to write adr_train.pt and adr_test.pt")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for the train/test split shuffle.")
    ap.add_argument("--normalize", action="store_true",
                    help="Channel-wise z-score on the train split, "
                    "applied to test using train stats.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Reading {args.input} ...")
    xs, ys, pes, das = [], [], [], []
    with h5py.File(args.input, "r") as f:
        samples = f["samples"]
        keys = sorted(samples.keys())
        for k in keys:
            g = samples[k]
            traj = g["u"][:]  # (T, H, W) float32
            pes.append(float(g.attrs["pe"]))
            das.append(float(g.attrs["da"]))
            ic = traj[0]  # (H, W) — initial condition
            T = traj.shape[0]
            # Broadcast IC across T to make a 5D-compatible input volume.
            x = np.broadcast_to(ic[None, :, :], (T, *ic.shape)).copy()
            xs.append(x)
            ys.append(traj)

    x_arr = np.stack(xs, axis=0)[:, None, ...]   # (N, 1, T, H, W)
    y_arr = np.stack(ys, axis=0)[:, None, ...]
    pe_arr = np.array(pes, dtype=np.float32)
    da_arr = np.array(das, dtype=np.float32)
    N = x_arr.shape[0]
    print(f"  loaded {N} samples, x shape {x_arr.shape}, y shape {y_arr.shape}")
    print(f"  Pe range [{pe_arr.min():.3g}, {pe_arr.max():.3g}]")
    print(f"  Da range [{da_arr.min():.3g}, {da_arr.max():.3g}]")

    # Deterministic train/test split.
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_train = int(round(args.train_frac * N))
    tr_idx, te_idx = perm[:n_train], perm[n_train:]

    x_tr, y_tr = x_arr[tr_idx], y_arr[tr_idx]
    x_te, y_te = x_arr[te_idx], y_arr[te_idx]
    pe_tr, pe_te = pe_arr[tr_idx], pe_arr[te_idx]
    da_tr, da_te = da_arr[tr_idx], da_arr[te_idx]

    if args.normalize:
        x_mean = x_tr.mean()
        x_std = x_tr.std() + 1e-8
        y_mean = y_tr.mean()
        y_std = y_tr.std() + 1e-8
        x_tr, x_te = (x_tr - x_mean) / x_std, (x_te - x_mean) / x_std
        y_tr, y_te = (y_tr - y_mean) / y_std, (y_te - y_mean) / y_std
        print(f"  normalized: x ~ N({x_mean:.3g}, {x_std:.3g}), "
              f"y ~ N({y_mean:.3g}, {y_std:.3g})")

    train_pt = os.path.join(args.out_dir, "adr_train.pt")
    test_pt = os.path.join(args.out_dir, "adr_test.pt")
    torch.save({
        "x": torch.from_numpy(x_tr).float(),
        "y": torch.from_numpy(y_tr).float(),
        "pe": torch.from_numpy(pe_tr),
        "da": torch.from_numpy(da_tr),
    }, train_pt)
    torch.save({
        "x": torch.from_numpy(x_te).float(),
        "y": torch.from_numpy(y_te).float(),
        "pe": torch.from_numpy(pe_te),
        "da": torch.from_numpy(da_te),
    }, test_pt)

    sz_tr = os.path.getsize(train_pt) / (1024 * 1024)
    sz_te = os.path.getsize(test_pt) / (1024 * 1024)
    print(f"\nWrote {train_pt} ({sz_tr:.1f} MB, n={len(tr_idx)})")
    print(f"Wrote {test_pt}  ({sz_te:.1f} MB, n={len(te_idx)})")
    print("\nCompatible with the existing pipeline — same {x, y} shape "
          "convention as data/darcy_3d/darcy3d_*.pt. The encoder treats "
          "T as the depth axis (D), so all 3D paths work unchanged.")


if __name__ == "__main__":
    main()
