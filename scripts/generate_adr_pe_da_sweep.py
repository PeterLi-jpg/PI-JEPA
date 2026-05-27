#!/usr/bin/env python
"""
Generate a 2D advection-diffusion-reaction (ADR) dataset with a sweep
across Péclet (Pe) and Damköhler (Da) non-dimensional numbers.

Reviewer-requested as the mechanistic ablation backbone: the resubmission
needs to show that the model handles a regime sweep, not just one set of
PDE coefficients. The original PDEBench 2D `diff_react` sim has no
advection term, so "Pe sweep" doesn't apply to it — this generator adds
an explicit advection field and varies the dimensionless groups.

PDE solved (single species u(x, y, t) on the unit square):
    du/dt + V * (cos θ, sin θ) · ∇u = D * Δu + k * u * (1 - u)
where:
    Pe := V * L / D       (advection / diffusion strength)
    Da := k * L^2 / D     (reaction / diffusion strength)

We fix L = 1, D = 1 and pick (V, k) per cell of the sweep so the
resulting (Pe, Da) lands on a chosen grid. Initial condition is a
sum of K randomly placed Gaussian bumps; angle θ is randomized per
sample so the encoder doesn't learn a fixed flow direction.

Solver: 2nd-order finite-difference Laplacian + upwind advection +
explicit Euler. CFL-limited dt; safe for the Pe/Da ranges used here.

Output layout (HDF5):
    sweep_grid              : (n_pe, n_da) — Pe/Da values used
    n_samples_per_cell      : scalar
    grid_resolution         : scalar
    samples
      └── pe{i}_da{j}_seed{s}
              ├── u    : (T, H, W) time-series of u
              ├── pe   : scalar Pe value
              ├── da   : scalar Da value
              ├── V    : advection magnitude
              ├── k    : reaction rate
              └── theta: advection angle

Typical use:
    python scripts/generate_adr_pe_da_sweep.py \
        --n-pe 5 --n-da 5 --n-per-cell 8 \
        --grid 64 --t-final 1.0 --n-t 16 \
        --output data/pdebench_adr/pe_da_sweep.h5
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Tuple

import h5py
import numpy as np


def _initial_field(rng: np.random.Generator, H: int, W: int,
                   n_bumps: int = 5) -> np.ndarray:
    """Sum of K Gaussian bumps with random center/scale/amplitude.

    Domain [0, 1] x [0, 1]. Field clipped to [0, 1] to match the
    reaction nonlinearity u(1-u) being well-defined there.
    """
    x = np.linspace(0.0, 1.0, W)
    y = np.linspace(0.0, 1.0, H)
    X, Y = np.meshgrid(x, y, indexing="ij")
    u = np.zeros_like(X)
    for _ in range(n_bumps):
        cx, cy = rng.uniform(0.15, 0.85, size=2)
        sx, sy = rng.uniform(0.05, 0.18, size=2)
        amp = rng.uniform(0.3, 1.0)
        u += amp * np.exp(-((X - cx) ** 2 / (2 * sx**2)
                            + (Y - cy) ** 2 / (2 * sy**2)))
    return np.clip(u, 0.0, 1.0)


def _step_adr(u: np.ndarray, V: float, theta: float, D: float, k: float,
              dx: float, dy: float, dt: float) -> np.ndarray:
    """One explicit-Euler step of u_t + V·∇u = D Δu + k u (1-u).

    Upwind for advection (1st-order, stable for Pe>>1) + 2nd-order
    central FD for Laplacian. Periodic BC via np.roll.
    """
    # Advection velocity components
    vx = V * np.cos(theta)
    vy = V * np.sin(theta)

    # Upwind advection: pick the upwind neighbor based on sign of v
    if vx >= 0:
        adv_x = vx * (u - np.roll(u, 1, axis=1)) / dx
    else:
        adv_x = vx * (np.roll(u, -1, axis=1) - u) / dx
    if vy >= 0:
        adv_y = vy * (u - np.roll(u, 1, axis=0)) / dy
    else:
        adv_y = vy * (np.roll(u, -1, axis=0) - u) / dy

    # Laplacian (5-point, periodic BC)
    lap = (
        (np.roll(u, 1, axis=1) - 2 * u + np.roll(u, -1, axis=1)) / dx**2
        + (np.roll(u, 1, axis=0) - 2 * u + np.roll(u, -1, axis=0)) / dy**2
    )

    # Reaction (logistic / Fisher–KPP) — only well-defined on [0, 1].
    rxn = k * u * (1.0 - u)

    u_next = u + dt * (-adv_x - adv_y + D * lap + rxn)
    # Physical clamp + NaN guard. Fisher-KPP solutions stay in [0, 1] by
    # construction; the FD step can drift slightly outside or blow up at
    # extreme Pe/Da combinations. Clamping keeps the integrator stable
    # without distorting solution quality in the well-resolved regime.
    return np.clip(np.nan_to_num(u_next, nan=0.0, posinf=1.0, neginf=0.0),
                   0.0, 1.0)


def _simulate_one(pe: float, da: float, seed: int, H: int, W: int,
                  t_final: float, n_t: int, L: float = 1.0,
                  D: float = 1.0) -> Tuple[np.ndarray, dict]:
    """Run one ADR simulation at given (Pe, Da). Returns (T, H, W) trajectory."""
    rng = np.random.default_rng(seed)
    V = pe * D / L
    k = da * D / (L * L)
    theta = float(rng.uniform(0.0, 2.0 * np.pi))

    dx = L / W
    dy = L / H

    # CFL-safe dt: take the strictest of (advection, diffusion, reaction).
    dt_adv = 0.5 * min(dx, dy) / max(V, 1e-12)
    dt_diff = 0.25 * min(dx, dy) ** 2 / max(D, 1e-12)
    dt_rxn = 0.25 / max(k, 1e-12)
    dt = float(min(dt_adv, dt_diff, dt_rxn, t_final / n_t / 4))

    n_steps_total = int(np.ceil(t_final / dt))
    # Snapshot times — uniformly spaced including the final state.
    snapshot_steps = np.linspace(0, n_steps_total, n_t, dtype=int)
    snapshot_set = set(int(s) for s in snapshot_steps)

    u = _initial_field(rng, H, W)
    frames = [u.copy()]  # t=0 snapshot
    for step in range(1, n_steps_total + 1):
        u = _step_adr(u, V, theta, D, k, dx, dy, dt)
        if step in snapshot_set and len(frames) < n_t:
            frames.append(u.copy())
    # Ensure we got exactly n_t frames (pad with final state if early-CFL kicked us short)
    while len(frames) < n_t:
        frames.append(u.copy())
    arr = np.stack(frames[:n_t], axis=0).astype(np.float32)
    meta = {"pe": float(pe), "da": float(da), "V": float(V),
            "k": float(k), "theta": theta, "L": float(L), "D": float(D),
            "dt": float(dt), "n_steps_total": int(n_steps_total)}
    return arr, meta


def main():
    ap = argparse.ArgumentParser(description="ADR Pe/Da sweep generator")
    ap.add_argument("--n-pe", type=int, default=5,
                    help="Number of Pe values to sweep")
    ap.add_argument("--n-da", type=int, default=5,
                    help="Number of Da values to sweep")
    ap.add_argument("--pe-range", type=float, nargs=2, default=[0.1, 100.0],
                    help="Pe log-spaced range (min, max)")
    ap.add_argument("--da-range", type=float, nargs=2, default=[0.1, 100.0],
                    help="Da log-spaced range (min, max)")
    ap.add_argument("--n-per-cell", type=int, default=8,
                    help="Samples per (Pe, Da) cell (different IC seeds)")
    ap.add_argument("--grid", type=int, default=64, help="Spatial resolution")
    ap.add_argument("--t-final", type=float, default=1.0)
    ap.add_argument("--n-t", type=int, default=16,
                    help="Time snapshots per trajectory")
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--output", required=True, help="HDF5 output path")
    args = ap.parse_args()

    pe_vals = np.logspace(np.log10(args.pe_range[0]),
                          np.log10(args.pe_range[1]), args.n_pe)
    da_vals = np.logspace(np.log10(args.da_range[0]),
                          np.log10(args.da_range[1]), args.n_da)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    total_cells = args.n_pe * args.n_da
    total_samples = total_cells * args.n_per_cell
    print(f"Generating {total_samples} ADR samples over {total_cells} "
          f"(Pe, Da) cells × {args.n_per_cell} seeds each")
    print(f"  Pe range: {pe_vals[0]:.3g} .. {pe_vals[-1]:.3g}")
    print(f"  Da range: {da_vals[0]:.3g} .. {da_vals[-1]:.3g}")
    print(f"  Grid: {args.grid}², T={args.t_final}, n_t={args.n_t}")

    t_start = time.time()
    with h5py.File(args.output, "w") as f:
        f.create_dataset("pe_vals", data=pe_vals)
        f.create_dataset("da_vals", data=da_vals)
        f.attrs["n_samples_per_cell"] = args.n_per_cell
        f.attrs["grid_resolution"] = args.grid
        f.attrs["t_final"] = args.t_final
        f.attrs["n_t"] = args.n_t
        samples_grp = f.create_group("samples")

        seed_counter = args.seed_start
        for i, pe in enumerate(pe_vals):
            for j, da in enumerate(da_vals):
                for s in range(args.n_per_cell):
                    arr, meta = _simulate_one(
                        pe=float(pe), da=float(da), seed=seed_counter,
                        H=args.grid, W=args.grid,
                        t_final=args.t_final, n_t=args.n_t,
                    )
                    name = f"pe{i}_da{j}_seed{s}"
                    g = samples_grp.create_group(name)
                    g.create_dataset("u", data=arr, compression="gzip",
                                     compression_opts=4)
                    for mk, mv in meta.items():
                        g.attrs[mk] = mv
                    seed_counter += 1
                    if seed_counter % 50 == 0:
                        elapsed = time.time() - t_start
                        rate = seed_counter / elapsed
                        eta = (total_samples - seed_counter) / max(rate, 1e-9)
                        print(f"  [{seed_counter}/{total_samples}] "
                              f"{rate:.1f} sample/s, ETA {eta/60:.1f}min")

    dt = time.time() - t_start
    size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\nDone in {dt/60:.2f} min. Wrote {args.output} ({size_mb:.1f} MB)")
    print(f"  {total_samples} samples × shape ({args.n_t}, {args.grid}, "
          f"{args.grid}) under /samples/pe{{i}}_da{{j}}_seed{{s}}/u")


if __name__ == "__main__":
    main()
