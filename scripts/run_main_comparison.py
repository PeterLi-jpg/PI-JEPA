#!/usr/bin/env python
"""
Main paper-table runner: PI-JEPA (pretrained + fine-tuned) vs supervised baselines.

For each (method, n_labeled, seed) cell, trains and evaluates and reports
relative-L2 with bootstrap 95% CIs. Produces main_comparison_table.json
that maps cleanly into the paper's headline results table.

Methods compared:
  - pi_jepa             (pretrained + fine-tuned)
  - fno3d               (supervised)
  - ufno3d              (supervised, CCS-specific)
  - pino3d              (supervised + physics residual)

Sample-efficiency sweep: N_labeled ∈ {16, 32, 64, 128} (configurable).
Multi-seed bootstrap CI for every cell.

Currently supports synthetic 3D Darcy (`darcy_3d_pt`). Adding real datasets
is one entry in DATASET_SPECS.

Usage:
    python scripts/run_main_comparison.py \
        --output outputs_main/v1 \
        --dataset darcy_3d_synthetic \
        --n-labeled 16 32 64 \
        --n-seeds 3 --epochs 10

For a full Brev run:
    python scripts/run_main_comparison.py \
        --output outputs_main/full \
        --dataset darcy_3d_synthetic \
        --n-labeled 16 32 64 128 256 \
        --n-seeds 5 --epochs 100
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PI-JEPA"))

from eval.paper_metrics import bootstrap_ci_95


DATASET_SPECS = {
    "darcy_3d_synthetic": {
        "pretrain_config": "configs/darcy_3d.yaml",
        "finetune_dataset": "darcy_3d_pt",
        "train_pt": "data/darcy_3d/darcy3d_train.pt",
        "test_pt":  "data/darcy_3d/darcy3d_test.pt",
        "modes": [4, 8, 8],     # baseline FNO/U-FNO/PINO modes
    },
    "darcy_3d_mf": {
        "pretrain_config": "configs/darcy_3d_mf_smoke.yaml",
        "finetune_dataset": "darcy_3d_pt",
        "train_pt": "data/darcy_3d/darcy3d_train.pt",
        "test_pt":  "data/darcy_3d/darcy3d_test.pt",
        "modes": [4, 8, 8],
    },
}


def run_subproc(cmd: List[str], label: str) -> Tuple[int, float]:
    print(f"[{label}] {' '.join(cmd[:6])} ...")
    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[{label}] FAILED ({dt:.1f}s)")
        print(res.stderr[-1500:])
    else:
        print(f"[{label}] done {dt:.1f}s")
    return res.returncode, dt


def pretrain_one(ds_spec: Dict, out_root: str, seed: int) -> str:
    cfg_path = ds_spec["pretrain_config"]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("experiment", {})["seed"] = int(seed)
    seed_dir = os.path.join(out_root, "pretrains", f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    per_seed_cfg = os.path.join(seed_dir, "_pretrain_cfg.yaml")
    with open(per_seed_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    out_dir = os.path.join(seed_dir, "pretrain")
    ckpt = os.path.join(out_dir, "checkpoint_final.pt")
    if os.path.exists(ckpt):
        return ckpt
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain.py"),
        "--config", per_seed_cfg, "--output", out_dir,
    ]
    rc, _ = run_subproc(cmd, f"pretrain seed={seed}")
    return ckpt if rc == 0 and os.path.exists(ckpt) else None


def finetune_pijepa_cell(ds_spec, ckpt, out_root, seed, n_l, epochs) -> Dict:
    out_dir = os.path.join(out_root, "pijepa", f"seed{seed}_n{n_l}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_pijepa.py"),
        "--pretrain-checkpoint", ckpt,
        "--pretrain-config", ds_spec["pretrain_config"],
        "--dataset", ds_spec["finetune_dataset"],
        "--train-pt", ds_spec["train_pt"],
        "--test-pt", ds_spec["test_pt"],
        "--n-labeled", str(n_l),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--output", out_dir,
    ]
    rc, _ = run_subproc(cmd, f"PI-JEPA n={n_l} seed={seed}")
    if rc != 0:
        return {"_failed": True}
    p = os.path.join(out_dir, "pijepa_result.json")
    return json.load(open(p)) if os.path.exists(p) else {"_failed": True}


def baseline_cell(baseline, ds_spec, out_root, seed, n_l, epochs, modes) -> Dict:
    out_dir = os.path.join(out_root, baseline, f"seed{seed}_n{n_l}")
    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_baseline.py"),
        "--baseline", baseline,
        "--dataset", ds_spec["finetune_dataset"],
        "--train-pt", ds_spec["train_pt"],
        "--test-pt", ds_spec["test_pt"],
        "--n-labeled", str(n_l),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--hidden-channels", "16", "--n-blocks", "2",
        "--modes", str(modes[0]), str(modes[1]), str(modes[2]),
        "--output", out_dir,
    ]
    rc, _ = run_subproc(cmd, f"{baseline} n={n_l} seed={seed}")
    if rc != 0:
        return {"_failed": True}
    p = os.path.join(out_dir, "baseline_result.json")
    return json.load(open(p)) if os.path.exists(p) else {"_failed": True}


def main():
    ap = argparse.ArgumentParser(description="PI-JEPA vs baselines head-to-head table")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", choices=list(DATASET_SPECS), default="darcy_3d_synthetic")
    ap.add_argument("--n-labeled", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--baselines", nargs="+", default=["fno3d", "ufno3d", "pino3d"])
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    ds_spec = DATASET_SPECS[args.dataset]
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    # 1. Pretrain per seed (cached)
    ckpts = {s: pretrain_one(ds_spec, args.output, s) for s in seeds}

    # 2. PI-JEPA fine-tune × N_labeled × seed
    raw: Dict[str, Dict[int, Dict[int, Dict]]] = {}
    raw["pi_jepa"] = {}
    for n_l in args.n_labeled:
        raw["pi_jepa"][n_l] = {}
        for s in seeds:
            if ckpts[s] is None:
                raw["pi_jepa"][n_l][s] = {"_failed": True}
                continue
            raw["pi_jepa"][n_l][s] = finetune_pijepa_cell(
                ds_spec, ckpts[s], args.output, s, n_l, args.epochs
            )

    # 3. Baselines × N_labeled × seed
    for b in args.baselines:
        raw[b] = {}
        for n_l in args.n_labeled:
            raw[b][n_l] = {}
            for s in seeds:
                raw[b][n_l][s] = baseline_cell(
                    b, ds_spec, args.output, s, n_l, args.epochs, ds_spec["modes"]
                )

    # 4. Aggregate (method, n_labeled) → bootstrap CI on relative_l2_mean
    table = {}
    for method, by_n in raw.items():
        table[method] = {}
        for n_l, by_seed in by_n.items():
            vals = [
                m["relative_l2_mean"] for m in by_seed.values()
                if not m.get("_failed") and "relative_l2_mean" in m
            ]
            if vals:
                arr = np.array(vals)
                mean, lo, hi = bootstrap_ci_95(arr)
                table[method][n_l] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n_seeds": len(vals)}
            else:
                table[method][n_l] = {"_failed": True}

    out = {
        "dataset": args.dataset,
        "seeds": seeds,
        "n_labeled_sweep": args.n_labeled,
        "baselines": args.baselines,
        "table": table,
        "raw": {k: {str(n): {str(s): m for s, m in by_seed.items()} for n, by_seed in by_n.items()}
                for k, by_n in raw.items()},
    }
    out_path = os.path.join(args.output, "main_comparison_table.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    # Print a quick text table
    print("\nMain comparison (relative L2, mean [95% CI], lower=better):")
    header = f"  {'method':<14s} " + "".join(f"N_l={n:<8d}" for n in args.n_labeled)
    print(header)
    print("  " + "-" * (15 + 12 * len(args.n_labeled)))
    for method in ["pi_jepa"] + list(args.baselines):
        cells = []
        for n_l in args.n_labeled:
            c = table.get(method, {}).get(n_l, {})
            if c.get("_failed") or c.get("n_seeds", 0) == 0:
                cells.append("FAILED  ")
            else:
                cells.append(f"{c['mean']:.3f}   ")
        print(f"  {method:<14s} " + "".join(cells))


if __name__ == "__main__":
    main()
