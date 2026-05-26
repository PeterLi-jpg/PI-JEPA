#!/usr/bin/env python
"""
Cross-domain transfer runner — produces the paper's Table 3.

For every (pretrain_dataset, finetune_dataset) pair, this script:
  1. Pretrains a PI-JEPA encoder on the pretrain dataset (if no cached
     checkpoint exists)
  2. Fine-tunes on the finetune dataset's labeled corpus
  3. Evaluates on the finetune dataset's test split
  4. Aggregates over seeds with bootstrap CIs

Output: cross_domain_table.json — a 2D matrix of relative-L2 with CIs.

The paper claim being tested: a single PI-JEPA backbone PRETRAINED ON ANY
SUBSURFACE DATASET transfers to any OTHER subsurface dataset. The
original PI-JEPA paper asserted this in framing but only verified one
cross-domain row (Darcy → twophase).

Usage:
    python scripts/run_cross_domain.py \
        --datasets darcy_3d_synthetic \
        --output outputs_crossdomain/v1 \
        --n-seeds 3

For real cross-domain (Brev):
    python scripts/run_cross_domain.py \
        --datasets darcy_3d_synthetic ccsnet fno4co2 \
        --output outputs_crossdomain/full \
        --n-seeds 5 --epochs-pretrain 500 --epochs-finetune 100
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PI-JEPA"))

from eval.paper_metrics import bootstrap_ci_95


# Registry: name -> (pretrain config path, eval-data spec)
# eval-data spec: dict with keys understood by finetune_pijepa.py
DATASET_REGISTRY: Dict[str, Dict] = {
    "darcy_3d_synthetic": {
        "pretrain_config": "configs/darcy_3d.yaml",
        "finetune_dataset": "darcy_3d_pt",
        "train_pt": "data/darcy_3d/darcy3d_train.pt",
        "test_pt": "data/darcy_3d/darcy3d_test.pt",
    },
    "darcy_3d_mf": {
        "pretrain_config": "configs/darcy_3d_mf_smoke.yaml",
        "finetune_dataset": "darcy_3d_pt",
        "train_pt": "data/darcy_3d/darcy3d_train.pt",
        "test_pt": "data/darcy_3d/darcy3d_test.pt",
    },
    # The CCSNet / FNO4CO2 entries assume the matching finetune_pijepa
    # adapters get wired before this is run for real. For now we keep them
    # documented but unused if not present.
    "ccsnet": {
        "pretrain_config": "configs/ccsnet_3d_smoke.yaml",
        "finetune_dataset": "ccsnet",   # NOT YET supported by finetune_pijepa
        "train_pt": None,
        "test_pt": None,
    },
}


def run_subproc(cmd: List[str], log_label: str) -> Tuple[int, float]:
    print(f"[{log_label}] {' '.join(cmd[:6])} ...")
    t0 = time.time()
    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[{log_label}] FAILED ({dt:.1f}s)")
        print(res.stderr[-1500:])
    else:
        print(f"[{log_label}] done {dt:.1f}s")
    return res.returncode, dt


def pretrain_one(name: str, ds: Dict, out_root: str, seed: int) -> str:
    """Pretrain on dataset `name` for `seed`. Returns checkpoint path or None."""
    cfg_path = ds["pretrain_config"]
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("experiment", {})["seed"] = int(seed)
    seed_dir = os.path.join(out_root, "pretrains", name, f"seed{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    per_seed_cfg = os.path.join(seed_dir, "_pretrain_cfg.yaml")
    with open(per_seed_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    out_dir = os.path.join(seed_dir, "pretrain")
    ckpt = os.path.join(out_dir, "checkpoint_final.pt")
    if os.path.exists(ckpt):
        print(f"[pretrain {name} seed={seed}] using cached {ckpt}")
        return ckpt

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain.py"),
        "--config", per_seed_cfg,
        "--output", out_dir,
    ]
    rc, _ = run_subproc(cmd, f"pretrain {name} seed={seed}")
    return ckpt if rc == 0 and os.path.exists(ckpt) else None


def finetune_and_eval(
    src_name: str, dst_name: str,
    ckpt_path: str, src_ds: Dict, dst_ds: Dict,
    out_root: str, seed: int, n_labeled: int, epochs: int,
) -> Dict:
    """Fine-tune on dst, eval on dst. Returns metrics dict."""
    out_dir = os.path.join(
        out_root, "finetunes", f"{src_name}__to__{dst_name}", f"seed{seed}_n{n_labeled}"
    )
    os.makedirs(out_dir, exist_ok=True)

    finetune_dataset = dst_ds.get("finetune_dataset")
    if finetune_dataset != "darcy_3d_pt":
        # We currently only support darcy_3d_pt for finetune. Skip gracefully.
        return {
            "_failed": True,
            "_reason": f"finetune_pijepa.py does not yet support dataset={finetune_dataset}",
        }

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "finetune_pijepa.py"),
        "--pretrain-checkpoint", ckpt_path,
        "--pretrain-config", src_ds["pretrain_config"],
        "--dataset", finetune_dataset,
        "--train-pt", dst_ds["train_pt"],
        "--test-pt", dst_ds["test_pt"],
        "--n-labeled", str(n_labeled),
        "--epochs", str(epochs),
        "--seed", str(seed),
        "--output", out_dir,
    ]
    rc, _ = run_subproc(cmd, f"finetune {src_name}→{dst_name} seed={seed} n={n_labeled}")
    if rc != 0:
        return {"_failed": True, "_reason": "finetune subprocess failed"}

    json_path = os.path.join(out_dir, "pijepa_result.json")
    if not os.path.exists(json_path):
        return {"_failed": True, "_reason": "no pijepa_result.json"}
    with open(json_path, "r") as f:
        m = json.load(f)
    m["_failed"] = False
    return m


def aggregate_cell(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    arr = np.array(values, dtype=np.float64)
    mean, lo, hi = bootstrap_ci_95(arr)
    return {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(values)}


def main():
    ap = argparse.ArgumentParser(description="Cross-domain transfer runner")
    ap.add_argument("--datasets", nargs="+", required=True,
                    help="Subset of registered dataset names")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--n-labeled", type=int, default=32)
    ap.add_argument("--epochs-finetune", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    unknown = [d for d in args.datasets if d not in DATASET_REGISTRY]
    if unknown:
        print(f"Unknown datasets: {unknown}. Registered: {list(DATASET_REGISTRY)}")
        sys.exit(1)

    # 1. Pretrain each source dataset for each seed (cached)
    pretrains: Dict[Tuple[str, int], str] = {}
    for src_name in args.datasets:
        src_ds = DATASET_REGISTRY[src_name]
        for seed in range(args.seed_start, args.seed_start + args.n_seeds):
            ckpt = pretrain_one(src_name, src_ds, args.output, seed)
            pretrains[(src_name, seed)] = ckpt

    # 2. For every (src, dst) pair, finetune + eval
    rows: Dict[str, Dict[str, Dict]] = {}
    raw: Dict[str, Dict[str, List[Dict]]] = {}
    for src_name in args.datasets:
        rows[src_name] = {}
        raw[src_name] = {}
        for dst_name in args.datasets:
            seed_results: List[Dict] = []
            for seed in range(args.seed_start, args.seed_start + args.n_seeds):
                ckpt = pretrains.get((src_name, seed))
                if not ckpt:
                    seed_results.append({"_failed": True, "_reason": "pretrain failed"})
                    continue
                m = finetune_and_eval(
                    src_name, dst_name, ckpt,
                    DATASET_REGISTRY[src_name], DATASET_REGISTRY[dst_name],
                    args.output, seed, args.n_labeled, args.epochs_finetune,
                )
                seed_results.append(m)
            raw[src_name][dst_name] = seed_results
            l2_vals = [
                r["relative_l2_mean"] for r in seed_results
                if not r.get("_failed") and "relative_l2_mean" in r
            ]
            rows[src_name][dst_name] = aggregate_cell(l2_vals)

    out = {
        "datasets": args.datasets,
        "n_seeds": args.n_seeds,
        "n_labeled": args.n_labeled,
        "matrix": rows,
        "raw": raw,
    }
    out_path = os.path.join(args.output, "cross_domain_table.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote {out_path}")

    print("\nCross-domain matrix (rows=pretrain, cols=fine-tune; mean rel_L2 [95% CI]):")
    print(f"  {'pretrain':<22s} " + "".join(f"{d:<28s}" for d in args.datasets))
    for src in args.datasets:
        cells = []
        for dst in args.datasets:
            c = rows[src].get(dst, {})
            if c.get("n", 0) > 0:
                cells.append(f"{c['mean']:.3f}[{c['ci_low']:.3f},{c['ci_high']:.3f}]")
            else:
                cells.append("—")
        print(f"  {src:<22s} " + "".join(f"{c:<28s}" for c in cells))


if __name__ == "__main__":
    main()
