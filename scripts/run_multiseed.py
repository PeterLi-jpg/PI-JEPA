#!/usr/bin/env python
"""
Multi-seed orchestrator for PI-JEPA experiments.

Runs the full pretrain -> finetune -> eval pipeline across N seeds and
aggregates with bootstrap 95% CIs, writing a paper-ready JSON output.

Usage:
    python scripts/run_multiseed.py \
        --pretrain-config configs/darcy_3d_mf_smoke.yaml \
        --finetune-config configs/darcy_3d_finetune.yaml \
        --eval-data data/darcy_3d/darcy3d_test.pt \
        --n-seeds 5 \
        --n-labeled 100 \
        --output outputs_multiseed/run1

For now this is a SCAFFOLD: it runs pretraining for each seed (using the
existing pretrain.py CLI) and writes an aggregation JSON. Finetuning and
the full eval cycle integrate as additional CLI invocations as those
scripts mature.

Output JSON schema:
    {
      "pretrain_config": "...",
      "n_seeds": 5,
      "n_labeled": 100,
      "seeds": [42, 43, 44, 45, 46],
      "per_seed": {
        42: {"pretrain_loss": 0.054, "pretrain_jepa": 0.005, ...},
        43: {...},
        ...
      },
      "aggregated": {
        "pretrain_loss": {"mean": 0.054, "ci_low": 0.048, "ci_high": 0.061},
        ...
      }
    }
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

# Reuse the bootstrap helper from paper_metrics
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))
from eval.paper_metrics import bootstrap_ci_95


def run_pretrain(pretrain_config: str, output_dir: str, seed: int) -> Dict[str, float]:
    """Spawn scripts/pretrain.py with a seed override, parse the final checkpoint metrics."""
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = env.get("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # We rely on the config's experiment.seed field. Patch via a tempfile.
    # Simpler: read the config, patch the seed, write to a per-seed file.
    import yaml
    with open(pretrain_config, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("experiment", {})["seed"] = int(seed)
    per_seed_cfg = os.path.join(output_dir, f"_pretrain_cfg_seed{seed}.yaml")
    os.makedirs(output_dir, exist_ok=True)
    with open(per_seed_cfg, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    pretrain_out = os.path.join(output_dir, f"seed{seed}", "pretrain")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain.py"),
        "--config", per_seed_cfg,
        "--output", pretrain_out,
    ]
    print(f"[seed {seed}] launching pretrain ...")
    t0 = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"[seed {seed}] PRETRAIN FAILED in {dt:.1f}s")
        print(result.stdout[-2000:])
        print(result.stderr[-2000:])
        return {"_failed": True, "_dt_seconds": dt}

    # Parse the final checkpoint metrics
    ckpt_path = os.path.join(pretrain_out, "checkpoint_final.pt")
    if not os.path.exists(ckpt_path):
        return {"_failed": True, "_dt_seconds": dt, "_reason": "no checkpoint"}
    blob = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    metrics = blob.get("metrics", {})
    metrics["_dt_seconds"] = dt
    metrics["_failed"] = False
    metrics["_ckpt_path"] = ckpt_path
    print(f"[seed {seed}] pretrain done in {dt:.1f}s — {metrics}")
    return metrics


def aggregate_seeds(per_seed: Dict[int, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """For each numeric metric present in every seed result, compute bootstrap CI."""
    keys = set()
    for s in per_seed.values():
        for k, v in s.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float)):
                keys.add(k)

    agg = {}
    for k in sorted(keys):
        values = []
        for seed, m in per_seed.items():
            if not m.get("_failed") and k in m and isinstance(m[k], (int, float)):
                values.append(float(m[k]))
        if not values:
            continue
        mean, lo, hi = bootstrap_ci_95(np.array(values), n_boot=2000)
        agg[k] = {
            "mean": mean,
            "ci_low": lo,
            "ci_high": hi,
            "n": len(values),
            "raw": values,
        }
    return agg


def main():
    ap = argparse.ArgumentParser(description="Multi-seed PI-JEPA experiment orchestrator")
    ap.add_argument("--pretrain-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--n-labeled", type=int, default=None,
                    help="Recorded in the output JSON; finetune integration TBD")
    ap.add_argument("--note", type=str, default="",
                    help="Free-text note saved into the output JSON")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    per_seed: Dict[int, Dict[str, float]] = {}
    for seed in seeds:
        per_seed[seed] = run_pretrain(args.pretrain_config, args.output, seed)

    aggregated = aggregate_seeds(per_seed)

    out_json = {
        "pretrain_config": args.pretrain_config,
        "n_seeds": args.n_seeds,
        "seeds": seeds,
        "n_labeled": args.n_labeled,
        "note": args.note,
        "per_seed": {str(s): m for s, m in per_seed.items()},
        "aggregated": aggregated,
    }
    out_path = os.path.join(args.output, "multiseed_results.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nWrote {out_path}")
    print("\nAggregated (mean [95% CI]):")
    for k, v in aggregated.items():
        print(f"  {k:20s} {v['mean']:.4f}  [{v['ci_low']:.4f}, {v['ci_high']:.4f}]  (n={v['n']})")


if __name__ == "__main__":
    main()
