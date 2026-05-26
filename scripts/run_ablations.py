#!/usr/bin/env python
"""
Ablation orchestrator for PI-JEPA — produces the paper's Table 2.

Runs the full PI-JEPA model and each ablation variant (one component
removed at a time), pre-trains for N seeds each, and emits an aggregated
JSON suitable for direct paper-table generation.

Ablation variants (each is a config override on top of a base config):
    full                  — everything on (K predictors, per-stage decoders, physics, spectral)
    no_chain              — num_predictors = 1 (single monolithic predictor, no operator-split chain)
    no_per_stage_decoders — decoder.per_stage = false (single shared decoder)
    no_physics            — pretraining.physics.enabled = false
    fd_physics            — physics.residual_type = "fd" (vs spectral)
    spectral_physics      — physics.residual_type = "spectral"
    no_vicreg             — pretraining.vicreg weights → 0

Note: variants that depended on un-wired config keys (splitting dispatch,
multifidelity loader) were removed to keep the ablation table honest.
Reintroduce them after the corresponding pretrainer plumbing lands.

For each variant: run N seeds, bootstrap-CI aggregate JEPA loss + per-loss
components from the final checkpoints. Saves ablation_table.json.

Usage:
    python scripts/run_ablations.py \
        --base-config configs/darcy_3d_mf_smoke.yaml \
        --output outputs_ablation/darcy_3d \
        --n-seeds 3
"""

import argparse
import copy
import json
import os
import subprocess
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PI-JEPA"))

from eval.paper_metrics import bootstrap_ci_95


def deep_update(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and k in out and isinstance(out[k], dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out


# Each variant is (name, override-dict-applied-to-base-config).
# Every variant here must override a config key that has a real consumer in
# pretrainer.py / model builder — otherwise the variant produces identical
# numbers to `full` and the ablation is misleading.
VARIANTS = [
    ("full", {}),  # K=2 chained predictors, per-stage decoders, current physics
    ("no_chain", {
        # Collapse to a single monolithic predictor — genuinely changes the
        # model (no operator-split chain to traverse). Mirrors the working
        # `no_splitting` variant in scripts/run_full_benchmarks.py.
        "model": {
            "num_predictors": 1,
            "predictor": {
                "stages": [{"name": "unified", "depth": 4, "heads": 4,
                            "hidden_dim": 256}],
            },
        },
    }),
    ("no_per_stage_decoders", {
        # Read by scripts/finetune_pijepa.py at the decoder build site.
        "decoder": {"per_stage": False},
    }),
    ("no_physics", {
        "pretraining": {"physics": {"enabled": False}},
        "loss": {"physics": {"enabled": False}},
    }),
    ("fd_physics", {
        "physics": {"residual_type": "fd"},
        "pretraining": {"physics": {"enabled": True}},
        "loss": {"physics": {"enabled": True}},
    }),
    ("spectral_physics", {
        "physics": {"residual_type": "spectral"},
        "pretraining": {"physics": {"enabled": True}},
        "loss": {"physics": {"enabled": True}},
    }),
    ("no_vicreg", {
        "pretraining": {"vicreg": {"variance_weight": 0.0, "covariance_weight": 0.0}},
        "loss": {"regularization": {"variance": {"weight": 0.0}, "covariance": {"weight": 0.0}}},
    }),
]


def run_one_variant(name: str, cfg: dict, seed: int, out_root: str) -> Dict:
    """Run a single pretrain for (variant, seed). Returns final checkpoint metrics."""
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("experiment", {})["seed"] = int(seed)
    variant_dir = os.path.join(out_root, name, f"seed{seed}")
    os.makedirs(variant_dir, exist_ok=True)
    cfg_path = os.path.join(variant_dir, "_pretrain_cfg.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env = os.environ.copy()
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrain.py"),
        "--config", cfg_path,
        "--output", os.path.join(variant_dir, "pretrain"),
    ]
    print(f"  [{name} | seed {seed}] launching pretrain")
    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"  [{name} | seed {seed}] FAILED ({dt:.1f}s)")
        print(res.stderr[-1500:])
        return {"_failed": True, "_dt": dt}

    ckpt = os.path.join(variant_dir, "pretrain", "checkpoint_final.pt")
    if not os.path.exists(ckpt):
        return {"_failed": True, "_dt": dt, "_reason": "no checkpoint"}
    blob = torch.load(ckpt, weights_only=False, map_location="cpu")
    m = dict(blob.get("metrics", {}))
    m["_dt"] = dt
    m["_failed"] = False
    return m


def aggregate(variant_per_seed: Dict[str, Dict[int, Dict]]) -> Dict[str, Dict]:
    agg = {}
    for variant, per_seed in variant_per_seed.items():
        agg[variant] = {}
        all_keys = set()
        for s in per_seed.values():
            for k, v in s.items():
                if k.startswith("_"):
                    continue
                if isinstance(v, (int, float)):
                    all_keys.add(k)
        for k in sorted(all_keys):
            vals = []
            for sd, m in per_seed.items():
                if not m.get("_failed") and k in m and isinstance(m[k], (int, float)):
                    vals.append(float(m[k]))
            if not vals:
                continue
            mean, lo, hi = bootstrap_ci_95(np.array(vals), n_boot=2000)
            agg[variant][k] = {"mean": mean, "ci_low": lo, "ci_high": hi, "n": len(vals)}
    return agg


def main():
    ap = argparse.ArgumentParser(description="PI-JEPA ablation orchestrator")
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-seeds", type=int, default=3)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--variants", nargs="+", default=None,
                    help="Subset of variants by name; default = all")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    with open(args.base_config, "r") as f:
        base_cfg = yaml.safe_load(f)

    variants = VARIANTS
    if args.variants:
        wanted = set(args.variants)
        variants = [(n, o) for (n, o) in VARIANTS if n in wanted]
        if not variants:
            print(f"None of the requested variants match. Available: {[v[0] for v in VARIANTS]}")
            sys.exit(1)

    print(f"Running ablation: {len(variants)} variants × {args.n_seeds} seeds")

    per_variant: Dict[str, Dict[int, Dict]] = {}
    for name, override in variants:
        cfg = deep_update(base_cfg, override)
        per_seed: Dict[int, Dict] = {}
        for seed in range(args.seed_start, args.seed_start + args.n_seeds):
            per_seed[seed] = run_one_variant(name, cfg, seed, args.output)
        per_variant[name] = per_seed

    agg = aggregate(per_variant)

    out_json = {
        "base_config": args.base_config,
        "n_seeds": args.n_seeds,
        "variants": [v[0] for v in variants],
        "per_seed": {n: {str(s): m for s, m in d.items()} for n, d in per_variant.items()},
        "aggregated": agg,
    }
    out_path = os.path.join(args.output, "ablation_table.json")
    with open(out_path, "w") as f:
        json.dump(out_json, f, indent=2)
    print(f"\nWrote {out_path}")

    # Print a quick text table on JEPA loss
    print("\nAblation summary (lower = better, JEPA loss mean [95% CI]):")
    print(f"  {'variant':<26s} {'JEPA':<28s} {'total':<28s}")
    for name in [v[0] for v in variants]:
        j = agg.get(name, {}).get("jepa")
        t = agg.get(name, {}).get("total")
        if j:
            j_str = f"{j['mean']:.4f} [{j['ci_low']:.4f},{j['ci_high']:.4f}]"
        else:
            j_str = "FAILED"
        if t:
            t_str = f"{t['mean']:.4f} [{t['ci_low']:.4f},{t['ci_high']:.4f}]"
        else:
            t_str = "FAILED"
        print(f"  {name:<26s} {j_str:<28s} {t_str:<28s}")


if __name__ == "__main__":
    main()
