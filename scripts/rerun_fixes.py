#!/usr/bin/env python
"""
Targeted rerun: ablations (with fixed physics residual) + PINO on ADR.

This script reruns ONLY the experiments affected by the two fixes:
  1. Ablation study — physics residual no longer has the conflicting
     reconstruction loss (p was being pushed toward K). Needs fresh
     pretraining for the "full" and "no_physics" variants to see if
     the physics residual now helps instead of hurts.
  2. PINO on ADR — fix_shape now accepts multi-channel tensors.

Everything else from publication_v2 is reused as-is.

Usage:
    python scripts/rerun_fixes.py --output publication_v2
"""

import os
import sys
import json
import copy
import math
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from utils import load_config
from models import PredictionHead, build_encoder
from benchmarks import PINOWrapper
from benchmarks.utils import set_seed


# Import shared helpers from the main benchmark script
from run_full_benchmarks import (
    compute_ci, fmt_ci,
    load_darcy, load_adr,
    _build_unlabeled_loader_darcy,
    pretrain_on_domain, _deep_update,
    load_encoder, _make_head, finetune_pijepa, eval_model,
    limit, rel_l2,
    ABLATION_CONFIGS, ADR_EVAL_REGIME, ADR_REGIMES,
    DEFAULT_N_SEEDS, PRETRAIN_EPOCHS,
)


# ============================================================================
# 1. Rerun ablation study with fixed physics residual
# ============================================================================

def rerun_ablations(config, device, output_dir, n_seeds):
    """
    Rerun ablation study from scratch.

    The "full" variant needs fresh pretraining because the physics
    residual implementation changed (removed conflicting recon loss).
    The "no_physics" variant is unchanged but we rerun for consistency.
    """
    print(f"\n{'='*60}")
    print(f"Ablation Study (fixed physics residual, {n_seeds} seeds)")
    print(f"{'='*60}")

    tr, te, in_ch, out_ch = load_darcy()
    n_l = 100
    seed0 = config.get("experiment", {}).get("seed", 42)

    # Fresh output dir — no stale checkpoints to worry about
    # Pretraining will run from scratch into this directory

    results = {}

    # Full model (fresh pretrain with fixed physics)
    print("\n--- Full PI-JEPA (fixed physics) ---")
    enc_full = pretrain_on_domain("darcy", config, device, output_dir)
    errs = []
    for s in range(n_seeds):
        e, h, ca = finetune_pijepa(enc_full, config, tr, n_l, device,
                                   seed0 + s, in_ch, out_ch)
        errs.append(eval_model(e, h, te, device, 1, ca))
    m, c = compute_ci(errs)
    results["full"] = {"mean": m, "ci95": c, "seeds": errs}
    print(f"  full:            {fmt_ci(m, c)}")

    # Each ablation variant
    for abl_name, overrides in ABLATION_CONFIGS.items():
        print(f"\n--- {abl_name} ---")
        try:
            enc_abl = pretrain_on_domain(
                "darcy", config, device, output_dir,
                config_overrides=overrides)
            errs = []
            for s in range(n_seeds):
                e, h, ca = finetune_pijepa(enc_abl, config, tr, n_l, device,
                                           seed0 + s, in_ch, out_ch)
                errs.append(eval_model(e, h, te, device, 1, ca))
            m, c = compute_ci(errs)
            results[abl_name] = {"mean": m, "ci95": c, "seeds": errs}
            print(f"  {abl_name:16s} {fmt_ci(m, c)}")
        except Exception as ex:
            print(f"  {abl_name} FAILED: {ex}")
            import traceback; traceback.print_exc()
            results[abl_name] = {"error": str(ex)}

    # Summary
    full_mean = results["full"]["mean"]
    print(f"\n{'='*60}")
    print(f"Ablation Summary (N_l={n_l}, fixed physics)")
    print(f"{'='*60}")
    print(f"  {'Variant':<20s} {'Error':>14s}  {'Δ vs full':>10s}")
    print(f"  {'-'*46}")
    for name in ["full", "no_physics", "no_splitting", "no_vicreg", "no_masking"]:
        res = results.get(name, {})
        if "mean" in res:
            delta = (res["mean"] - full_mean) / full_mean * 100
            print(f"  {name:<20s} {fmt_ci(res['mean'], res['ci95']):>14s}  {delta:>+9.1f}%")

    # Save
    abl_dir = os.path.join(output_dir, "ablation_darcy")
    os.makedirs(abl_dir, exist_ok=True)
    with open(os.path.join(abl_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {abl_dir}/results.json")

    return results


# ============================================================================
# 2. Rerun PINO on ADR (fix_shape now handles multi-channel)
# ============================================================================

class _BaselineDL:
    """Adapter: (x, y) DataLoader → dict-yielding loader."""
    def __init__(self, ld, ic, oc):
        self._ld, self._ic, self._oc = ld, ic, oc
        self.batch_size = ld.batch_size
        self.dataset = ld.dataset
    def __iter__(self):
        for x, y in self._ld:
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            yield {"x": x[:, :self._ic], "y": y[:, :self._oc]}
    def __len__(self): return len(self._ld)


def rerun_pino_adr(device, n_seeds, output_dir):
    """Rerun PINO baseline on ADR with fixed multi-channel fix_shape."""
    print(f"\n{'='*60}")
    print(f"PINO on ADR (fixed multi-channel, {n_seeds} seeds)")
    print(f"{'='*60}")

    tr, te, in_ch, out_ch = load_adr()
    seed0 = 42
    n_labeled_list = [10, 25, 50, 100, 250, 500]

    results = {}

    for n_l in n_labeled_list:
        print(f"\n--- N_l = {n_l} ---")
        errs = []
        for s in range(n_seeds):
            set_seed(seed0 + s)
            try:
                w = PINOWrapper(
                    device=device,
                    in_channels=in_ch,
                    out_channels=out_ch,
                    modes=(16, 16),
                    hidden_channels=64,
                    physics_weight=0.1,
                )
                sub = limit(tr, n_l, seed0 + s)
                w.train_model(
                    _BaselineDL(sub, in_ch, out_ch),
                    epochs=300, lr=1e-3,
                )
                ps, ts = [], []
                with torch.no_grad():
                    for x, y in te:
                        x, y = x.to(device), y.to(device)
                        if x.dim() == 3: x = x.unsqueeze(1)
                        if y.dim() == 3: y = y.unsqueeze(1)
                        ps.append(w.predict(x[:, :in_ch]).cpu())
                        ts.append(y[:, :out_ch].cpu())
                err = rel_l2(torch.cat(ps), torch.cat(ts))
                errs.append(err)
            except Exception as ex:
                print(f"  seed {s} failed: {ex}")
                import traceback; traceback.print_exc()

        if errs:
            m, c = compute_ci(errs)
            results[n_l] = {"mean": m, "ci95": c, "seeds": errs}
            print(f"  PINO: {fmt_ci(m, c)}")
        else:
            print(f"  PINO: ALL SEEDS FAILED")

    # Save
    pino_dir = os.path.join(output_dir, "pino_adr_fix")
    os.makedirs(pino_dir, exist_ok=True)
    with open(os.path.join(pino_dir, "results.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2,
                  default=str)
    print(f"\n  Saved: {pino_dir}/results.json")

    # Also patch the main ADR results file
    adr_path = os.path.join(output_dir, "adr", "results.json")
    if os.path.exists(adr_path):
        with open(adr_path) as f:
            adr_results = json.load(f)
        adr_results["pino"] = {str(k): v for k, v in results.items()}
        with open(adr_path, "w") as f:
            json.dump(adr_results, f, indent=2, default=str)
        print(f"  Patched: {adr_path}")

    return results


# ============================================================================
# Main
# ============================================================================

def main():
    p = argparse.ArgumentParser(
        description="Targeted rerun: ablations + PINO ADR fix")
    p.add_argument("--config", default="configs/darcy.yaml")
    p.add_argument("--output", default="publication_v2_fixes")
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--skip-ablation", action="store_true")
    p.add_argument("--skip-pino", action="store_true")
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PI-JEPA Targeted Rerun")
    print(f"Device: {device}")
    print(f"Seeds: {args.n_seeds}")
    print(f"Output: {args.output}")
    print(f"Started: {datetime.now()}\n")

    if not args.skip_ablation:
        rerun_ablations(config, device, args.output, args.n_seeds)

    if not args.skip_pino:
        rerun_pino_adr(device, args.n_seeds, args.output)

    print(f"\n{'='*60}")
    print(f"Targeted rerun complete.")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
