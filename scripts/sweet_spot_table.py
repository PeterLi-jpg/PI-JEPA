#!/usr/bin/env python
"""
Sweet-spot characterization for PI-JEPA (reviewer YkpY W3).

YkpY's W3 critique: the original paper claimed broad applicability but the
results showed PI-JEPA wins only in a narrow regime ("when PDE involves
sharp fronts, label count ≈ 100, pretraining distribution is domain-
matched"). The honest response in the resubmission is to *characterize*
this regime explicitly, per-(dataset, N_labeled, baseline) cell, instead
of headlining the average.

This script walks the outputs/ tree produced by run_focused_paper.sh,
groups results by (dataset, N_labeled, method), aggregates mean +
bootstrap 95% CI across seeds, and emits:

  - `sweet_spot.json` : the full machine-readable table
  - `sweet_spot.md`   : a human-readable markdown report with ✓/–/✗
    markers per cell (PI-JEPA wins / ties within CI / loses)

Usage:
    python scripts/sweet_spot_table.py \
        --output-root outputs_focused/v1 \
        --out outputs_focused/v1/figures/sweet_spot

Layout it expects under <output-root>:
    pijepa_finetune/seed{S}_n{N}/pijepa_result.json
    pijepa_scratch/seed{S}_n{N}/pijepa_result.json
    pijepa_frozen/seed{S}_n{N}/pijepa_result.json
    baselines/<baseline>/seed{S}_n{N}/baseline_result.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


SEED_N_RE = re.compile(r"seed(\d+)_n(\d+)")


def _bootstrap_ci_95(xs: List[float], n_boot: int = 2000) -> Tuple[float, float, float]:
    """Mean + 95% CI of a list of scalars via percentile bootstrap."""
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), float(arr[0]), float(arr[0])
    means = []
    rng = np.random.default_rng(0)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        means.append(float(sample.mean()))
    return float(arr.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _collect_results(output_root: str) -> Dict[Tuple[str, int], List[float]]:
    """Walk the output tree, group rel_L2 by (method, N_labeled)."""
    bucket: Dict[Tuple[str, int], List[float]] = defaultdict(list)

    def _consume(json_path: str, default_method: str) -> None:
        m = SEED_N_RE.search(json_path)
        if m is None:
            return
        n_l = int(m.group(2))
        try:
            with open(json_path) as f:
                data = json.load(f)
        except Exception:
            return
        rl2 = data.get("relative_l2_mean")
        if rl2 is None or not np.isfinite(rl2):
            return
        method = data.get("method") or data.get("baseline") or default_method
        bucket[(str(method), n_l)].append(float(rl2))

    for p in glob.glob(os.path.join(output_root, "pijepa_finetune", "seed*_n*",
                                    "pijepa_result.json")):
        _consume(p, "pi_jepa_finetuned")
    for p in glob.glob(os.path.join(output_root, "pijepa_scratch", "seed*_n*",
                                    "pijepa_result.json")):
        _consume(p, "pi_jepa_from_scratch")
    for p in glob.glob(os.path.join(output_root, "pijepa_frozen", "seed*_n*",
                                    "pijepa_result.json")):
        _consume(p, "pi_jepa_frozen")
    for p in glob.glob(os.path.join(output_root, "baselines", "*", "seed*_n*",
                                    "baseline_result.json")):
        baseline_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        _consume(p, baseline_name)
    # Brandon's per-domain baseline path (back-compat with the old single-FNO
    # focused_paper structure).
    for p in glob.glob(os.path.join(output_root, "*_baseline", "seed*_n*",
                                    "baseline_result.json")):
        baseline_name = os.path.basename(os.path.dirname(os.path.dirname(p))).replace(
            "_baseline", ""
        )
        _consume(p, baseline_name)

    return bucket


def _aggregate(bucket: Dict[Tuple[str, int], List[float]]) -> Dict[Tuple[str, int], Dict]:
    out = {}
    for key, vals in bucket.items():
        mean, lo, hi = _bootstrap_ci_95(vals)
        out[key] = {"mean": mean, "ci_low": lo, "ci_high": hi,
                    "n_seeds": len(vals), "raw": vals}
    return out


def _verdict(pijepa: Dict, baseline: Dict) -> str:
    """Per-cell verdict: PI-JEPA wins / ties (CIs overlap) / loses."""
    if not pijepa or not baseline:
        return "?"
    p_lo, p_hi = pijepa["ci_low"], pijepa["ci_high"]
    b_lo, b_hi = baseline["ci_low"], baseline["ci_high"]
    # Lower rel_L2 is better.
    if pijepa["mean"] < baseline["mean"] and p_hi < b_lo:
        return "✓"  # PI-JEPA strictly better (CIs don't overlap)
    if pijepa["mean"] > baseline["mean"] and p_lo > b_hi:
        return "✗"  # PI-JEPA strictly worse
    return "–"  # CIs overlap → tie


def main():
    ap = argparse.ArgumentParser(description="Sweet-spot win/loss table per cell")
    ap.add_argument("--output-root", required=True,
                    help="Directory containing pijepa_*/, baselines/, etc.")
    ap.add_argument("--out", required=True,
                    help="Output path prefix (writes .json and .md).")
    ap.add_argument("--pijepa-method", default="pi_jepa_finetuned",
                    help="Which PI-JEPA variant to use as the comparison "
                    "anchor (default: pi_jepa_finetuned).")
    args = ap.parse_args()

    bucket = _collect_results(args.output_root)
    agg = _aggregate(bucket)
    if not agg:
        raise SystemExit(f"No result JSONs found under {args.output_root}")

    methods = sorted(set(m for (m, _) in agg.keys()))
    n_labeled = sorted(set(n for (_, n) in agg.keys()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # JSON
    json_out = {
        "output_root": args.output_root,
        "pijepa_method": args.pijepa_method,
        "n_labeled": n_labeled,
        "methods": methods,
        "cells": {
            f"{m}|{n}": {
                "mean": v["mean"], "ci_low": v["ci_low"], "ci_high": v["ci_high"],
                "n_seeds": v["n_seeds"],
            }
            for (m, n), v in agg.items()
        },
    }
    json_path = args.out + ".json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"Wrote {json_path}")

    # Markdown table — one row per N_labeled, one column per non-PI-JEPA method,
    # cell = mean ± half-CI plus ✓/–/✗ verdict against PI-JEPA at that N_l.
    baselines = [m for m in methods if m != args.pijepa_method]
    lines = []
    lines.append(f"# Sweet-spot table (anchor: `{args.pijepa_method}`)\n")
    lines.append(f"Reviewer YkpY W3 — honest per-(N_labeled, baseline) win/loss.\n")
    lines.append("Lower rel_L2 = better. ✓ = PI-JEPA strictly better (95% CIs disjoint), – = tie (CIs overlap), ✗ = PI-JEPA strictly worse.\n")
    header = "| N_labeled | " + " | ".join([args.pijepa_method] + baselines) + " |"
    sep = "|---" * (1 + 1 + len(baselines)) + "|"
    lines.append(header)
    lines.append(sep)
    for n in n_labeled:
        cells = [str(n)]
        pj = agg.get((args.pijepa_method, n), {})
        if pj:
            pj_str = f"{pj['mean']:.4f} ±{(pj['ci_high']-pj['ci_low'])/2:.4f}"
        else:
            pj_str = "—"
        cells.append(pj_str)
        for b in baselines:
            bv = agg.get((b, n), {})
            if not bv:
                cells.append("—")
                continue
            verdict = _verdict(pj, bv)
            cells.append(f"{bv['mean']:.4f} ±{(bv['ci_high']-bv['ci_low'])/2:.4f} {verdict}")
        lines.append("| " + " | ".join(cells) + " |")
    md_path = args.out + ".md"
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {md_path}")

    # Console summary: PI-JEPA win count
    wins, ties, losses = 0, 0, 0
    for n in n_labeled:
        pj = agg.get((args.pijepa_method, n), {})
        for b in baselines:
            v = _verdict(pj, agg.get((b, n), {}))
            if v == "✓":
                wins += 1
            elif v == "–":
                ties += 1
            elif v == "✗":
                losses += 1
    total = wins + ties + losses
    print(f"\nSummary across {total} (N_labeled, baseline) cells:")
    print(f"  PI-JEPA strictly better: {wins}/{total}")
    print(f"  Tie (CIs overlap):       {ties}/{total}")
    print(f"  PI-JEPA strictly worse:  {losses}/{total}")


if __name__ == "__main__":
    main()
