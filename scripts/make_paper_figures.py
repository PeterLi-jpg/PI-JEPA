#!/usr/bin/env python
"""
Paper figures generator.

Produces publication-quality matplotlib figures from saved experiment
outputs. Three figure types:

  --kind qualitative   Worst- and best-case prediction panels (truth | pred | |err|).
                       Needs a checkpoint + test .pt path.

  --kind ablation      Bar chart of ablation_table.json with bootstrap CIs.

  --kind data_eff      Sample-efficiency curve (relative L2 vs N_labeled)
                       from a directory of run_multiseed JSONs.

Saves to a single output PNG. Matplotlib only; no seaborn / no extra deps.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PI-JEPA"))

import yaml


def fig_qualitative(args):
    """Worst- and best-case prediction panels for a fine-tuned checkpoint."""
    from models import build_encoder, Decoder3D, PIJEPA, Predictor
    import torch.nn as nn

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    blob = torch.load(args.checkpoint, weights_only=False, map_location="cpu")

    # Rebuild PIJEPA + decoders
    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available() else "cpu")
    encoder = build_encoder(cfg, in_channels=1).to(device)
    target_encoder = build_encoder(cfg, in_channels=1).to(device)
    encoder.load_state_dict(blob["encoder_state_dict"])
    target_encoder.load_state_dict(blob["target_encoder_state_dict"])
    predictors = [Predictor(cfg).to(device) for _ in blob["predictor_state_dicts"]]
    for p, sd in zip(predictors, blob["predictor_state_dicts"]):
        p.load_state_dict(sd)
    m = PIJEPA(encoder=encoder, target_encoder=target_encoder, predictors=predictors,
               embed_dim=cfg["model"]["encoder"]["embed_dim"],
               patch_size=cfg["model"]["encoder"]["patch_size"]).to(device)
    decoders = nn.ModuleList([
        Decoder3D(
            embed_dim=cfg["decoder"]["embed_dim"],
            out_channels=cfg["decoder"]["out_channels"],
            image_size=cfg["decoder"]["image_size"],
            patch_size=cfg["decoder"]["patch_size"],
        )
        for _ in blob["decoder_state_dicts"]
    ]).to(device)
    for d, sd in zip(decoders, blob["decoder_state_dicts"]):
        d.load_state_dict(sd)

    test_blob = torch.load(args.test_pt, weights_only=False, map_location="cpu")
    x = test_blob["x"].to(device).float()
    y = test_blob["y"].to(device).float()

    # Predict via the operator-split chain on the full latent
    with torch.no_grad():
        z_full = m.encoder(x)
        B, N, D = z_full.shape
        z_t = m.mask_token.expand(B, N, D).contiguous()
        for pred in m.predictors:
            z_t = pred.forward_chained(z_t, z_full)
        out = decoders[-1](z_t)

    # Compute per-sample rel L2
    err = (out - y).flatten(1).pow(2).sum(dim=1).sqrt() / (y.flatten(1).pow(2).sum(dim=1).sqrt() + 1e-8)
    order = err.argsort()
    best = order[:2].tolist()
    worst = order[-2:].tolist()

    # Pick middle z-slice for visualization
    fig, axes = plt.subplots(4, 3, figsize=(9, 12))
    for ridx, (label, idx) in enumerate([("BEST 1", best[0]), ("BEST 2", best[1]),
                                          ("WORST 1", worst[0]), ("WORST 2", worst[1])]):
        # tensors are (B, C, D, H, W); pick mid z-slice
        z_mid = x.shape[2] // 2
        truth_slice = y[idx, 0, z_mid].cpu().numpy()
        pred_slice = out[idx, 0, z_mid].cpu().numpy()
        err_slice = np.abs(pred_slice - truth_slice)
        for cidx, (sub, title, cmap) in enumerate([
            (truth_slice, "truth", "viridis"),
            (pred_slice, "prediction", "viridis"),
            (err_slice, "|error|", "magma"),
        ]):
            ax = axes[ridx, cidx]
            im = ax.imshow(sub, cmap=cmap)
            plt.colorbar(im, ax=ax, fraction=0.046)
            ax.set_title(f"{label} — {title}  (rel L2={err[idx]:.3f})" if cidx == 0 else title)
            ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Wrote {args.out}")


def fig_ablation(args):
    """Bar chart of ablation variants on JEPA loss with bootstrap CIs."""
    with open(args.input_json, "r") as f:
        data = json.load(f)
    agg = data["aggregated"]
    variants = data["variants"]
    means = []
    los = []
    his = []
    metric = args.metric
    for v in variants:
        cell = agg.get(v, {}).get(metric)
        if cell:
            means.append(cell["mean"])
            los.append(cell["mean"] - cell["ci_low"])
            his.append(cell["ci_high"] - cell["mean"])
        else:
            means.append(np.nan); los.append(0); his.append(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    xs = np.arange(len(variants))
    ax.bar(xs, means, yerr=[los, his], capsize=4,
           color=["#2266aa"] + ["#dd6655"] * (len(variants) - 1))
    ax.set_xticks(xs)
    ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel(f"{metric} (mean, 95% CI)")
    ax.set_title(f"PI-JEPA ablation on {data.get('base_config', '?')}")
    for x_, m_ in zip(xs, means):
        if not np.isnan(m_):
            ax.text(x_, m_, f"{m_:.4f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Wrote {args.out}")


def fig_data_efficiency(args):
    """Sample-efficiency curve from a directory of multiseed JSONs.

    Expects a directory layout like:
        <input_dir>/<method>/n<N_labeled>/multiseed_results.json
    or a flat list of *.json files named like `<method>_n<N>.json`.
    """
    pattern = os.path.join(args.input_dir, "**", "multiseed_results.json")
    files = sorted(glob.glob(pattern, recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.json")))

    series = {}  # method -> list of (n_labeled, mean, ci_low, ci_high)
    for f in files:
        with open(f, "r") as fh:
            d = json.load(fh)
        method = d.get("note") or os.path.basename(os.path.dirname(f)) or os.path.basename(f)
        n_l = d.get("n_labeled")
        agg = d.get("aggregated", {})
        cell = agg.get(args.metric)
        if cell and n_l is not None:
            series.setdefault(method, []).append(
                (n_l, cell["mean"], cell["ci_low"], cell["ci_high"])
            )

    fig, ax = plt.subplots(figsize=(8, 5))
    for method, pts in series.items():
        pts.sort(key=lambda t: t[0])
        ns = [p[0] for p in pts]
        means = [p[1] for p in pts]
        los = [p[1] - p[2] for p in pts]
        his = [p[3] - p[1] for p in pts]
        ax.errorbar(ns, means, yerr=[los, his], marker="o", capsize=3, label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("# labeled samples (N_l)")
    ax.set_ylabel(f"{args.metric} (mean, 95% CI)")
    ax.set_title("Sample-efficiency curve")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches="tight")
    print(f"Wrote {args.out}")


def main():
    ap = argparse.ArgumentParser(description="PI-JEPA paper figure generator")
    sub = ap.add_subparsers(dest="kind", required=True)

    qp = sub.add_parser("qualitative", help="Best/worst prediction panels")
    qp.add_argument("--checkpoint", required=True)
    qp.add_argument("--config", required=True)
    qp.add_argument("--test-pt", required=True)
    qp.add_argument("--out", required=True)

    ap2 = sub.add_parser("ablation", help="Bar chart of ablation_table.json")
    ap2.add_argument("--input-json", required=True)
    ap2.add_argument("--metric", default="jepa")
    ap2.add_argument("--out", required=True)

    dp = sub.add_parser("data_eff", help="Sample-efficiency curves")
    dp.add_argument("--input-dir", required=True)
    dp.add_argument("--metric", default="relative_l2_mean")
    dp.add_argument("--out", required=True)

    args = ap.parse_args()
    if args.kind == "qualitative":
        fig_qualitative(args)
    elif args.kind == "ablation":
        fig_ablation(args)
    elif args.kind == "data_eff":
        fig_data_efficiency(args)


if __name__ == "__main__":
    main()
