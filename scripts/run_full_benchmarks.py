#!/usr/bin/env python
"""
Full PI-JEPA benchmark suite for publication.

Paper-specified benchmarks:
  1. Single-phase Darcy flow — 1,000 samples, 64x64
  2. Two-phase CO2-water flow — 2,000 trajectories, 64x64, K=2
  3. ADR reactive transport  — 1,000 trajectories/regime, 64x64, K=3

Each benchmark: pretrain → finetune sweep → evaluate vs baselines
5-seed averaging with 95% confidence intervals per data point.

Baselines: FNO, DeepONet, PINO

Ablation studies:
  - no_physics: pretraining without PDE residual loss
  - no_splitting: single monolithic predictor (K=1) instead of operator-split bank
  - no_vicreg: pretraining without VICReg collapse prevention
  - no_masking: pretraining with full context (no spatial block masking)
"""

import os
import sys
import json
import argparse
import copy
import math
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

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
from benchmarks import FNOWrapper, DeepONetWrapper, PINOWrapper
from benchmarks.utils import set_seed

# ============================================================================
# Paper-specified constants
# ============================================================================
DARCY_N_TRAIN = 1000
DARCY_N_TEST = 200
TWOPHASE_N_TRAIN = 1600
TWOPHASE_N_TEST = 200
ADR_N_TRAIN = 1000   # per regime
ADR_N_TEST = 200     # per regime
ADR_REGIMES = [
    (0.1, 0.01), (0.1, 0.1), (0.1, 1.0),
    (1.0, 0.01), (1.0, 0.1), (1.0, 1.0),
    (10.0, 0.01), (10.0, 0.1), (10.0, 1.0),
]
ADR_EVAL_REGIME = (1.0, 0.1)  # held-out regime for labeled finetuning

PRETRAIN_EPOCHS = 500
DEFAULT_N_SEEDS = 5  # upgraded from 3


# ============================================================================
# Statistics helpers
# ============================================================================

def compute_ci(values, confidence=0.95):
    """Compute mean and 95% CI half-width from a list of values."""
    n = len(values)
    if n < 2:
        return float(np.mean(values)), 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    # t-distribution critical value for 95% CI
    from scipy.stats import t as t_dist
    t_crit = t_dist.ppf((1 + confidence) / 2, df=n - 1)
    ci = t_crit * std / math.sqrt(n)
    return mean, ci


def fmt_ci(mean, ci):
    """Format mean ± CI for display."""
    if ci == 0:
        return f"{mean:.4f}"
    return f"{mean:.4f}±{ci:.4f}"


# ============================================================================
# Data generation
# ============================================================================

def ensure_darcy_data(seed=42):
    train_path = "data/darcy/darcy_train.pt"
    test_path = "data/darcy/darcy_test.pt"
    if os.path.exists(train_path) and os.path.exists(test_path):
        return
    print("Generating Darcy data (1000 train / 200 test, 64x64)...")
    from generate_darcy_data import generate_dataset
    os.makedirs("data/darcy", exist_ok=True)
    K_tr, p_tr = generate_dataset(DARCY_N_TRAIN, 64, seed=seed)
    K_te, p_te = generate_dataset(DARCY_N_TEST, 64, seed=seed + 1)
    torch.save({"x": torch.from_numpy(K_tr).float().unsqueeze(1),
                "y": torch.from_numpy(p_tr).float().unsqueeze(1)}, train_path)
    torch.save({"x": torch.from_numpy(K_te).float().unsqueeze(1),
                "y": torch.from_numpy(p_te).float().unsqueeze(1)}, test_path)
    print("  Done.")


def ensure_twophase_data(seed=42):
    train_path = "data/twophase/twophase_train.h5"
    test_path = "data/twophase/twophase_test.h5"
    if os.path.exists(train_path) and os.path.exists(test_path):
        return
    print(f"Generating two-phase data ({TWOPHASE_N_TRAIN} train / {TWOPHASE_N_TEST} test, 64x64)...")
    from generate_twophase_data import generate_dataset as gen_tp, save_hdf5
    os.makedirs("data/twophase", exist_ok=True)
    save_hdf5(gen_tp(TWOPHASE_N_TRAIN, 64, n_steps=10, seed=seed),  train_path)
    save_hdf5(gen_tp(TWOPHASE_N_TEST,  64, n_steps=10, seed=seed+1), test_path)
    print("  Done.")


def ensure_adr_data(seed=42):
    base = "data/adr"
    os.makedirs(base, exist_ok=True)
    from generate_adr_data import generate_regime, save_hdf5
    for Pe, Da in ADR_REGIMES:
        tag = f"Pe{Pe}_Da{Da}"
        tr = os.path.join(base, f"adr_train_{tag}.h5")
        te = os.path.join(base, f"adr_test_{tag}.h5")
        if os.path.exists(tr) and os.path.exists(te):
            continue
        print(f"  Generating ADR regime Pe={Pe} Da={Da} ({ADR_N_TRAIN} train / {ADR_N_TEST} test)...")
        save_hdf5(generate_regime(ADR_N_TRAIN, 64, 20, Pe, Da, seed), tr)
        save_hdf5(generate_regime(ADR_N_TEST,  64, 20, Pe, Da, seed+1), te)
    print("  ADR data ready.")


# ============================================================================
# Data loaders
# ============================================================================

def load_darcy(bs=32):
    tr = torch.load("data/darcy/darcy_train.pt", weights_only=False)
    te = torch.load("data/darcy/darcy_test.pt", weights_only=False)
    return (DataLoader(TensorDataset(tr["x"], tr["y"]), batch_size=bs, shuffle=True),
            DataLoader(TensorDataset(te["x"], te["y"]), batch_size=bs, shuffle=False),
            1, 1)


def load_twophase(bs=32):
    import h5py
    def _load(path):
        with h5py.File(path, "r") as f:
            x = torch.from_numpy(f["permeability"][:]).float().unsqueeze(1)
            y = torch.from_numpy(f["pressure"][:, 0]).float().unsqueeze(1)
        return x, y
    x_tr, y_tr = _load("data/twophase/twophase_train.h5")
    x_te, y_te = _load("data/twophase/twophase_test.h5")
    x_mean, x_std = x_tr.mean(), x_tr.std() + 1e-8
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    x_tr = (x_tr - x_mean) / x_std
    x_te = (x_te - x_mean) / x_std
    y_tr = (y_tr - y_mean) / y_std
    y_te = (y_te - y_mean) / y_std
    return (DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True),
            DataLoader(TensorDataset(x_te, y_te), batch_size=bs, shuffle=False),
            1, 1)


def load_adr(bs=32):
    import h5py
    Pe, Da = ADR_EVAL_REGIME
    tag = f"Pe{Pe}_Da{Da}"
    def _load(path):
        with h5py.File(path, "r") as f:
            c = torch.from_numpy(f["concentration"][:]).float()
        return c[:, :, 0], c[:, :, -1]
    x_tr, y_tr = _load(f"data/adr/adr_train_{tag}.h5")
    x_te, y_te = _load(f"data/adr/adr_test_{tag}.h5")
    n_sp = x_tr.shape[1]
    return (DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True),
            DataLoader(TensorDataset(x_te, y_te), batch_size=bs, shuffle=False),
            n_sp, n_sp)


# ============================================================================
# Domain-matched pretraining
# ============================================================================

def _build_unlabeled_loader_darcy(bs=64):
    d = torch.load("data/darcy/darcy_train.pt", weights_only=False)
    return DataLoader(TensorDataset(d["x"]), batch_size=bs, shuffle=True)


def _build_unlabeled_loader_twophase(bs=64):
    import h5py
    with h5py.File("data/twophase/twophase_train.h5", "r") as f:
        x = torch.from_numpy(f["permeability"][:]).float().unsqueeze(1)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return DataLoader(TensorDataset(x), batch_size=bs, shuffle=True)


def _build_unlabeled_loader_adr(bs=64):
    import h5py
    all_x = []
    for Pe, Da in ADR_REGIMES:
        tag = f"Pe{Pe}_Da{Da}"
        path = f"data/adr/adr_train_{tag}.h5"
        if not os.path.exists(path):
            continue
        with h5py.File(path, "r") as f:
            c = torch.from_numpy(f["concentration"][:]).float()
        c0 = c[:, :, 0].mean(dim=1, keepdim=True)
        all_x.append(c0)
    x = torch.cat(all_x, dim=0)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return DataLoader(TensorDataset(x), batch_size=bs, shuffle=True)


def pretrain_on_domain(domain, config, device, output_dir, n_epochs=None,
                       config_overrides=None):
    """Pretrain a PI-JEPA encoder on domain-specific unlabeled data."""
    from training.pretrainer import build_model_for_pretraining, SelfSupervisedPretrainer

    if n_epochs is None:
        n_epochs = PRETRAIN_EPOCHS

    # Apply config overrides for ablations
    cfg = copy.deepcopy(config)
    if config_overrides:
        _deep_update(cfg, config_overrides)

    suffix = ""
    if config_overrides:
        suffix = "_" + "_".join(sorted(config_overrides.keys()))
    ckpt_dir = os.path.join(output_dir, f"pretrain_{domain}{suffix}")
    best_ckpt = os.path.join(ckpt_dir, "checkpoint_best.pt")

    if os.path.exists(best_ckpt):
        print(f"  Found existing {domain}{suffix} pretrain checkpoint")
        return load_encoder(best_ckpt, cfg, device)

    print(f"  Pretraining on {domain} unlabeled data ({n_epochs} epochs){suffix}...")

    loaders = {
        "darcy": _build_unlabeled_loader_darcy,
        "twophase": _build_unlabeled_loader_twophase,
        "adr": _build_unlabeled_loader_adr,
    }
    raw_loader = loaders[domain]()

    class _DictWrapper(torch.utils.data.Dataset):
        def __init__(self, ds):
            self.ds = ds
        def __len__(self):
            return len(self.ds)
        def __getitem__(self, idx):
            return {"x": self.ds[idx][0]}

    data_loader = DataLoader(
        _DictWrapper(raw_loader.dataset),
        batch_size=raw_loader.batch_size, shuffle=True,
    )

    model, decoder = build_model_for_pretraining(cfg, device)
    pretrainer = SelfSupervisedPretrainer(
        model=model, decoder=decoder, config=cfg, device=device
    )
    pretrainer.pretrain(data_loader=data_loader, n_epochs=n_epochs,
                        checkpoint_dir=ckpt_dir)
    return load_encoder(best_ckpt, cfg, device)


def _deep_update(base, overrides):
    """Recursively update nested dict."""
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v


# ============================================================================
# Core training / evaluation
# ============================================================================

def rel_l2(pred, target, eps=1e-8):
    d = (pred - target).reshape(pred.shape[0], -1)
    t = target.reshape(target.shape[0], -1)
    return (torch.norm(d, dim=1) / (torch.norm(t, dim=1) + eps)).mean().item()


def limit(loader, n, seed=42):
    ds = loader.dataset
    n_use = min(n, len(ds))
    sub = Subset(ds, list(range(n_use)))
    return DataLoader(sub, batch_size=min(loader.batch_size, n_use),
                      shuffle=True, generator=torch.Generator().manual_seed(seed))


def _make_head(config, out_ch):
    c = config.get("model", {}).get("encoder", {})
    f = config.get("finetuning", {})
    return PredictionHead(
        embed_dim=c.get("embed_dim", 384),
        hidden_dim=f.get("prediction_head", {}).get("hidden_dim", 768),
        output_channels=out_ch,
        image_size=c.get("image_size", 64),
        patch_size=c.get("patch_size", 8),
    )


def finetune_pijepa(encoder, config, loader, n_l, device, seed, in_ch, out_ch):
    """Finetune pretrained encoder + new head + optional channel adapter."""
    set_seed(seed)
    enc = copy.deepcopy(encoder).to(device)
    enc.train()
    for p in enc.parameters():
        p.requires_grad = True

    head = _make_head(config, out_ch).to(device)
    ch_adapt = nn.Conv2d(in_ch, 1, 1).to(device) if in_ch > 1 else None

    ft = config.get("finetuning", {})
    lr = float(ft.get("optim", {}).get("lr", 5e-4))
    enc_lr = lr * float(ft.get("full_finetune", {}).get("encoder_lr_multiplier", 0.2))
    epochs = ft.get("epochs", 300)

    params = [{"params": head.parameters(), "lr": lr},
              {"params": enc.parameters(), "lr": enc_lr}]
    if ch_adapt:
        params.append({"params": ch_adapt.parameters(), "lr": lr})

    opt = torch.optim.AdamW(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for _ in range(epochs):
        for x, y in limit(loader, n_l, seed):
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            x_enc = ch_adapt(x) if ch_adapt else x[:, :1]
            opt.zero_grad()
            F.mse_loss(head(enc(x_enc)), y).backward()
            opt.step()
        sched.step()
    return enc, head, ch_adapt


def train_scratch(config, loader, n_l, device, seed, in_ch, out_ch):
    set_seed(seed)
    enc = build_encoder(config, in_channels=in_ch).to(device)
    enc.train()
    head = _make_head(config, out_ch).to(device)

    ft = config.get("finetuning", {})
    lr = float(ft.get("optim", {}).get("lr", 5e-4))
    epochs = ft.get("epochs", 300)

    opt = torch.optim.AdamW([
        {"params": head.parameters(), "lr": lr},
        {"params": enc.parameters(), "lr": lr * 0.1},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    for _ in range(epochs):
        for x, y in limit(loader, n_l, seed):
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            opt.zero_grad()
            F.mse_loss(head(enc(x[:, :in_ch])), y).backward()
            opt.step()
        sched.step()
    return enc, head


def eval_model(enc, head, loader, device, in_ch, ch_adapt=None):
    enc.eval(); head.eval()
    if ch_adapt: ch_adapt.eval()
    ps, ts = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            x_enc = ch_adapt(x) if ch_adapt else x[:, :in_ch]
            ps.append(head(enc(x_enc)).cpu())
            ts.append(y.cpu())
    return rel_l2(torch.cat(ps), torch.cat(ts))


class _BaselineDL:
    """Adapter that wraps a (x, y) DataLoader into dict-yielding loader."""
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


def train_eval_baseline(name, tr, te, n_l, device, seed, in_ch, out_ch):
    set_seed(seed)
    in_eff = 1 if name == "deeponet" else in_ch
    out_eff = 1 if name == "deeponet" else out_ch

    if name == "fno":
        w = FNOWrapper(device=device, in_channels=in_eff, out_channels=out_eff,
                       modes=(16, 16), hidden_channels=64, n_layers=4)
    elif name == "pino":
        w = PINOWrapper(device=device, in_channels=in_eff, out_channels=out_eff,
                        modes=(16, 16), hidden_channels=64, physics_weight=0.1)
    elif name == "deeponet":
        w = DeepONetWrapper(device=device)
    else:
        raise ValueError(name)

    w.train_model(_BaselineDL(limit(tr, n_l, seed), in_eff, out_eff),
                  epochs=300, lr=1e-3)

    ps, ts = [], []
    with torch.no_grad():
        for x, y in te:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            ps.append(w.predict(x[:, :in_eff]).cpu())
            ts.append(y[:, :out_eff].cpu())
    return rel_l2(torch.cat(ps), torch.cat(ts))


# ============================================================================
# Benchmark runner (5-seed with CI)
# ============================================================================

def run_benchmark(name, config, encoder, tr, te, device,
                  in_ch, out_ch, n_labeled, baselines, n_seeds, out_dir):
    """Run a full benchmark with 5-seed averaging and 95% CI."""
    seed0 = config.get("experiment", {}).get("seed", 42)
    # Store per-seed errors for CI computation
    raw = {"pi_jepa": {}, "pi_jepa_scratch": {}}
    for b in baselines:
        raw[b] = {}

    print(f"\n{'='*60}\nBenchmark: {name}  ({n_seeds} seeds)\n{'='*60}")

    for n_l in n_labeled:
        print(f"\n--- N_l = {n_l} ---")

        # PI-JEPA pretrained
        errs = []
        for s in range(n_seeds):
            e, h, ca = finetune_pijepa(encoder, config, tr, n_l, device,
                                       seed0+s, in_ch, out_ch)
            errs.append(eval_model(e, h, te, device, 1, ca))
        raw["pi_jepa"][n_l] = errs
        m, c = compute_ci(errs)
        print(f"  pi_jepa:         {fmt_ci(m, c)}")

        # PI-JEPA scratch
        errs = []
        for s in range(n_seeds):
            e, h = train_scratch(config, tr, n_l, device, seed0+s, in_ch, out_ch)
            errs.append(eval_model(e, h, te, device, in_ch))
        raw["pi_jepa_scratch"][n_l] = errs
        m, c = compute_ci(errs)
        print(f"  scratch:         {fmt_ci(m, c)}")

        # Baselines (FNO, PINO, DeepONet)
        for bname in baselines:
            errs = []
            for s in range(n_seeds):
                try:
                    errs.append(train_eval_baseline(bname, tr, te, n_l, device,
                                                    seed0+s, in_ch, out_ch))
                except Exception as ex:
                    print(f"  {bname} seed {s} failed: {ex}")
            if errs:
                raw[bname][n_l] = errs
                m, c = compute_ci(errs)
                print(f"  {bname:18s} {fmt_ci(m, c)}")

    # Build results dict with mean, std, ci, and raw per-seed values
    results = {}
    for model_name, nl_dict in raw.items():
        results[model_name] = {}
        for n_l, errs in nl_dict.items():
            m, c = compute_ci(errs)
            results[model_name][n_l] = {
                "mean": m,
                "std": float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0,
                "ci95": c,
                "seeds": errs,
            }

    # Save
    bdir = os.path.join(out_dir, name)
    os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, "results.json"), "w") as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)

    # Summary table
    print(f"\n{'='*60}\n{name} — Summary (mean ± 95% CI)\n{'='*60}")
    models = list(results.keys())
    hdr = "N_l".ljust(8) + "".join(m.ljust(24) for m in models)
    print(hdr)
    print("-" * len(hdr))
    for n_l in n_labeled:
        row = str(n_l).ljust(8)
        for m in models:
            entry = results[m].get(n_l, {})
            if entry:
                row += fmt_ci(entry["mean"], entry["ci95"]).ljust(24)
            else:
                row += "N/A".ljust(24)
        print(row)

    return results


# ============================================================================
# Ablation studies
# ============================================================================

ABLATION_CONFIGS = {
    "no_physics": {
        "pretraining": {"physics": {"enabled": False}},
        "loss": {"physics": {"enabled": False}},
    },
    "no_splitting": {
        "model": {
            "num_predictors": 1,
            "predictor": {
                "stages": [{"name": "unified", "depth": 6, "heads": 6,
                            "hidden_dim": 384}],
            },
        },
    },
    "no_vicreg": {
        "pretraining": {"vicreg": {"variance_weight": 0.0,
                                    "covariance_weight": 0.0}},
    },
    "no_masking": {
        "masking": {"context_ratio": 1.0},
    },
}


def run_ablation_study(config, device, output_dir, n_seeds, benchmark="darcy"):
    """
    Run ablation study on Darcy benchmark.

    For each ablation variant, pretrain a new encoder with the modified config,
    then finetune at N_l=100 and compare against the full PI-JEPA model.
    """
    print(f"\n{'='*60}\nAblation Study (benchmark={benchmark}, N_l=100, {n_seeds} seeds)")
    print(f"{'='*60}")

    # Load data
    loaders = {"darcy": load_darcy, "twophase": load_twophase, "adr": load_adr}
    tr, te, in_ch, out_ch = loaders[benchmark]()
    n_l = 100
    seed0 = config.get("experiment", {}).get("seed", 42)

    ablation_results = {}

    # Full model (baseline for comparison)
    print("\n--- Full PI-JEPA (baseline) ---")
    enc_full = pretrain_on_domain(benchmark, config, device, output_dir)
    errs = []
    for s in range(n_seeds):
        e, h, ca = finetune_pijepa(enc_full, config, tr, n_l, device,
                                   seed0+s, in_ch, out_ch)
        errs.append(eval_model(e, h, te, device, 1, ca))
    m, c = compute_ci(errs)
    ablation_results["full"] = {"mean": m, "ci95": c, "seeds": errs}
    print(f"  full:          {fmt_ci(m, c)}")

    # Each ablation
    for abl_name, overrides in ABLATION_CONFIGS.items():
        print(f"\n--- {abl_name} ---")
        try:
            enc_abl = pretrain_on_domain(
                benchmark, config, device, output_dir,
                config_overrides=overrides)
            errs = []
            for s in range(n_seeds):
                e, h, ca = finetune_pijepa(enc_abl, config, tr, n_l, device,
                                           seed0+s, in_ch, out_ch)
                errs.append(eval_model(e, h, te, device, 1, ca))
            m, c = compute_ci(errs)
            ablation_results[abl_name] = {"mean": m, "ci95": c, "seeds": errs}
            print(f"  {abl_name:16s} {fmt_ci(m, c)}")
        except Exception as ex:
            print(f"  {abl_name} FAILED: {ex}")
            ablation_results[abl_name] = {"error": str(ex)}

    # Summary
    print(f"\n{'='*60}\nAblation Summary ({benchmark}, N_l={n_l})\n{'='*60}")
    full_mean = ablation_results["full"]["mean"]
    print(f"  {'Variant':<20s} {'Error':>14s}  {'Δ vs full':>10s}")
    print(f"  {'-'*46}")
    for name, res in ablation_results.items():
        if "mean" in res:
            delta = (res["mean"] - full_mean) / full_mean * 100
            print(f"  {name:<20s} {fmt_ci(res['mean'], res['ci95']):>14s}  {delta:>+9.1f}%")

    # Save
    abl_dir = os.path.join(output_dir, f"ablation_{benchmark}")
    os.makedirs(abl_dir, exist_ok=True)
    with open(os.path.join(abl_dir, "results.json"), "w") as f:
        json.dump(ablation_results, f, indent=2, default=str)

    return ablation_results


# ============================================================================
# Paper-ready export
# ============================================================================

def _get_mean(results, model, n_l):
    """Safely extract mean from results dict."""
    entry = results.get(model, {}).get(n_l, results.get(model, {}).get(str(n_l), {}))
    if isinstance(entry, dict):
        return entry.get("mean", float("nan"))
    if isinstance(entry, (int, float)):
        return float(entry)
    return float("nan")


def _get_ci(results, model, n_l):
    """Safely extract CI from results dict."""
    entry = results.get(model, {}).get(n_l, results.get(model, {}).get(str(n_l), {}))
    if isinstance(entry, dict):
        return entry.get("ci95", 0.0)
    return 0.0


def export_pgfplots_dat(results, benchmark, models, n_labeled, out_dir):
    """Export a .dat file per model for pgfplots \\addplot table."""
    bdir = os.path.join(out_dir, "pgfplots")
    os.makedirs(bdir, exist_ok=True)
    for model in models:
        path = os.path.join(bdir, f"{benchmark}_{model}.dat")
        with open(path, "w") as f:
            f.write("n_l  mean  ci_lo  ci_hi\n")
            for n_l in n_labeled:
                m = _get_mean(results, model, n_l)
                c = _get_ci(results, model, n_l)
                f.write(f"{n_l}  {m:.6f}  {m - c:.6f}  {m + c:.6f}\n")
    print(f"  pgfplots data: {bdir}/{benchmark}_*.dat")


def export_latex_table(results, benchmark, models, n_labeled, out_dir,
                       caption="", label=""):
    """Export a ready-to-paste LaTeX table with mean ± CI and bold best."""
    path = os.path.join(out_dir, "tables", f"table_{benchmark}.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Display names
    display = {
        "pi_jepa": "PI-JEPA",
        "pi_jepa_scratch": "Scratch",
        "fno": "FNO",
        "pino": "PINO",
        "deeponet": "DeepONet",
    }

    ncols = len(models)
    col_spec = "r" + "c" * ncols

    lines = []
    lines.append(f"\\begin{{table}}[H]")
    lines.append(f"\\centering")
    if caption:
        lines.append(f"\\caption{{{caption}}}")
    if label:
        lines.append(f"\\label{{{label}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(f"\\toprule")

    header = "$N_\\ell$"
    for m in models:
        name = display.get(m, m)
        if m == "pi_jepa":
            header += f" & \\textbf{{{name}}}"
        else:
            header += f" & {name}"
    header += " \\\\"
    lines.append(header)
    lines.append("\\midrule")

    for n_l in n_labeled:
        means = {m: _get_mean(results, m, n_l) for m in models}
        cis = {m: _get_ci(results, m, n_l) for m in models}
        valid = {m: v for m, v in means.items() if not math.isnan(v)}
        best_model = min(valid, key=valid.get) if valid else None

        row = str(n_l)
        for m in models:
            mv = means[m]
            cv = cis[m]
            if math.isnan(mv):
                cell = "---"
            elif cv > 0:
                cell = f"{mv:.3f}$\\pm${cv:.3f}"
            else:
                cell = f"{mv:.3f}"
            if m == best_model:
                cell = f"\\textbf{{{cell}}}"
            row += f" & {cell}"
        row += " \\\\"
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  LaTeX table: {path}")


def export_paper_artifacts(all_results, out_dir):
    """Export all paper-ready artifacts: pgfplots .dat files and LaTeX tables."""
    benchmark_configs = {
        "darcy": {
            "models": ["pi_jepa", "pi_jepa_scratch", "fno", "pino", "deeponet"],
            "n_labeled": [10, 25, 50, 100, 250, 500],
            "caption": "Relative $\\ell_2$ error on single-phase Darcy flow ($64\\times64$). "
                       "Mean $\\pm$ 95\\% CI over 5 seeds. Best per row in \\textbf{bold}.",
            "label": "tab:darcy",
        },
        "twophase": {
            "models": ["pi_jepa", "pi_jepa_scratch", "fno", "pino", "deeponet"],
            "n_labeled": [10, 25, 50, 100, 250, 500],
            "caption": "Relative $\\ell_2$ error on two-phase CO$_2$-water flow ($64\\times64$). "
                       "Mean $\\pm$ 95\\% CI over 5 seeds.",
            "label": "tab:twophase",
        },
        "adr": {
            "models": ["pi_jepa", "pi_jepa_scratch", "fno", "pino", "deeponet"],
            "n_labeled": [10, 25, 50, 100, 250, 500],
            "caption": "Relative $\\ell_2$ error on PDEBench ADR ($64\\times64$, $n_c=2$). "
                       "Mean $\\pm$ 95\\% CI over 5 seeds.",
            "label": "tab:adr",
        },
    }

    for bname, bcfg in benchmark_configs.items():
        if bname not in all_results:
            continue
        results = all_results[bname]
        export_pgfplots_dat(results, bname, bcfg["models"],
                            bcfg["n_labeled"], out_dir)
        export_latex_table(results, bname, bcfg["models"],
                           bcfg["n_labeled"], out_dir,
                           caption=bcfg["caption"], label=bcfg["label"])

    # Domain-matched comparison table (if available)
    for bname in ["twophase", "adr"]:
        cross_key = f"{bname}_crossdomain"
        if cross_key in all_results and bname in all_results:
            _export_domain_comparison(
                all_results[bname], all_results[cross_key],
                bname, out_dir)

    # Ablation table
    if "ablation" in all_results:
        _export_ablation_table(all_results["ablation"], out_dir)


def _export_domain_comparison(domain_matched, cross_domain, benchmark, out_dir):
    """Export table comparing domain-matched vs cross-domain pretraining."""
    path = os.path.join(out_dir, "tables", f"table_{benchmark}_domain_comparison.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    n_labeled = [10, 25, 50, 100, 250, 500]

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Domain-matched vs.\\ cross-domain pretraining on {benchmark}. "
                 f"Mean $\\pm$ 95\\% CI over 5 seeds.}}")
    lines.append(f"\\label{{tab:{benchmark}_domain}}")
    lines.append("\\begin{tabular}{rccc}")
    lines.append("\\toprule")
    lines.append("$N_\\ell$ & Domain-matched & Darcy-pretrained & Scratch \\\\")
    lines.append("\\midrule")

    for n_l in n_labeled:
        dm_m = _get_mean(domain_matched, "pi_jepa", n_l)
        dm_c = _get_ci(domain_matched, "pi_jepa", n_l)
        cd_m = _get_mean(cross_domain, "pi_jepa", n_l)
        cd_c = _get_ci(cross_domain, "pi_jepa", n_l)
        sc_m = _get_mean(domain_matched, "pi_jepa_scratch", n_l)
        sc_c = _get_ci(domain_matched, "pi_jepa_scratch", n_l)

        vals = [v for v in [dm_m, cd_m, sc_m] if not math.isnan(v)]
        best = min(vals) if vals else None

        def _fmt(m, c, is_best):
            if math.isnan(m):
                return "---"
            s = f"{m:.3f}$\\pm${c:.3f}" if c > 0 else f"{m:.3f}"
            return f"\\textbf{{{s}}}" if is_best else s

        row = (f"{n_l} & {_fmt(dm_m, dm_c, dm_m == best)} "
               f"& {_fmt(cd_m, cd_c, cd_m == best)} "
               f"& {_fmt(sc_m, sc_c, sc_m == best)} \\\\")
        lines.append(row)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Domain comparison table: {path}")


def _export_ablation_table(ablation_results, out_dir):
    """Export ablation study as a LaTeX table."""
    path = os.path.join(out_dir, "tables", "table_ablation.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Ablation study on Darcy flow at $N_\\ell = 100$. "
                 "Mean $\\pm$ 95\\% CI over 5 seeds. "
                 "$\\Delta$ is relative change vs.\\ full model.}")
    lines.append("\\label{tab:ablation}")
    lines.append("\\begin{tabular}{lcc}")
    lines.append("\\toprule")
    lines.append("Variant & Rel.\\ $\\ell_2$ error & $\\Delta$ (\\%) \\\\")
    lines.append("\\midrule")

    full_mean = ablation_results.get("full", {}).get("mean", float("nan"))

    display = {
        "full": "Full PI-JEPA",
        "no_physics": "w/o physics residual",
        "no_splitting": "w/o operator splitting",
        "no_vicreg": "w/o VICReg",
        "no_masking": "w/o spatial masking",
    }

    for name in ["full", "no_physics", "no_splitting", "no_vicreg", "no_masking"]:
        res = ablation_results.get(name, {})
        if "mean" not in res:
            continue
        m, c = res["mean"], res.get("ci95", 0.0)
        cell = f"{m:.3f}$\\pm${c:.3f}" if c > 0 else f"{m:.3f}"
        if name == "full":
            cell = f"\\textbf{{{cell}}}"
            delta = "---"
        else:
            d = (m - full_mean) / full_mean * 100
            delta = f"{d:+.1f}"
        lines.append(f"{display.get(name, name)} & {cell} & {delta} \\\\")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Ablation table: {path}")


def export_compute_table(out_dir):
    """Export a computational cost summary table for the paper.

    Scans pretrain checkpoint dirs for timing metadata and model param counts.
    Falls back to estimates when metadata is unavailable.
    """
    import glob

    path = os.path.join(out_dir, "tables", "table_compute.tex")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Collect info from checkpoints
    rows = []
    for ckpt_path in sorted(glob.glob(os.path.join(out_dir, "pretrain_*",
                                                     "checkpoint_best.pt"))):
        domain = os.path.basename(os.path.dirname(ckpt_path)).replace("pretrain_", "")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            n_epochs = ckpt.get("epoch", PRETRAIN_EPOCHS)
            n_steps = ckpt.get("global_step", "—")
            metrics = ckpt.get("metrics", {})
            final_loss = metrics.get("total", "—")
            # Count encoder params
            enc_sd = ckpt.get("encoder_state_dict", {})
            n_params = sum(p.numel() for p in
                           (torch.zeros(s) for s in
                            (v.shape for v in enc_sd.values())))
            rows.append((domain, n_epochs, n_steps, n_params, final_loss))
        except Exception:
            rows.append((domain, "—", "—", "—", "—"))

    # Also count finetune cost (fixed: 300 epochs for all)
    lines = []
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{Computational cost summary. Encoder parameters are "
                 "shared across all benchmarks. Pretraining uses only unlabeled "
                 "data (no PDE solves). Finetuning uses 300 epochs on $N_\\ell$ "
                 "labeled samples.}")
    lines.append("\\label{tab:compute}")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Phase & Domain & Epochs & Steps & Encoder params \\\\")
    lines.append("\\midrule")

    for domain, epochs, steps, params, loss in rows:
        p_str = f"{params:,}" if isinstance(params, int) else str(params)
        lines.append(f"Pretrain & {domain} & {epochs} & {steps} & {p_str} \\\\")

    lines.append("\\midrule")
    lines.append("Finetune & all & 300 & — & (frozen or 0.2$\\times$ LR) \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Compute table: {path}")


# ============================================================================
# Main
# ============================================================================

def load_encoder(path, config, device):
    enc = build_encoder(config, in_channels=1).to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "encoder_state_dict" in ckpt:
        enc.load_state_dict(ckpt["encoder_state_dict"])
    elif "student_encoder" in ckpt:
        enc.load_state_dict(ckpt["student_encoder"])
    print(f"Loaded encoder from {path}")
    return enc


def main():
    p = argparse.ArgumentParser(description="PI-JEPA publication benchmarks")
    p.add_argument("--config", default="configs/darcy.yaml")
    p.add_argument("--checkpoint", default=None,
                   help="Pretrained checkpoint (auto-pretrains if missing)")
    p.add_argument("--output", default="outputs/publication")
    p.add_argument("--benchmarks", nargs="+",
                   default=["darcy", "twophase", "adr"])
    p.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    p.add_argument("--domain-matched", action="store_true",
                   help="Pretrain separate encoders per benchmark domain")
    p.add_argument("--ablation", action="store_true",
                   help="Run ablation study after benchmarks")
    p.add_argument("--ablation-benchmark", default="darcy",
                   help="Which benchmark to use for ablation (default: darcy)")
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PI-JEPA Publication Benchmark Suite")
    print(f"Device: {device}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Seeds per point: {args.n_seeds}")
    print(f"Domain-matched pretraining: {args.domain_matched}")
    print(f"Ablation study: {args.ablation}")
    print(f"Started: {datetime.now()}\n")

    # --- Data generation ---
    print("="*60 + "\nPhase 0: Data Generation\n" + "="*60)
    ensure_darcy_data()
    if "twophase" in args.benchmarks:
        ensure_twophase_data()
    if "adr" in args.benchmarks:
        ensure_adr_data()

    # --- Pretraining ---
    print("\n" + "="*60 + "\nPhase 1: Pretraining\n" + "="*60)

    # Always pretrain a Darcy encoder (needed as cross-domain baseline)
    darcy_encoder = None

    if args.domain_matched:
        encoders = {}
        for bname in args.benchmarks:
            encoders[bname] = pretrain_on_domain(
                bname, config, device, args.output)
        darcy_encoder = encoders.get("darcy")
        if darcy_encoder is None:
            darcy_encoder = pretrain_on_domain("darcy", config, device,
                                               args.output)
    else:
        ckpt = args.checkpoint
        if ckpt and os.path.exists(ckpt):
            print(f"Using existing checkpoint: {ckpt}")
        else:
            pretrain_dir = os.path.join(args.output, "pretrain")
            default_ckpt = os.path.join(pretrain_dir, "checkpoint_best.pt")
            if os.path.exists(default_ckpt):
                ckpt = default_ckpt
                print(f"Found existing checkpoint: {ckpt}")
            else:
                print("Running pretraining (500 epochs)...")
                darcy_encoder = pretrain_on_domain(
                    "darcy", config, device, args.output)
                ckpt = os.path.join(args.output, "pretrain_darcy",
                                    "checkpoint_best.pt")
                print(f"Pretraining done: {ckpt}")
        darcy_encoder = load_encoder(ckpt, config, device)
        encoders = {b: darcy_encoder for b in args.benchmarks}

    # --- Benchmarks ---
    os.makedirs(args.output, exist_ok=True)
    all_results = {}

    darcy_nl = [10, 25, 50, 100, 250, 500]
    adr_nl   = [10, 25, 50, 100, 250, 500]

    baseline_list = ["fno", "pino", "deeponet"]

    if "darcy" in args.benchmarks:
        tr, te, ic, oc = load_darcy()
        all_results["darcy"] = run_benchmark(
            "darcy", config, encoders["darcy"], tr, te, device,
            ic, oc, darcy_nl, baseline_list, args.n_seeds, args.output)

    if "twophase" in args.benchmarks:
        tr, te, ic, oc = load_twophase()
        all_results["twophase"] = run_benchmark(
            "twophase", config, encoders["twophase"], tr, te, device,
            ic, oc, darcy_nl, baseline_list, args.n_seeds, args.output)
        # Cross-domain comparison: also run Darcy-pretrained on twophase
        if args.domain_matched and encoders.get("twophase") is not darcy_encoder:
            print("\n--- Cross-domain comparison: Darcy-pretrained on twophase ---")
            all_results["twophase_crossdomain"] = run_benchmark(
                "twophase_crossdomain", config, darcy_encoder, tr, te, device,
                ic, oc, darcy_nl, [], args.n_seeds, args.output)

    if "adr" in args.benchmarks:
        tr, te, ic, oc = load_adr()
        all_results["adr"] = run_benchmark(
            "adr", config, encoders["adr"], tr, te, device,
            ic, oc, adr_nl, baseline_list, args.n_seeds, args.output)
        # Cross-domain comparison: also run Darcy-pretrained on ADR
        if args.domain_matched and encoders.get("adr") is not darcy_encoder:
            print("\n--- Cross-domain comparison: Darcy-pretrained on ADR ---")
            all_results["adr_crossdomain"] = run_benchmark(
                "adr_crossdomain", config, darcy_encoder, tr, te, device,
                ic, oc, adr_nl, [], args.n_seeds, args.output)

    # Save combined JSON
    with open(os.path.join(args.output, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Ablation study ---
    if args.ablation:
        print("\n" + "="*60 + "\nPhase 3: Ablation Study\n" + "="*60)
        if args.ablation_benchmark == "twophase":
            ensure_twophase_data()
        elif args.ablation_benchmark == "adr":
            ensure_adr_data()
        abl_results = run_ablation_study(
            config, device, args.output, args.n_seeds,
            benchmark=args.ablation_benchmark)
        all_results["ablation"] = abl_results
        with open(os.path.join(args.output, "all_results.json"), "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # --- Export paper-ready artifacts ---
    print("\n" + "="*60 + "\nPhase 4: Paper Export\n" + "="*60)
    export_paper_artifacts(all_results, args.output)

    # --- Computational cost summary ---
    print("\n" + "="*60 + "\nPhase 5: Computational Cost Summary\n" + "="*60)
    export_compute_table(args.output)

    print(f"\n{'='*60}\nAll benchmarks complete.")
    print(f"Results: {args.output}/")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
