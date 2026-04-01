#!/usr/bin/env python
"""
Full PI-JEPA benchmark suite for publication.

Paper-specified benchmarks:
  1. Single-phase Darcy flow — 1,000 samples, 64x64
  2. Two-phase CO2-water flow — 2,000 trajectories, 64x64, K=2
  3. ADR reactive transport  — 1,000 trajectories/regime, 64x64, K=3

Each benchmark: pretrain → finetune sweep → evaluate vs baselines (FNO, DeepONet)
3-seed averaging per data point.
"""

import os
import sys
import json
import argparse
import copy
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
from benchmarks import FNOWrapper, DeepONetWrapper
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
    # Normalize: zero-mean unit-variance per channel using training stats
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
    """Load the held-out ADR regime (Pe=1, Da=0.1) for finetuning evaluation."""
    import h5py
    Pe, Da = ADR_EVAL_REGIME
    tag = f"Pe{Pe}_Da{Da}"
    def _load(path):
        with h5py.File(path, "r") as f:
            c = torch.from_numpy(f["concentration"][:]).float()  # (N, n_sp, T, H, W)
        return c[:, :, 0], c[:, :, -1]  # initial -> final
    x_tr, y_tr = _load(f"data/adr/adr_train_{tag}.h5")
    x_te, y_te = _load(f"data/adr/adr_test_{tag}.h5")
    n_sp = x_tr.shape[1]
    return (DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True),
            DataLoader(TensorDataset(x_te, y_te), batch_size=bs, shuffle=False),
            n_sp, n_sp)


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


def train_eval_baseline(name, tr, te, n_l, device, seed, in_ch, out_ch):
    set_seed(seed)
    # DeepONet only handles 1 channel
    in_eff = 1 if name == "deeponet" else in_ch
    out_eff = 1 if name == "deeponet" else out_ch

    if name == "fno":
        w = FNOWrapper(device=device, in_channels=in_eff, out_channels=out_eff,
                       modes=(16, 16), hidden_channels=64, n_layers=4)
    elif name == "deeponet":
        w = DeepONetWrapper(device=device)
    else:
        raise ValueError(name)

    class DL:
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

    w.train_model(DL(limit(tr, n_l, seed), in_eff, out_eff), epochs=300, lr=1e-3)

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
# Benchmark runner
# ============================================================================

def run_benchmark(name, config, encoder, tr, te, device,
                  in_ch, out_ch, n_labeled, baselines, n_seeds, out_dir):
    seed0 = config.get("experiment", {}).get("seed", 42)
    results = {"pi_jepa": {}, "pi_jepa_scratch": {}}
    for b in baselines:
        results[b] = {}

    print(f"\n{'='*60}\nBenchmark: {name}\n{'='*60}")

    for n_l in n_labeled:
        print(f"\n--- N_l = {n_l} ---")

        # PI-JEPA pretrained
        errs = []
        for s in range(n_seeds):
            e, h, ca = finetune_pijepa(encoder, config, tr, n_l, device,
                                       seed0+s, in_ch, out_ch)
            errs.append(eval_model(e, h, te, device, 1, ca))
        results["pi_jepa"][n_l] = sum(errs)/len(errs)
        print(f"  pi_jepa:         {results['pi_jepa'][n_l]:.4f}  ({[f'{e:.4f}' for e in errs]})")

        # PI-JEPA scratch
        errs = []
        for s in range(n_seeds):
            e, h = train_scratch(config, tr, n_l, device, seed0+s, in_ch, out_ch)
            errs.append(eval_model(e, h, te, device, in_ch))
        results["pi_jepa_scratch"][n_l] = sum(errs)/len(errs)
        print(f"  pi_jepa_scratch: {results['pi_jepa_scratch'][n_l]:.4f}")

        # Baselines
        for bname in baselines:
            errs = []
            for s in range(n_seeds):
                try:
                    errs.append(train_eval_baseline(bname, tr, te, n_l, device,
                                                    seed0+s, in_ch, out_ch))
                except Exception as ex:
                    print(f"  {bname} seed {s} failed: {ex}")
            if errs:
                results[bname][n_l] = sum(errs)/len(errs)
                print(f"  {bname:18s} {results[bname][n_l]:.4f}")

    # Save
    bdir = os.path.join(out_dir, name)
    os.makedirs(bdir, exist_ok=True)
    with open(os.path.join(bdir, "results.json"), "w") as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()}
                   for k, v in results.items()}, f, indent=2)

    # Summary table
    print(f"\n{'='*60}\n{name} — Summary\n{'='*60}")
    models = list(results.keys())
    hdr = "N_l".ljust(8) + "".join(m.ljust(20) for m in models)
    print(hdr)
    print("-" * len(hdr))
    for n_l in n_labeled:
        row = str(n_l).ljust(8)
        for m in models:
            v = results[m].get(n_l, float("nan"))
            row += f"{v:.4f}".ljust(20)
        print(row)

    # Improvement summary
    print(f"\nPretraining benefit (pi_jepa vs scratch):")
    for n_l in n_labeled:
        pj = results["pi_jepa"].get(n_l)
        sc = results["pi_jepa_scratch"].get(n_l)
        if pj and sc and sc > 0:
            imp = (sc - pj) / sc * 100
            print(f"  N_l={n_l:>4d}: {imp:+.1f}%")

    return results


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
    p.add_argument("--n-seeds", type=int, default=3)
    args = p.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"PI-JEPA Publication Benchmark Suite")
    print(f"Device: {device}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Seeds per point: {args.n_seeds}")
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
            from pretrain import pretrain
            ckpt = pretrain(args.config, pretrain_dir)
            print(f"Pretraining done: {ckpt}")

    encoder = load_encoder(ckpt, config, device)

    # --- Benchmarks ---
    os.makedirs(args.output, exist_ok=True)
    all_results = {}

    # Paper N_l sweeps
    darcy_nl = [10, 25, 50, 100, 250, 500]
    adr_nl   = [10, 25, 50, 100, 250]

    if "darcy" in args.benchmarks:
        tr, te, ic, oc = load_darcy()
        all_results["darcy"] = run_benchmark(
            "darcy", config, encoder, tr, te, device,
            ic, oc, darcy_nl, ["fno", "deeponet"], args.n_seeds, args.output)

    if "twophase" in args.benchmarks:
        tr, te, ic, oc = load_twophase()
        all_results["twophase"] = run_benchmark(
            "twophase", config, encoder, tr, te, device,
            ic, oc, darcy_nl, ["fno", "deeponet"], args.n_seeds, args.output)

    if "adr" in args.benchmarks:
        tr, te, ic, oc = load_adr()
        all_results["adr"] = run_benchmark(
            "adr", config, encoder, tr, te, device,
            ic, oc, adr_nl, ["fno", "deeponet"], args.n_seeds, args.output)

    # Save combined
    with open(os.path.join(args.output, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}\nAll benchmarks complete.")
    print(f"Results: {args.output}/")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
