#!/usr/bin/env python
"""
Full benchmark suite matching the PI-JEPA paper:
  1. Single-phase Darcy flow (entry-level)
  2. Two-phase CO2-water flow (K=2 operator splitting)
  3. Advection-diffusion-reaction (K=3 operator splitting)

Each benchmark:
  - Generates data if not present
  - Pretrains PI-JEPA on unlabeled fields
  - Evaluates data efficiency vs baselines (FNO, GeoFNO, DeepONet)
  - Saves results as CSV
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
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from utils import load_config
from models import ViTEncoder, PredictionHead, build_encoder
from benchmarks import FNOWrapper, DeepONetWrapper
from benchmarks.utils import set_seed


# ============================================================================
# Data generation helpers
# ============================================================================

def ensure_darcy_data(resolution=64, n_train=1000, n_test=200, seed=42):
    """Generate single-phase Darcy data if not present."""
    train_path = "data/darcy/darcy_train.pt"
    test_path = "data/darcy/darcy_test.pt"
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  Darcy data exists at {train_path}")
        return train_path, test_path

    print(f"  Generating {resolution}x{resolution} Darcy data...")
    from generate_darcy_data import generate_dataset
    os.makedirs("data/darcy", exist_ok=True)

    K_tr, p_tr = generate_dataset(n_train, resolution, seed=seed)
    K_te, p_te = generate_dataset(n_test, resolution, seed=seed + 1)

    torch.save({"x": torch.from_numpy(K_tr).float().unsqueeze(1),
                "y": torch.from_numpy(p_tr).float().unsqueeze(1)}, train_path)
    torch.save({"x": torch.from_numpy(K_te).float().unsqueeze(1),
                "y": torch.from_numpy(p_te).float().unsqueeze(1)}, test_path)
    return train_path, test_path


def ensure_twophase_data(resolution=64, n_train=500, n_test=100, seed=42):
    """Generate two-phase data if not present."""
    train_path = "data/twophase/twophase_train.h5"
    test_path = "data/twophase/twophase_test.h5"
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  Two-phase data exists at {train_path}")
        return train_path, test_path

    print(f"  Generating {resolution}x{resolution} two-phase data...")
    from generate_twophase_data import generate_dataset as gen_tp, save_hdf5
    os.makedirs("data/twophase", exist_ok=True)

    train_data = gen_tp(n_train, resolution, n_steps=10, seed=seed)
    save_hdf5(train_data, train_path)
    test_data = gen_tp(n_test, resolution, n_steps=10, seed=seed + 1)
    save_hdf5(test_data, test_path)
    return train_path, test_path


def ensure_adr_data(resolution=64, n_train=200, n_test=50, seed=42):
    """Generate ADR data if not present."""
    base = "data/adr"
    tag = "Pe1.0_Da0.1"
    train_path = os.path.join(base, f"adr_train_{tag}.h5")
    test_path = os.path.join(base, f"adr_test_{tag}.h5")
    if os.path.exists(train_path) and os.path.exists(test_path):
        print(f"  ADR data exists at {train_path}")
        return train_path, test_path

    print(f"  Generating {resolution}x{resolution} ADR data...")
    from generate_adr_data import generate_regime, save_hdf5
    os.makedirs(base, exist_ok=True)

    train_data = generate_regime(n_train, resolution, 20, 1.0, 0.1, seed)
    save_hdf5(train_data, train_path)
    test_data = generate_regime(n_test, resolution, 20, 1.0, 0.1, seed + 1)
    save_hdf5(test_data, test_path)
    return train_path, test_path


# ============================================================================
# Data loading helpers
# ============================================================================

def load_darcy_loaders(batch_size=32):
    train_data = torch.load("data/darcy/darcy_train.pt", weights_only=False)
    test_data = torch.load("data/darcy/darcy_test.pt", weights_only=False)
    tr = DataLoader(TensorDataset(train_data["x"], train_data["y"]),
                    batch_size=batch_size, shuffle=True)
    te = DataLoader(TensorDataset(test_data["x"], test_data["y"]),
                    batch_size=batch_size, shuffle=False)
    return tr, te, 1, 1  # in_channels, out_channels


def load_twophase_loaders(batch_size=32):
    """Load two-phase data as (permeability -> pressure_t0) regression."""
    import h5py
    with h5py.File("data/twophase/twophase_train.h5", "r") as f:
        x_tr = torch.from_numpy(f["permeability"][:]).float().unsqueeze(1)
        y_tr = torch.from_numpy(f["pressure"][:, 0]).float().unsqueeze(1)  # first timestep
    with h5py.File("data/twophase/twophase_test.h5", "r") as f:
        x_te = torch.from_numpy(f["permeability"][:]).float().unsqueeze(1)
        y_te = torch.from_numpy(f["pressure"][:, 0]).float().unsqueeze(1)
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    return tr, te, 1, 1


def load_adr_loaders(batch_size=32):
    """Load ADR data as (initial_concentration -> final_concentration) regression."""
    import h5py
    tag = "Pe1.0_Da0.1"
    with h5py.File(f"data/adr/adr_train_{tag}.h5", "r") as f:
        conc = torch.from_numpy(f["concentration"][:]).float()  # (N, n_sp, T, H, W)
        x_tr = conc[:, :, 0]   # initial: (N, n_sp, H, W)
        y_tr = conc[:, :, -1]  # final:   (N, n_sp, H, W)
    with h5py.File(f"data/adr/adr_test_{tag}.h5", "r") as f:
        conc = torch.from_numpy(f["concentration"][:]).float()
        x_te = conc[:, :, 0]
        y_te = conc[:, :, -1]
    n_sp = x_tr.shape[1]
    tr = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    te = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    return tr, te, n_sp, n_sp


# ============================================================================
# Training / evaluation core
# ============================================================================

def compute_relative_l2(pred, target, eps=1e-8):
    diff = (pred - target).reshape(pred.shape[0], -1)
    tgt = target.reshape(target.shape[0], -1)
    return (torch.norm(diff, dim=1) / (torch.norm(tgt, dim=1) + eps)).mean().item()


def limit_loader(loader, n, seed=42):
    ds = loader.dataset
    n_use = min(n, len(ds))
    subset = Subset(ds, list(range(n_use)))
    return DataLoader(subset, batch_size=min(loader.batch_size, n_use),
                      shuffle=True, generator=torch.Generator().manual_seed(seed))


def train_pijepa(encoder, config, train_loader, n_labeled, device, seed=42,
                 in_ch=1, out_ch=1):
    """Finetune pretrained encoder + prediction head."""
    set_seed(seed)
    enc = copy.deepcopy(encoder).to(device)
    enc.train()
    for p in enc.parameters():
        p.requires_grad = True

    cfg_enc = config.get("model", {}).get("encoder", {})
    cfg_ft = config.get("finetuning", {})
    head = PredictionHead(
        embed_dim=cfg_enc.get("embed_dim", 384),
        hidden_dim=cfg_ft.get("prediction_head", {}).get("hidden_dim", 768),
        output_channels=out_ch,
        image_size=cfg_enc.get("image_size", 64),
        patch_size=cfg_enc.get("patch_size", 8),
    ).to(device)

    # Channel adapter: project multi-channel input to 1 channel for pretrained encoder
    ch_adapt = None
    if in_ch > 1:
        ch_adapt = nn.Conv2d(in_ch, 1, 1).to(device)

    lr = float(cfg_ft.get("optim", {}).get("lr", 5e-4))
    enc_lr = lr * float(cfg_ft.get("full_finetune", {}).get("encoder_lr_multiplier", 0.2))
    epochs = cfg_ft.get("epochs", 300)

    params = [
        {"params": head.parameters(), "lr": lr},
        {"params": enc.parameters(), "lr": enc_lr},
    ]
    if ch_adapt is not None:
        params.append({"params": ch_adapt.parameters(), "lr": lr})

    opt = torch.optim.AdamW(params)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    ldr = limit_loader(train_loader, n_labeled, seed)

    for ep in range(epochs):
        for x, y in ldr:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            x_enc = ch_adapt(x) if ch_adapt is not None else x[:, :1]
            opt.zero_grad()
            z = enc(x_enc)
            pred = head(z)
            F.mse_loss(pred, y).backward()
            opt.step()
        sched.step()

    return enc, head, ch_adapt


def eval_pijepa(encoder, head, test_loader, device, in_ch=1, ch_adapt=None):
    encoder.eval()
    head.eval()
    if ch_adapt is not None:
        ch_adapt.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            if ch_adapt is not None:
                x_enc = ch_adapt(x)
            else:
                x_enc = x[:, :in_ch]
            z = encoder(x_enc)
            preds.append(head(z).cpu())
            tgts.append(y.cpu())
    return compute_relative_l2(torch.cat(preds), torch.cat(tgts))


def train_scratch(config, train_loader, n_labeled, device, seed=42,
                  in_ch=1, out_ch=1):
    """Train PI-JEPA architecture from scratch (no pretraining)."""
    set_seed(seed)
    enc = build_encoder(config, in_channels=in_ch).to(device)
    enc.train()

    cfg_enc = config.get("model", {}).get("encoder", {})
    cfg_ft = config.get("finetuning", {})
    head = PredictionHead(
        embed_dim=cfg_enc.get("embed_dim", 384),
        hidden_dim=cfg_ft.get("prediction_head", {}).get("hidden_dim", 768),
        output_channels=out_ch,
        image_size=cfg_enc.get("image_size", 64),
        patch_size=cfg_enc.get("patch_size", 8),
    ).to(device)

    lr = float(cfg_ft.get("optim", {}).get("lr", 5e-4))
    epochs = cfg_ft.get("epochs", 300)
    opt = torch.optim.AdamW([
        {"params": head.parameters(), "lr": lr},
        {"params": enc.parameters(), "lr": lr * 0.1},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
    ldr = limit_loader(train_loader, n_labeled, seed)

    for ep in range(epochs):
        for x, y in ldr:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            opt.zero_grad()
            z = enc(x[:, :in_ch])
            F.mse_loss(head(z), y).backward()
            opt.step()
        sched.step()

    return enc, head


def train_eval_baseline(name, train_loader, test_loader, n_labeled, device,
                        seed=42, in_ch=1, out_ch=1):
    """Train and evaluate a baseline model."""
    set_seed(seed)

    if name == "fno":
        wrapper = FNOWrapper(device=device, in_channels=in_ch, out_channels=out_ch,
                             modes=(16, 16), hidden_channels=64, n_layers=4)
    elif name == "geo_fno":
        from benchmarks.geo_fno import GeoFNOWrapper
        wrapper = GeoFNOWrapper(device=device)
    elif name == "deeponet":
        # DeepONet only supports single-channel; for multi-channel, use first channel
        wrapper = DeepONetWrapper(device=device)
        in_ch_eff = 1
        out_ch_eff = 1
    else:
        raise ValueError(f"Unknown baseline: {name}")

    if name != "deeponet":
        in_ch_eff = in_ch
        out_ch_eff = out_ch

    # Build dict loader — ensure shapes match what wrapper expects
    ldr = limit_loader(train_loader, n_labeled, seed)
    class DL:
        def __init__(self, loader, in_c, out_c):
            self._loader = loader
            self._in_c = in_c
            self._out_c = out_c
            self.batch_size = loader.batch_size
            self.dataset = loader.dataset
        def __iter__(self):
            for x, y in self._loader:
                if x.dim() == 3: x = x.unsqueeze(1)
                if y.dim() == 3: y = y.unsqueeze(1)
                # Wrappers with fix_shape expect specific channel counts
                # Reshape to match: (B, in_ch, H, W) -> (B, in_ch, H, W)
                yield {"x": x[:, :self._in_c], "y": y[:, :self._out_c]}
        def __len__(self):
            return len(self._loader)

    wrapper.train_model(DL(ldr, in_ch_eff, out_ch_eff), epochs=300, lr=1e-3)

    # Evaluate
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            if x.dim() == 3: x = x.unsqueeze(1)
            if y.dim() == 3: y = y.unsqueeze(1)
            p = wrapper.predict(x[:, :in_ch_eff])
            preds.append(p.cpu())
            tgts.append(y[:, :out_ch_eff].cpu())
    return compute_relative_l2(torch.cat(preds), torch.cat(tgts))


# ============================================================================
# Per-benchmark runner
# ============================================================================

def run_benchmark(
    name: str,
    config: Dict,
    pretrained_encoder: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    in_ch: int = 1,
    out_ch: int = 1,
    n_labeled_sweep: List[int] = None,
    baselines: List[str] = None,
    n_seeds: int = 3,
    output_dir: str = "outputs",
):
    if n_labeled_sweep is None:
        n_labeled_sweep = [10, 25, 50, 100, 250, 500]
    if baselines is None:
        baselines = ["fno", "deeponet"]

    seed_base = config.get("experiment", {}).get("seed", 42)
    results = {"pi_jepa": {}, "pi_jepa_scratch": {}}
    for b in baselines:
        results[b] = {}

    print(f"\n{'='*60}")
    print(f"Benchmark: {name}")
    print(f"{'='*60}")

    for n_l in n_labeled_sweep:
        print(f"\n--- N_l = {n_l} ---")

        # PI-JEPA (pretrained)
        errs = []
        for s in range(n_seeds):
            enc, head, ch_ad = train_pijepa(pretrained_encoder, config, train_loader,
                                     n_l, device, seed=seed_base + s,
                                     in_ch=in_ch, out_ch=out_ch)
            errs.append(eval_pijepa(enc, head, test_loader, device, in_ch, ch_ad))
        results["pi_jepa"][n_l] = sum(errs) / len(errs)
        print(f"  PI-JEPA:  {results['pi_jepa'][n_l]:.4f}")

        # PI-JEPA scratch
        errs = []
        for s in range(n_seeds):
            enc, head = train_scratch(config, train_loader, n_l, device,
                                      seed=seed_base + s, in_ch=in_ch, out_ch=out_ch)
            errs.append(eval_pijepa(enc, head, test_loader, device, in_ch))
        results["pi_jepa_scratch"][n_l] = sum(errs) / len(errs)
        print(f"  Scratch:  {results['pi_jepa_scratch'][n_l]:.4f}")

        # Baselines
        for bname in baselines:
            errs = []
            for s in range(n_seeds):
                try:
                    e = train_eval_baseline(bname, train_loader, test_loader,
                                            n_l, device, seed=seed_base + s,
                                            in_ch=in_ch, out_ch=out_ch)
                    errs.append(e)
                except Exception as ex:
                    print(f"  {bname} seed {s} failed: {ex}")
            if errs:
                results[bname][n_l] = sum(errs) / len(errs)
                print(f"  {bname}:  {results[bname][n_l]:.4f}")

    # Save
    out = os.path.join(output_dir, name)
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({k: {str(kk): vv for kk, vv in v.items()} for k, v in results.items()},
                  f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    header = "N_l".ljust(10)
    for m in results:
        header += m.ljust(18)
    print(header)
    print("-" * len(header))
    for n_l in n_labeled_sweep:
        row = str(n_l).ljust(10)
        for m in results:
            val = results[m].get(n_l, float("nan"))
            row += f"{val:.4f}".ljust(18)
        print(row)

    return results


# ============================================================================
# Main
# ============================================================================

def load_pretrained_encoder(checkpoint_path, config, device):
    enc = build_encoder(config, in_channels=1).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "encoder_state_dict" in ckpt:
        enc.load_state_dict(ckpt["encoder_state_dict"])
    elif "student_encoder" in ckpt:
        enc.load_state_dict(ckpt["student_encoder"])
    print(f"Loaded encoder from {checkpoint_path}")
    return enc


def main():
    parser = argparse.ArgumentParser(description="Full PI-JEPA benchmark suite")
    parser.add_argument("--config", default="configs/darcy.yaml")
    parser.add_argument("--checkpoint", default=None,
                        help="Pretrained encoder checkpoint (will pretrain if not provided)")
    parser.add_argument("--output", default="outputs/full_benchmarks")
    parser.add_argument("--benchmarks", nargs="+",
                        default=["darcy", "twophase", "adr"],
                        help="Which benchmarks to run")
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--n-labeled", type=int, nargs="+",
                        default=[10, 25, 50, 100, 250, 500])
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Benchmarks: {args.benchmarks}")
    print(f"Started: {datetime.now()}")

    # Ensure Darcy data exists (needed for pretraining regardless)
    ensure_darcy_data()

    # Load or run pretraining
    checkpoint_path = args.checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Using existing checkpoint: {checkpoint_path}")
    else:
        # Run pretraining
        pretrain_dir = os.path.join(args.output, "pretrain")
        os.makedirs(pretrain_dir, exist_ok=True)
        default_ckpt = os.path.join(pretrain_dir, "checkpoint_best.pt")

        if os.path.exists(default_ckpt):
            print(f"Found existing checkpoint: {default_ckpt}")
            checkpoint_path = default_ckpt
        else:
            print("\nNo checkpoint found — running pretraining...")
            from pretrain import pretrain
            checkpoint_path = pretrain(args.config, pretrain_dir)
            print(f"Pretraining complete: {checkpoint_path}")

    # Load pretrained encoder
    encoder = load_pretrained_encoder(checkpoint_path, config, device)

    all_results = {}

    # --- Benchmark 1: Single-phase Darcy ---
    if "darcy" in args.benchmarks:
        ensure_darcy_data()
        tr, te, in_ch, out_ch = load_darcy_loaders()
        all_results["darcy"] = run_benchmark(
            "darcy", config, encoder, tr, te, device,
            in_ch=in_ch, out_ch=out_ch,
            n_labeled_sweep=args.n_labeled, n_seeds=args.n_seeds,
            output_dir=args.output,
        )

    # --- Benchmark 2: Two-phase CO2-water ---
    if "twophase" in args.benchmarks:
        ensure_twophase_data()
        tr, te, in_ch, out_ch = load_twophase_loaders()
        all_results["twophase"] = run_benchmark(
            "twophase", config, encoder, tr, te, device,
            in_ch=in_ch, out_ch=out_ch,
            n_labeled_sweep=args.n_labeled, n_seeds=args.n_seeds,
            output_dir=args.output,
        )

    # --- Benchmark 3: ADR reactive transport ---
    if "adr" in args.benchmarks:
        ensure_adr_data()
        tr, te, in_ch, out_ch = load_adr_loaders()
        all_results["adr"] = run_benchmark(
            "adr", config, encoder, tr, te, device,
            in_ch=in_ch, out_ch=out_ch,
            n_labeled_sweep=args.n_labeled, n_seeds=args.n_seeds,
            baselines=["fno", "deeponet"],  # GeoFNO only supports 1-ch
            output_dir=args.output,
        )

    # Save combined results
    with open(os.path.join(args.output, "all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll benchmarks complete. Results in {args.output}/")
    print(f"Finished: {datetime.now()}")


if __name__ == "__main__":
    main()
