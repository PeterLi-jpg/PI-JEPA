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


# ----------------------------------------------------------------------------
# External CCS benchmarks (CCSNet, FNO4CO2): NOT generated by this script.
# Files must be downloaded out of band. We collapse the time axis to a single
# representative snapshot and resize spatially to 64x64 so the 2D encoder
# path consumes them unchanged.
# ----------------------------------------------------------------------------


CCSNET_TARGETS = ("SG", "BPR", "BXMF", "BYMF", "BDENW", "BDENG", "P_init")


def ensure_ccsnet_data(seed=42, target_var: str = "SG"):
    root = "data/ccsnet/CCSNet_v1.0"
    needed = ["test_x.hdf5", f"test_y_{target_var}.hdf5"]
    missing = [f for f in needed if not os.path.exists(os.path.join(root, f))]
    if missing:
        print(f"  [ccsnet:{target_var}] MISSING under {root}: {missing}")
        print("  Get them from Drive folder 1SVZFkaxkAIjcGKew3rzGTmKW5tSBUGf7")
        raise FileNotFoundError(f"CCSNet missing: {missing}")
    print(f"  CCSNet:{target_var} ready ({root})")


def ensure_fno4co2_data(seed=42):
    root = "data/fno4co2/dataset"
    needed = ["dP_test_a.pt", "dP_test_u.pt"]
    missing = [f for f in needed if not os.path.exists(os.path.join(root, f))]
    if missing:
        print(f"  [fno4co2] MISSING under {root}: {missing}")
        print("  Get them from Drive folder 1fZQfMn_vsjKUXAfRV0q_gswtl8JEkVGo")
        raise FileNotFoundError(f"FNO4CO2 missing: {missing}")
    print(f"  FNO4CO2 ready ({root})")


def ensure_pdebench_adr_data(seed=42):
    """Real PDEBench ADR dataset (distinct from Brandon's synthetic `adr`).

    The 2D_diff-react_NA_NA.h5 file is ~13 GB downloaded from DaRUS:
        curl -L -C - -o data/pdebench_adr/2D_diff-react_NA_NA.h5 \\
             https://darus.uni-stuttgart.de/api/access/datafile/133017
    """
    path = "data/pdebench_adr/2D_diff-react_NA_NA.h5"
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"[pdebench_adr] MISSING {path}\n"
            "  Get it from DaRUS via curl (see ensure_pdebench_adr_data docstring)"
        )
    # Quick sanity-check that it's not truncated.
    try:
        import h5py
        with h5py.File(path, "r") as f:
            _ = list(f.keys())
    except OSError as e:
        raise OSError(
            f"[pdebench_adr] file at {path} is TRUNCATED or unreadable: {e}\n"
            "  Resume the curl download (`curl -L -C - ...`) before running."
        ) from e
    print(f"  PDEBench ADR ready ({path})")


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


# ----------------------------------------------------------------------------
# External CCS benchmark loaders. They collapse time and resize to 64x64 so
# the rest of the 2D pipeline consumes them unchanged. Split is 80/20 from
# the test_x file when no separate train split is downloaded.
# ----------------------------------------------------------------------------


def _ccsnet_collapse_resize(arr, t_index, target=64):
    """(N, H, W, T, C) -> (N, C, target, target) at one timestep, normalized."""
    arr = arr[:, :, :, t_index, :]               # (N, H, W, C)
    arr = arr.transpose(0, 3, 1, 2)                # (N, C, H, W)
    t = torch.from_numpy(arr).float()
    t = F.interpolate(t, size=(target, target), mode="bilinear", align_corners=False)
    return t


def load_ccsnet(bs=32, target_var="SG", train_frac=0.8, seed=42):
    """Load CCSNet (test_x, test_y_<var>) collapsed to one timestep + 64x64.

    Since only the test split is downloaded, we slice 80/20 into our own
    train/test partition (fixed seed for reproducibility). Final timestep
    used as the supervised target.
    """
    import h5py
    import torch.nn.functional as F
    root = "data/ccsnet/CCSNet_v1.0"
    with h5py.File(os.path.join(root, "test_x.hdf5"), "r") as f:
        x_arr = f["test_x"][:]
    with h5py.File(os.path.join(root, f"test_y_{target_var}.hdf5"), "r") as f:
        y_key = list(f.keys())[0]
        y_arr = f[y_key][:]
    if y_arr.ndim == 4:
        y_arr = y_arr[..., None]

    x = _ccsnet_collapse_resize(x_arr, t_index=0, target=64)            # input at t=0
    y = _ccsnet_collapse_resize(y_arr, t_index=-1, target=64)           # target at final t

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(train_frac * x.shape[0])
    tr_idx, te_idx = perm[:n_train], perm[n_train:]
    x_tr, y_tr = x[tr_idx], y[tr_idx]
    x_te, y_te = x[te_idx], y[te_idx]
    x_mean, x_std = x_tr.mean(), x_tr.std() + 1e-8
    y_mean, y_std = y_tr.mean(), y_tr.std() + 1e-8
    x_tr, x_te = (x_tr - x_mean) / x_std, (x_te - x_mean) / x_std
    y_tr, y_te = (y_tr - y_mean) / y_std, (y_te - y_mean) / y_std
    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True),
        DataLoader(TensorDataset(x_te, y_te), batch_size=bs, shuffle=False),
        1, 1,
    )


def _build_unlabeled_loader_ccsnet(bs=64, train_frac=0.8, seed=42):
    import h5py
    import torch.nn.functional as F
    root = "data/ccsnet/CCSNet_v1.0"
    with h5py.File(os.path.join(root, "test_x.hdf5"), "r") as f:
        x_arr = f["test_x"][:]
    x = _ccsnet_collapse_resize(x_arr, t_index=0, target=64)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(train_frac * x.shape[0])
    x_tr = x[perm[:n_train]]
    x_tr = (x_tr - x_tr.mean()) / (x_tr.std() + 1e-8)
    return DataLoader(TensorDataset(x_tr), batch_size=bs, shuffle=True)


def load_fno4co2(bs=32, train_frac=0.8, seed=42):
    """Load FNO4CO2 dP_test_a/u collapsed to one timestep + 64x64.

    dP_test_a has 12 input channels (perm + porosity + BCs + injection
    schedule). We collapse the time axis but PRESERVE all 12 channels,
    so the encoder runs at in_channels=12.
    """
    import torch.nn.functional as F
    root = "data/fno4co2/dataset"
    a = torch.load(os.path.join(root, "dP_test_a.pt"), weights_only=False, map_location="cpu").float()
    u = torch.load(os.path.join(root, "dP_test_u.pt"), weights_only=False, map_location="cpu").float()
    if a.dim() != 5 or u.dim() != 4:
        raise ValueError(f"FNO4CO2 unexpected shapes a={a.shape}, u={u.shape}")

    # Collapse time: a (N,H,W,T,C)->(N,C,H,W) at t=0;  u (N,H,W,T)->(N,1,H,W) at t=-1
    a = a[:, :, :, 0, :].permute(0, 3, 1, 2).contiguous()
    u = u[:, :, :, -1].unsqueeze(1).contiguous()
    a = F.interpolate(a, size=(64, 64), mode="bilinear", align_corners=False)
    u = F.interpolate(u, size=(64, 64), mode="bilinear", align_corners=False)

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(a.shape[0], generator=g)
    n_train = int(train_frac * a.shape[0])
    a_tr, a_te = a[perm[:n_train]], a[perm[n_train:]]
    u_tr, u_te = u[perm[:n_train]], u[perm[n_train:]]
    # Channel-aware normalization on a; scalar on u.
    a_mean = a_tr.mean(dim=(0, 2, 3), keepdim=True)
    a_std = a_tr.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    a_tr, a_te = (a_tr - a_mean) / a_std, (a_te - a_mean) / a_std
    u_mean = u_tr.mean()
    u_std = u_tr.std() + 1e-8
    u_tr, u_te = (u_tr - u_mean) / u_std, (u_te - u_mean) / u_std
    return (
        DataLoader(TensorDataset(a_tr, u_tr), batch_size=bs, shuffle=True),
        DataLoader(TensorDataset(a_te, u_te), batch_size=bs, shuffle=False),
        a.shape[1], 1,
    )


def _build_unlabeled_loader_fno4co2(bs=64, train_frac=0.8, seed=42):
    import torch.nn.functional as F
    root = "data/fno4co2/dataset"
    a = torch.load(os.path.join(root, "dP_test_a.pt"), weights_only=False, map_location="cpu").float()
    a = a[:, :, :, 0, :].permute(0, 3, 1, 2).contiguous()
    a = F.interpolate(a, size=(64, 64), mode="bilinear", align_corners=False)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(a.shape[0], generator=g)
    n_train = int(train_frac * a.shape[0])
    a_tr = a[perm[:n_train]]
    a_mean = a_tr.mean(dim=(0, 2, 3), keepdim=True)
    a_std = a_tr.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    a_tr = (a_tr - a_mean) / a_std
    return DataLoader(TensorDataset(a_tr), batch_size=bs, shuffle=True)


# ----------------------------------------------------------------------------
# PDEBench ADR (real, not synthetic). One big H5 file with N samples grouped
# under per-sample keys, each holding a "data" tensor of shape (T, H, W, n_sp).
# We collapse to (input=t_0, target=t_-1) and resize to 64x64.
# ----------------------------------------------------------------------------


def _load_pdebench_adr_arrays(train_frac: float, seed: int, target: int = 64):
    """Read the PDEBench reaction-diffusion H5 once and return (x_tr, y_tr, x_te, y_te).

    The file structure is one HDF5 group per sample with a "data" dataset.
    """
    import h5py
    import torch.nn.functional as F
    path = "data/pdebench_adr/2D_diff-react_NA_NA.h5"
    xs, ys = [], []
    with h5py.File(path, "r") as f:
        keys = sorted(f.keys())
        for k in keys:
            item = f[k]
            if "data" in item:
                arr = item["data"][...]  # (T, H, W, n_sp) typically
            else:
                # Single top-level dataset variant
                arr = item[...]
            xs.append(arr[0])    # initial timestep
            ys.append(arr[-1])   # final timestep
    x = torch.from_numpy(np.stack(xs, axis=0)).float()   # (N, H, W, C)
    y = torch.from_numpy(np.stack(ys, axis=0)).float()
    # NHWC -> NCHW
    x = x.permute(0, 3, 1, 2).contiguous()
    y = y.permute(0, 3, 1, 2).contiguous()
    x = F.interpolate(x, size=(target, target), mode="bilinear", align_corners=False)
    y = F.interpolate(y, size=(target, target), mode="bilinear", align_corners=False)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(x.shape[0], generator=g)
    n_train = int(train_frac * x.shape[0])
    return x[perm[:n_train]], y[perm[:n_train]], x[perm[n_train:]], y[perm[n_train:]]


def load_pdebench_adr(bs=32, train_frac=0.8, seed=42):
    import numpy as np
    x_tr, y_tr, x_te, y_te = _load_pdebench_adr_arrays(train_frac, seed)
    # Per-channel normalize on train stats
    x_mean = x_tr.mean(dim=(0, 2, 3), keepdim=True)
    x_std = x_tr.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    y_mean = y_tr.mean(dim=(0, 2, 3), keepdim=True)
    y_std = y_tr.std(dim=(0, 2, 3), keepdim=True) + 1e-8
    x_tr, x_te = (x_tr - x_mean) / x_std, (x_te - x_mean) / x_std
    y_tr, y_te = (y_tr - y_mean) / y_std, (y_te - y_mean) / y_std
    n_sp = x_tr.shape[1]
    return (
        DataLoader(TensorDataset(x_tr, y_tr), batch_size=bs, shuffle=True),
        DataLoader(TensorDataset(x_te, y_te), batch_size=bs, shuffle=False),
        n_sp, n_sp,
    )


def _build_unlabeled_loader_pdebench_adr(bs=64, train_frac=0.8, seed=42):
    import numpy as np
    x_tr, _, _, _ = _load_pdebench_adr_arrays(train_frac, seed)
    x_tr = (x_tr - x_tr.mean(dim=(0, 2, 3), keepdim=True)) / (x_tr.std(dim=(0, 2, 3), keepdim=True) + 1e-8)
    return DataLoader(TensorDataset(x_tr), batch_size=bs, shuffle=True)


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
        "ccsnet": _build_unlabeled_loader_ccsnet,
        "fno4co2": _build_unlabeled_loader_fno4co2,
        "pdebench_adr": _build_unlabeled_loader_pdebench_adr,
    }
    # CCSNet variants (one per output variable) share the same unlabeled inputs.
    for _tv in CCSNET_TARGETS:
        loaders[f"ccsnet_{_tv.lower()}"] = _build_unlabeled_loader_ccsnet
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


# Map dataset names to their unlabeled-loader builders. Used by both
# pretrain_on_domain (above) and pretrain_on_combined_pool (below).
_UNLABELED_LOADER_BUILDERS = {
    "darcy": _build_unlabeled_loader_darcy,
    "twophase": _build_unlabeled_loader_twophase,
    "adr": _build_unlabeled_loader_adr,
    "ccsnet": _build_unlabeled_loader_ccsnet,
    "fno4co2": _build_unlabeled_loader_fno4co2,
    "pdebench_adr": _build_unlabeled_loader_pdebench_adr,
}


def parse_combined_pool_spec(spec_str: str):
    """Parse the --combined-pool CLI argument.

    Format: "name1[:weight1],name2[:weight2],..." e.g.
        "ccsnet,fno4co2"            -> equal weights
        "ccsnet:0.6,fno4co2:0.3,darcy:0.1"

    Returns: list of (name, weight) tuples. Unspecified weights default
    to 1.0; final weights are NOT normalized here (MultiFidelityPretrainer
    normalizes internally).
    """
    out = []
    for item in spec_str.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            name, w = item.split(":", 1)
            out.append((name.strip(), float(w)))
        else:
            out.append((item, 1.0))
    if not out:
        raise ValueError(f"Empty --combined-pool spec: {spec_str!r}")
    return out


def pretrain_on_combined_pool(
    pool_spec, config, device, output_dir,
    target_shape=(32, 64, 64), n_epochs=None, samples_per_epoch=1024,
    batch_size=8,
):
    """Pretrain a PI-JEPA encoder on a tier-weighted combined pool of unlabeled
    parameter fields drawn from MULTIPLE datasets.

    Uses Brandon's IrregularGridProcessor (NaN sanitization) and
    MultiFidelityPretrainer (tier sampling), wired together via
    PI-JEPA/data/combined_pool.py.

    Args:
        pool_spec: list of (dataset_name, weight) tuples from
            parse_combined_pool_spec().
        config: full config dict (passed to the pretrainer + encoder builder).
        device: torch device.
        output_dir: where the checkpoint dir lives.
        target_shape: (D, H, W) every sample is trilinear-resized to.
            Default (32, 64, 64) is a reasonable common ground for the 3D
            datasets we support.
        n_epochs: defaults to PRETRAIN_EPOCHS from this module.
        samples_per_epoch: how many samples one combined-pool epoch draws.
            Pick something comparable to a single-dataset epoch.
        batch_size: DataLoader batch size for the combined pool.

    Returns:
        A loaded encoder (same return type as pretrain_on_domain).
    """
    # Late imports keep the script's top-level startup cheap when this
    # entry point isn't used.
    from training.pretrainer import build_model_for_pretraining, SelfSupervisedPretrainer
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "PI-JEPA"))
    from data.combined_pool import build_combined_pool_loader

    if n_epochs is None:
        n_epochs = PRETRAIN_EPOCHS

    cfg = copy.deepcopy(config)

    # Stable checkpoint dir based on pool composition so we can resume / cache.
    tag = "_".join(f"{n}{w:g}" for n, w in pool_spec)
    ckpt_dir = os.path.join(output_dir, f"pretrain_pool_{tag}")
    best_ckpt = os.path.join(ckpt_dir, "checkpoint_best.pt")
    if os.path.exists(best_ckpt):
        print(f"  Found existing combined-pool checkpoint at {ckpt_dir}")
        return load_encoder(best_ckpt, cfg, device)

    # Ensure all required source datasets are present on disk before we
    # try to materialize them via their per-domain ensure_* hooks.
    ensure_hooks = {
        "darcy": ensure_darcy_data,
        "twophase": ensure_twophase_data,
        "adr": ensure_adr_data,
        "ccsnet": lambda: ensure_ccsnet_data(target_var="SG"),
        "fno4co2": ensure_fno4co2_data,
        "pdebench_adr": ensure_pdebench_adr_data,
    }
    for name, _ in pool_spec:
        hook = ensure_hooks.get(name)
        if hook is None:
            raise KeyError(
                f"--combined-pool dataset '{name}' has no ensure_* hook. "
                f"Supported: {sorted(ensure_hooks)}"
            )
        hook()

    # Build the tier_specs CombinedPoolDataset wants. Each build_fn returns
    # the underlying torch Dataset (we strip Brandon's DataLoader wrapper).
    def _make_build_fn(name):
        builder = _UNLABELED_LOADER_BUILDERS[name]
        return lambda: builder().dataset

    tier_specs = [
        {"name": name, "weight": w, "build_fn": _make_build_fn(name)}
        for name, w in pool_spec
    ]

    print(f"  Pretraining on COMBINED POOL ({n_epochs} epochs)")
    print(f"    tiers: {[(s['name'], s['weight']) for s in tier_specs]}")
    print(f"    target_shape: {target_shape}, samples/epoch: {samples_per_epoch}")

    data_loader = build_combined_pool_loader(
        tier_specs=tier_specs,
        target_shape=tuple(target_shape),
        batch_size=batch_size,
        samples_per_epoch=samples_per_epoch,
        progressive=False,
        num_workers=0,
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
    loaders = {
        "darcy": load_darcy,
        "twophase": load_twophase,
        "adr": load_adr,
        "ccsnet": load_ccsnet,
        "fno4co2": load_fno4co2,
        "pdebench_adr": load_pdebench_adr,
    }
    # Each CCSNet output variable becomes its own benchmark name:
    # ccsnet_sg, ccsnet_bpr, ccsnet_bxmf, ccsnet_bymf, ccsnet_bdenw, ccsnet_bdeng, ccsnet_p_init
    for _tv in CCSNET_TARGETS:
        loaders[f"ccsnet_{_tv.lower()}"] = (lambda tv=_tv: (lambda bs=32: load_ccsnet(bs=bs, target_var=tv)))()
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
    p.add_argument("--combined-pool", default=None,
                   help="Pretrain ONE encoder on a tier-weighted combined "
                   "pool of unlabeled inputs from multiple datasets. "
                   "Format: 'name1[:weight],name2[:weight],...' e.g. "
                   "'ccsnet:0.6,fno4co2:0.3,darcy:0.1'. Mutually exclusive "
                   "with --domain-matched.")
    p.add_argument("--combined-pool-shape", nargs=3, type=int,
                   default=[32, 64, 64], metavar=("D", "H", "W"),
                   help="Common (D H W) every combined-pool sample is "
                   "resized to. Default: 32 64 64.")
    p.add_argument("--combined-pool-samples-per-epoch", type=int, default=1024,
                   help="Samples drawn per combined-pool epoch. Default 1024.")
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
    if "darcy" in args.benchmarks:
        ensure_darcy_data()
    if "twophase" in args.benchmarks:
        ensure_twophase_data()
    if "adr" in args.benchmarks:
        ensure_adr_data()
    # CCSNet: each variant points at a different test_y_<VAR>.hdf5
    for bname in args.benchmarks:
        if bname == "ccsnet":
            ensure_ccsnet_data(target_var="SG")
        elif bname.startswith("ccsnet_"):
            tv = bname[len("ccsnet_"):].upper()
            if tv == "P_INIT":
                tv = "P_init"
            ensure_ccsnet_data(target_var=tv)
    if "fno4co2" in args.benchmarks:
        ensure_fno4co2_data()
    if "pdebench_adr" in args.benchmarks:
        ensure_pdebench_adr_data()

    # --- Pretraining ---
    print("\n" + "="*60 + "\nPhase 1: Pretraining\n" + "="*60)

    # Always pretrain a Darcy encoder (needed as cross-domain baseline)
    darcy_encoder = None

    if args.combined_pool and args.domain_matched:
        raise SystemExit(
            "--combined-pool and --domain-matched are mutually exclusive: "
            "the first builds ONE shared encoder, the second builds N."
        )

    if args.combined_pool:
        pool_spec = parse_combined_pool_spec(args.combined_pool)
        print(f"Pretraining ONE encoder on combined pool: {pool_spec}")
        shared_enc = pretrain_on_combined_pool(
            pool_spec, config, device, args.output,
            target_shape=tuple(args.combined_pool_shape),
            samples_per_epoch=args.combined_pool_samples_per_epoch,
        )
        encoders = {b: shared_enc for b in args.benchmarks}
        # darcy_encoder is used downstream for cross-domain comparisons; in
        # combined-pool mode the "cross-domain" baseline IS the shared
        # encoder, so we alias it.
        darcy_encoder = shared_enc
    elif args.domain_matched:
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

    # CCSNet — single name "ccsnet" defaults to SG target; or specify
    # individual variants like ccsnet_sg, ccsnet_bpr, ccsnet_bxmf, ccsnet_bymf,
    # ccsnet_bdenw, ccsnet_bdeng, ccsnet_p_init for separate prediction tasks.
    ccsnet_benchmarks = [b for b in args.benchmarks
                         if b == "ccsnet" or b.startswith("ccsnet_")]
    for bname in ccsnet_benchmarks:
        if bname == "ccsnet":
            tv = "SG"
        else:
            tv = bname[len("ccsnet_"):].upper()
            if tv == "P_INIT":
                tv = "P_init"
        tr, te, ic, oc = load_ccsnet(target_var=tv)
        all_results[bname] = run_benchmark(
            bname, config, encoders[bname], tr, te, device,
            ic, oc, darcy_nl, baseline_list, args.n_seeds, args.output)
        if args.domain_matched and encoders.get(bname) is not darcy_encoder:
            print(f"\n--- Cross-domain: Darcy-pretrained on {bname} ---")
            all_results[f"{bname}_crossdomain"] = run_benchmark(
                f"{bname}_crossdomain", config, darcy_encoder, tr, te, device,
                ic, oc, darcy_nl, [], args.n_seeds, args.output)

    if "fno4co2" in args.benchmarks:
        tr, te, ic, oc = load_fno4co2()
        all_results["fno4co2"] = run_benchmark(
            "fno4co2", config, encoders["fno4co2"], tr, te, device,
            ic, oc, darcy_nl, baseline_list, args.n_seeds, args.output)
        if args.domain_matched and encoders.get("fno4co2") is not darcy_encoder:
            print("\n--- Cross-domain comparison: Darcy-pretrained on FNO4CO2 ---")
            all_results["fno4co2_crossdomain"] = run_benchmark(
                "fno4co2_crossdomain", config, darcy_encoder, tr, te, device,
                ic, oc, darcy_nl, [], args.n_seeds, args.output)

    if "pdebench_adr" in args.benchmarks:
        tr, te, ic, oc = load_pdebench_adr()
        all_results["pdebench_adr"] = run_benchmark(
            "pdebench_adr", config, encoders["pdebench_adr"], tr, te, device,
            ic, oc, adr_nl, baseline_list, args.n_seeds, args.output)
        if args.domain_matched and encoders.get("pdebench_adr") is not darcy_encoder:
            print("\n--- Cross-domain comparison: Darcy-pretrained on PDEBench ADR ---")
            all_results["pdebench_adr_crossdomain"] = run_benchmark(
                "pdebench_adr_crossdomain", config, darcy_encoder, tr, te, device,
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
