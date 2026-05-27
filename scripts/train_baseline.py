#!/usr/bin/env python
"""
Train a baseline operator network on a labeled (x, y) dataset and emit
metrics. Used for the PI-JEPA paper's baseline comparison tables.

Currently supports: fno3d (vanilla 3D FNO). The runner is structured so
adding other baselines (U-FNO, DeepONet3D, etc.) is a one-class addition.

Usage:
    python scripts/train_baseline.py \
        --baseline fno3d \
        --dataset ccsnet \
        --train-x data/ccsnet/CCSNet_v1.0/train_x.hdf5 \
        --train-y data/ccsnet/CCSNet_v1.0/train_y_SG.hdf5 \
        --test-x data/ccsnet/CCSNet_v1.0/test_x.hdf5 \
        --test-y data/ccsnet/CCSNet_v1.0/test_y_SG.hdf5 \
        --n-labeled 100 --epochs 50 --batch-size 4 \
        --output outputs_baselines/fno3d_ccsnet_n100

For synthetic 3D Darcy:
    python scripts/train_baseline.py \
        --baseline fno3d \
        --dataset darcy_3d_pt \
        --train-pt data/darcy_3d/darcy3d_train.pt \
        --test-pt  data/darcy_3d/darcy3d_test.pt \
        --n-labeled 32 --epochs 5 --output outputs_baselines/fno3d_darcy_3d
"""

import argparse
import json
import os
import sys
import time
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# Make PI-JEPA package importable when invoked from repo root
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PI-JEPA"))

from benchmarks.fno_3d import FNO3D
from benchmarks.pino_3d import PINO3D
from benchmarks.ufno_3d import UFNO3D
from benchmarks.pi_deeponet_3d import DeepONet3D, PIDeepONet3D
from eval.paper_metrics import (
    relative_l2, nrmse, max_err, conservation_residual, bootstrap_ci_95,
)


def _resize_cube_5d(t: torch.Tensor, side: int) -> torch.Tensor:
    """Trilinear-resize a (N, C, D, H, W) tensor to a (N, C, side, side, side)
    cube. Used to fit Brandon's cubic-only fourier_encoder_3d AND to give
    every baseline the same input shape PI-JEPA sees."""
    if t.dim() != 5:
        raise ValueError(f"_resize_cube_5d expects 5D; got {tuple(t.shape)}")
    if t.shape[-3:] == (side, side, side):
        return t
    return F.interpolate(t, size=(side, side, side),
                         mode="trilinear", align_corners=False)


def load_pt_dataset(pt_path: str, n_samples: int = None):
    """Load a synthetic Darcy .pt dataset with x and y keys."""
    blob = torch.load(pt_path, weights_only=False, map_location="cpu")
    x = blob["x"].float()
    y = blob["y"].float()
    if n_samples is not None:
        x = x[:n_samples]
        y = y[:n_samples]
    return x, y


def load_ccsnet_pair(x_path: str, y_path: str, t_index: int = -1,
                     n_samples: int = None, layout: str = "ctxy"):
    """Load CCSNet (x, y) pair using the project's CCSNet loaders.

    Returns 5D tensors (N, C, T, H, W) by default. t_index=-1 means use the
    final timestep as the supervised target.
    """
    from data.ccsnet_loader import _read_ccsnet_array
    x_raw = _read_ccsnet_array(x_path)  # (N, H, W, T, C)
    y_raw = _read_ccsnet_array(y_path)  # (N, H, W, T, C) — or maybe (N, H, W, T) depending on output
    if y_raw.ndim == 4:
        y_raw = y_raw[..., None]

    # Permute to (N, C, T, H, W)
    x = np.transpose(x_raw, (0, 4, 3, 1, 2))
    y = np.transpose(y_raw, (0, 4, 3, 1, 2))

    if n_samples is not None:
        x = x[:n_samples]
        y = y[:n_samples]

    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()

    # Normalize per channel
    x_mean = x_t.mean(dim=(0, 2, 3, 4), keepdim=True)
    x_std = x_t.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
    y_mean = y_t.mean(dim=(0, 2, 3, 4), keepdim=True)
    y_std = y_t.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
    x_t = (x_t - x_mean) / x_std
    y_t = (y_t - y_mean) / y_std

    return x_t, y_t


def build_baseline(baseline_name: str, in_channels: int, out_channels: int,
                   volume_shape=(64, 64, 64),
                   modes=(8, 8, 8), hidden_channels=32, n_blocks=4,
                   physics_weight: float = 0.1,
                   residual_type: str = "fd") -> nn.Module:
    name = baseline_name.lower()
    if name == "fno3d":
        return FNO3D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
            modes=modes,
        )
    if name == "fno3d_large":
        # Size-matched ~150M-param FNO3D for the reviewer M4 confound.
        # Opt in by name; default grid drops it.
        return FNO3D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=192,
            n_blocks=6,
            modes=modes,
        )
    if name == "ufno3d":
        return UFNO3D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
            modes=modes,
        )
    if name == "pino3d":
        return PINO3D(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            n_blocks=n_blocks,
            modes=modes,
            physics_weight=physics_weight,
            residual_type=residual_type,
        )
    if name == "deeponet3d":
        return DeepONet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            volume_shape=tuple(volume_shape),
        )
    if name == "pi_deeponet3d":
        return PIDeepONet3D(
            in_channels=in_channels,
            out_channels=out_channels,
            volume_shape=tuple(volume_shape),
            physics_weight=physics_weight,
            residual_type=residual_type,
        )
    raise ValueError(f"Unknown baseline: {baseline_name}")


def train_supervised(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
) -> Dict:
    """Standard supervised training loop. Returns final metrics dict."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # If model exposes a `physics_loss(x, pred) -> scalar` (PINO3D),
    # add it to the training loss with `model.physics_weight`.
    has_physics = hasattr(model, "physics_loss")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_phys = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            if pred.shape[1] != y.shape[1]:
                pred = pred[:, :y.shape[1]]
            sup_loss = F.mse_loss(pred, y)
            loss = sup_loss
            if has_physics:
                phys = model.physics_loss(x, pred)
                loss = loss + float(model.physics_weight) * phys
                epoch_phys += float(phys.item())
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            log_msg = f"  baseline epoch {epoch+1}/{epochs} loss={epoch_loss/len(train_loader):.4f}"
            if has_physics:
                log_msg += f" phys={epoch_phys/len(train_loader):.4f}"
            print(log_msg)

    # Eval
    model.eval()
    rel_l2_list = []
    nrmse_list = []
    maxerr_list = []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if pred.shape[1] != y.shape[1]:
                pred = pred[:, :y.shape[1]]
            rel_l2_list.append(relative_l2(pred, y).cpu().numpy())
            nrmse_list.append(nrmse(pred, y).cpu().numpy())
            maxerr_list.append(max_err(pred, y).cpu().numpy())

    rl2 = np.concatenate(rel_l2_list)
    nrm = np.concatenate(nrmse_list)
    mxe = np.concatenate(maxerr_list)

    rl2_mean, rl2_lo, rl2_hi = bootstrap_ci_95(rl2)
    nrm_mean, nrm_lo, nrm_hi = bootstrap_ci_95(nrm)
    mxe_mean, mxe_lo, mxe_hi = bootstrap_ci_95(mxe)

    return {
        "relative_l2_mean": rl2_mean,
        "relative_l2_ci_low": rl2_lo,
        "relative_l2_ci_high": rl2_hi,
        "nrmse_mean": nrm_mean,
        "nrmse_ci_low": nrm_lo,
        "nrmse_ci_high": nrm_hi,
        "max_err_mean": mxe_mean,
        "max_err_ci_low": mxe_lo,
        "max_err_ci_high": mxe_hi,
        "n_test": len(rl2),
    }


def main():
    ap = argparse.ArgumentParser(description="Train a baseline operator network")
    ap.add_argument("--baseline", required=True,
                    choices=["fno3d", "fno3d_large", "ufno3d", "pino3d",
                             "deeponet3d", "pi_deeponet3d"])
    ap.add_argument("--dataset", required=True, choices=["darcy_3d_pt", "ccsnet"])
    ap.add_argument("--resize-cube", type=int, default=64,
                    help="Resize all input/output volumes to (N, N, N). "
                    "Required because Brandon's fourier_encoder_3d is "
                    "cubic-only; matches the cube PI-JEPA finetune sees. "
                    "Set 0 to skip.")
    # Synthetic Darcy: --train-pt + --test-pt
    ap.add_argument("--train-pt", type=str, default=None)
    ap.add_argument("--test-pt", type=str, default=None)
    # CCSNet: --train-x + --train-y + --test-x + --test-y
    ap.add_argument("--train-x", type=str, default=None)
    ap.add_argument("--train-y", type=str, default=None)
    ap.add_argument("--test-x", type=str, default=None)
    ap.add_argument("--test-y", type=str, default=None)
    # Common
    ap.add_argument("--n-labeled", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-channels", type=int, default=32)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument("--modes", type=int, nargs=3, default=[8, 8, 8])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)

    # Resolve dataset
    if args.dataset == "darcy_3d_pt":
        assert args.train_pt and args.test_pt, "Need --train-pt and --test-pt for darcy_3d_pt"
        x_tr, y_tr = load_pt_dataset(args.train_pt, n_samples=args.n_labeled)
        x_te, y_te = load_pt_dataset(args.test_pt, n_samples=args.n_test)
    elif args.dataset == "ccsnet":
        assert args.train_x and args.train_y and args.test_x and args.test_y, \
            "Need --train-x/y and --test-x/y for ccsnet"
        x_tr, y_tr = load_ccsnet_pair(args.train_x, args.train_y, n_samples=args.n_labeled)
        x_te, y_te = load_ccsnet_pair(args.test_x, args.test_y, n_samples=args.n_test)
    else:
        raise ValueError(args.dataset)

    # Cubic resize (Brandon's fourier_encoder_3d is cubic-only; we apply
    # the same resize to baselines so the input shape is apples-to-apples).
    if args.resize_cube and args.resize_cube > 0:
        side = int(args.resize_cube)
        x_tr = _resize_cube_5d(x_tr, side)
        y_tr = _resize_cube_5d(y_tr, side)
        x_te = _resize_cube_5d(x_te, side)
        y_te = _resize_cube_5d(y_te, side)
        print(f"resized to {side}^3 cube")

    print(f"train shapes: x={tuple(x_tr.shape)}, y={tuple(y_tr.shape)}")
    print(f"test  shapes: x={tuple(x_te.shape)}, y={tuple(y_te.shape)}")

    in_channels = x_tr.shape[1]
    out_channels = y_tr.shape[1]
    volume_shape = tuple(x_tr.shape[-3:])

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(x_te, y_te),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"device: {device}")

    model = build_baseline(
        args.baseline,
        in_channels=in_channels,
        out_channels=out_channels,
        volume_shape=volume_shape,
        modes=tuple(args.modes),
        hidden_channels=args.hidden_channels,
        n_blocks=args.n_blocks,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.baseline} params: {n_params:,}")

    # Peak-memory + inference-latency disclosure (reviewer qZsm M4).
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    metrics = train_supervised(
        model, train_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr,
    )
    dt = time.time() - t0

    # Inference latency: timed single forward on a 1-batch test sample.
    model.eval()
    inf_lat_ms = None
    try:
        with torch.no_grad():
            sample_x = x_te[:1].to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_inf0 = time.time()
            _ = model(sample_x[:, :in_channels])
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inf_lat_ms = (time.time() - t_inf0) * 1000.0
    except Exception as e:
        print(f"[warn] inference-latency measurement failed: {e}")

    peak_mem_mb = None
    if device.type == "cuda":
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    metrics["wall_clock_seconds"] = dt
    metrics["param_count"] = int(n_params)
    metrics["inference_latency_ms_batch1"] = inf_lat_ms
    metrics["peak_gpu_memory_mb"] = peak_mem_mb
    metrics["baseline"] = args.baseline
    metrics["dataset"] = args.dataset
    metrics["n_labeled"] = args.n_labeled
    metrics["epochs"] = args.epochs
    metrics["seed"] = args.seed
    metrics["volume_shape"] = list(volume_shape)
    metrics["in_channels"] = int(in_channels)
    metrics["out_channels"] = int(out_channels)

    out_json = os.path.join(args.output, "baseline_result.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {out_json}")
    print(f"rel_L2: {metrics['relative_l2_mean']:.4f} "
          f"[{metrics['relative_l2_ci_low']:.4f}, {metrics['relative_l2_ci_high']:.4f}] "
          f"(n_test={metrics['n_test']}, wall={dt:.1f}s)")


if __name__ == "__main__":
    main()
