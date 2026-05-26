#!/usr/bin/env python
"""
Fine-tune a pretrained PI-JEPA encoder on labeled (x, y) pairs.

Loads a pretrain checkpoint, attaches a small per-stage decoder (or reuses
the saved per-stage decoders), and trains on a small labeled corpus.
Emits the same metric JSON schema as scripts/train_baseline.py so the two
are directly comparable head-to-head in the paper's main results table.

Usage:
    python scripts/finetune_pijepa.py \
        --pretrain-checkpoint outputs_3d_mf/pretrain/checkpoint_final.pt \
        --pretrain-config configs/darcy_3d_mf_smoke.yaml \
        --dataset darcy_3d_pt \
        --train-pt data/darcy_3d/darcy3d_train.pt \
        --test-pt data/darcy_3d/darcy3d_test.pt \
        --n-labeled 32 --epochs 20 --batch-size 4 \
        --output outputs_finetune/pijepa_darcy_3d_n32
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "PI-JEPA"))

import yaml
from models import build_encoder, Decoder, Decoder3D, PIJEPA, Predictor
from eval.paper_metrics import (
    relative_l2, nrmse, max_err, bootstrap_ci_95,
)


def load_pt_dataset(pt_path: str, n_samples: int = None):
    blob = torch.load(pt_path, weights_only=False, map_location="cpu")
    x = blob["x"].float()
    y = blob["y"].float()
    if n_samples is not None:
        x = x[:n_samples]
        y = y[:n_samples]
    return x, y


def load_ccsnet_finetune(
    x_path: str,
    y_path: str,
    n_samples: int = None,
    layout: str = "ctxy",
    resize_to=None,
):
    """Load CCSNet (input, target) pair for fine-tuning.

    Both x and y are reshaped to (N, C, T, H, W). Channel-aware normalization.
    """
    from data.ccsnet_loader import _read_ccsnet_array
    x = _read_ccsnet_array(x_path)  # (N, H, W, T, C)
    y = _read_ccsnet_array(y_path)
    if y.ndim == 4:
        y = y[..., None]

    # Permute to (N, C, T, H, W)
    x = np.transpose(x, (0, 4, 3, 1, 2))
    y = np.transpose(y, (0, 4, 3, 1, 2))
    if n_samples is not None:
        x = x[:n_samples]
        y = y[:n_samples]
    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).float()

    if resize_to is not None:
        # Resize the spatial axes (last two), preserve C and T.
        N, C, T, H, W = x_t.shape
        x_flat = x_t.reshape(N, C * T, H, W)
        x_t = F.interpolate(x_flat, size=resize_to, mode="bilinear", align_corners=False).reshape(N, C, T, *resize_to)
        N2, C2, T2, H2, W2 = y_t.shape
        y_flat = y_t.reshape(N2, C2 * T2, H2, W2)
        y_t = F.interpolate(y_flat, size=resize_to, mode="bilinear", align_corners=False).reshape(N2, C2, T2, *resize_to)

    # Per-channel normalize across (N, T, H, W)
    x_mean = x_t.mean(dim=(0, 2, 3, 4), keepdim=True)
    x_std = x_t.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
    y_mean = y_t.mean(dim=(0, 2, 3, 4), keepdim=True)
    y_std = y_t.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
    x_t = (x_t - x_mean) / x_std
    y_t = (y_t - y_mean) / y_std
    return x_t, y_t


def load_fno4co2_finetune(
    a_path: str, u_path: str,
    n_samples: int = None,
    keep_channels=None,
    resize_to=None,
):
    """Load FNO4CO2 (input, target) pair for fine-tuning."""
    a = torch.load(a_path, weights_only=False, map_location="cpu").float()  # (N,H,W,T,C)
    u = torch.load(u_path, weights_only=False, map_location="cpu").float()  # (N,H,W,T)
    if keep_channels is not None:
        a = a[..., list(keep_channels)]
    if n_samples is not None:
        a = a[:n_samples]
        u = u[:n_samples]

    # Permute to channel-first time-preserving
    a = a.permute(0, 4, 3, 1, 2).contiguous()   # (N, C, T, H, W)
    u = u.permute(0, 3, 1, 2).unsqueeze(1).contiguous()  # (N, 1, T, H, W)

    if resize_to is not None:
        N, C, T, H, W = a.shape
        a = F.interpolate(a.reshape(N, C * T, H, W), size=resize_to, mode="bilinear", align_corners=False).reshape(N, C, T, *resize_to)
        N2, _, T2, H2, W2 = u.shape
        u = F.interpolate(u.reshape(N2, T2, H2, W2), size=resize_to, mode="bilinear", align_corners=False).reshape(N2, 1, T2, *resize_to)

    a_mean = a.mean(dim=(0, 2, 3, 4), keepdim=True)
    a_std = a.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
    u_mean = u.mean(dim=(0, 2, 3, 4), keepdim=True)
    u_std = u.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
    a = (a - a_mean) / a_std
    u = (u - u_mean) / u_std
    return a, u


def build_pijepa_random_init(config: dict, device: torch.device):
    """Build a freshly-initialized PIJEPA + per-stage decoders (no pretrain).

    This is the "PI-JEPA from scratch" baseline — same architecture, same
    fine-tuning loop, but no self-supervised pretraining. It directly tests
    whether the pretraining is what's helping, vs the operator-split
    architecture alone.
    """
    encoder = build_encoder(config, in_channels=1).to(device)
    target_encoder = build_encoder(config, in_channels=1).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.load_state_dict(encoder.state_dict())

    K = config["model"]["num_predictors"]
    predictors = [Predictor(config).to(device) for _ in range(K)]

    model = PIJEPA(
        encoder=encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        embed_dim=config["model"]["encoder"]["embed_dim"],
        patch_size=config["model"]["encoder"]["patch_size"],
    ).to(device)

    decoder_cfg = config.get("decoder", {})
    encoder_type = config.get("model", {}).get("encoder", {}).get("type", "vit").lower()
    is_3d = encoder_type in ("fourier_3d", "fourier3d")

    def _make_decoder():
        cls = Decoder3D if is_3d else Decoder
        return cls(
            embed_dim=decoder_cfg.get("embed_dim", config["model"]["encoder"]["embed_dim"]),
            out_channels=decoder_cfg.get("out_channels", 1),
            image_size=decoder_cfg.get("image_size", config["model"]["encoder"]["image_size"]),
            patch_size=decoder_cfg.get("patch_size", config["model"]["encoder"]["patch_size"]),
        )

    per_stage = bool(decoder_cfg.get("per_stage", True)) and K > 1
    if per_stage:
        decoders = nn.ModuleList([_make_decoder() for _ in range(K)]).to(device)
    else:
        decoders = nn.ModuleList([_make_decoder()]).to(device)

    return model, decoders


def restore_pijepa_from_checkpoint(
    checkpoint_path: str,
    config: dict,
    device: torch.device,
):
    """Reconstruct PIJEPA + per-stage decoders from a saved pretrain ckpt."""
    blob = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    encoder = build_encoder(config, in_channels=1).to(device)
    target_encoder = build_encoder(config, in_channels=1).to(device)
    for p in target_encoder.parameters():
        p.requires_grad = False

    encoder.load_state_dict(blob["encoder_state_dict"])
    target_encoder.load_state_dict(blob["target_encoder_state_dict"])

    K = config["model"]["num_predictors"]
    predictors = [Predictor(config).to(device) for _ in range(K)]
    for p, sd in zip(predictors, blob["predictor_state_dicts"]):
        p.load_state_dict(sd)

    model = PIJEPA(
        encoder=encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        embed_dim=config["model"]["encoder"]["embed_dim"],
        patch_size=config["model"]["encoder"]["patch_size"],
    ).to(device)

    # Load per-stage decoders if saved, else fall back to legacy single decoder
    decoder_cfg = config.get("decoder", {})
    encoder_type = config.get("model", {}).get("encoder", {}).get("type", "vit").lower()
    is_3d = encoder_type in ("fourier_3d", "fourier3d")

    def _make_decoder():
        if is_3d:
            return Decoder3D(
                embed_dim=decoder_cfg.get("embed_dim", config["model"]["encoder"]["embed_dim"]),
                out_channels=decoder_cfg.get("out_channels", 1),
                image_size=decoder_cfg.get("image_size", config["model"]["encoder"]["image_size"]),
                patch_size=decoder_cfg.get("patch_size", config["model"]["encoder"]["patch_size"]),
            )
        return Decoder(
            embed_dim=decoder_cfg.get("embed_dim", config["model"]["encoder"]["embed_dim"]),
            out_channels=decoder_cfg.get("out_channels", 1),
            image_size=decoder_cfg.get("image_size", config["model"]["encoder"]["image_size"]),
            patch_size=decoder_cfg.get("patch_size", config["model"]["encoder"]["patch_size"]),
        )

    if "decoder_state_dicts" in blob:
        decoders = nn.ModuleList([_make_decoder() for _ in range(len(blob["decoder_state_dicts"]))]).to(device)
        for d, sd in zip(decoders, blob["decoder_state_dicts"]):
            d.load_state_dict(sd)
    else:
        d = _make_decoder().to(device)
        d.load_state_dict(blob["decoder_state_dict"])
        decoders = nn.ModuleList([d]).to(device)

    return model, decoders


class PIJEPAFinetuner(nn.Module):
    """Wraps a pretrained PI-JEPA + decoders into a forward(x) -> y network.

    Uses the LAST per-stage decoder as the prediction head and runs the
    full Lie-Trotter chain. Encoder is fine-tunable; per-stage decoders
    are fine-tunable; target encoder is frozen (only used to pump EMA
    targets during pretraining — at finetune time we don't refresh it).
    """

    def __init__(self, pijepa: PIJEPA, decoders: nn.ModuleList):
        super().__init__()
        self.pijepa = pijepa
        self.decoders = decoders

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode full input
        z_full = self.pijepa.encoder(x)
        # All patches are "context" at finetune time (no masking)
        B, N, D = z_full.shape
        idx_all = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)
        # Run the operator-split chain on the full latent
        z_t = self.pijepa.mask_token.expand(B, N, D).contiguous()
        # context = the encoded representation; target = all positions
        z_context = z_full
        for predictor in self.pijepa.predictors:
            z_t = predictor.forward_chained(z_t, z_context)
        # Decode the final stage
        decoded = self.decoders[-1](z_t)
        return decoded


def train_finetune(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr_head: float,
    lr_encoder: float,
    freeze_encoder: bool,
):
    """Fine-tune the model. encoder and decoders are optimized at different LRs."""
    model.to(device)

    # Build parameter groups: encoder gets a SMALLER lr, decoders get the head lr.
    encoder_params = list(model.pijepa.encoder.parameters()) + list(model.pijepa.predictors.parameters())
    head_params = list(model.decoders.parameters())

    if freeze_encoder:
        for p in encoder_params:
            p.requires_grad = False
        param_groups = [{"params": head_params, "lr": lr_head}]
    else:
        param_groups = [
            {"params": head_params, "lr": lr_head},
            {"params": encoder_params, "lr": lr_encoder},
        ]
    opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = model(x)
            if pred.shape[1] != y.shape[1]:
                pred = pred[:, :y.shape[1]]
            loss = F.mse_loss(pred, y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
        sched.step()
        if (epoch + 1) % max(1, epochs // 5) == 0 or epoch == 0:
            print(f"  finetune epoch {epoch+1}/{epochs} loss={epoch_loss/len(train_loader):.4f}")

    model.eval()
    rl2, nrm, mxe = [], [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            if pred.shape[1] != y.shape[1]:
                pred = pred[:, :y.shape[1]]
            rl2.append(relative_l2(pred, y).cpu().numpy())
            nrm.append(nrmse(pred, y).cpu().numpy())
            mxe.append(max_err(pred, y).cpu().numpy())

    rl2 = np.concatenate(rl2)
    nrm = np.concatenate(nrm)
    mxe = np.concatenate(mxe)
    rl2_m, rl2_lo, rl2_hi = bootstrap_ci_95(rl2)
    nrm_m, nrm_lo, nrm_hi = bootstrap_ci_95(nrm)
    mxe_m, mxe_lo, mxe_hi = bootstrap_ci_95(mxe)
    return {
        "relative_l2_mean": rl2_m,
        "relative_l2_ci_low": rl2_lo,
        "relative_l2_ci_high": rl2_hi,
        "nrmse_mean": nrm_m,
        "nrmse_ci_low": nrm_lo,
        "nrmse_ci_high": nrm_hi,
        "max_err_mean": mxe_m,
        "max_err_ci_low": mxe_lo,
        "max_err_ci_high": mxe_hi,
        "n_test": len(rl2),
    }


def main():
    ap = argparse.ArgumentParser(description="Fine-tune pretrained PI-JEPA on labeled data")
    ap.add_argument("--pretrain-checkpoint", default=None,
                    help="Required unless --from-scratch is set")
    ap.add_argument("--pretrain-config", required=True,
                    help="Architecture/config file — needed even with --from-scratch")
    ap.add_argument("--from-scratch", action="store_true",
                    help="Skip pretrain checkpoint; randomly initialize PI-JEPA + decoders. "
                         "Lets us ablate 'pretraining helps' vs 'architecture helps'.")
    ap.add_argument("--dataset", default="darcy_3d_pt",
                    choices=["darcy_3d_pt", "ccsnet", "fno4co2"])
    ap.add_argument("--train-pt", type=str, default=None)
    ap.add_argument("--test-pt", type=str, default=None)
    # CCSNet / FNO4CO2 specific
    ap.add_argument("--train-x", type=str, default=None)
    ap.add_argument("--train-y", type=str, default=None)
    ap.add_argument("--test-x", type=str, default=None)
    ap.add_argument("--test-y", type=str, default=None)
    ap.add_argument("--resize-to", type=int, nargs=2, default=None,
                    help="Optional (H, W) resize for CCSNet/FNO4CO2")
    ap.add_argument("--fno4co2-variant", type=str, default="dP",
                    choices=["dP", "sg"])
    ap.add_argument("--n-labeled", type=int, default=32)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-encoder", type=float, default=1e-4)
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    torch.manual_seed(args.seed)

    with open(args.pretrain_config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    # Load data
    resize = tuple(args.resize_to) if args.resize_to else None
    if args.dataset == "darcy_3d_pt":
        assert args.train_pt and args.test_pt
        x_tr, y_tr = load_pt_dataset(args.train_pt, n_samples=args.n_labeled)
        x_te, y_te = load_pt_dataset(args.test_pt, n_samples=args.n_test)
    elif args.dataset == "ccsnet":
        assert args.train_x and args.train_y and args.test_x and args.test_y, \
            "ccsnet needs --train-x/y and --test-x/y"
        x_tr, y_tr = load_ccsnet_finetune(args.train_x, args.train_y,
                                          n_samples=args.n_labeled, resize_to=resize)
        x_te, y_te = load_ccsnet_finetune(args.test_x, args.test_y,
                                          n_samples=args.n_test, resize_to=resize)
    elif args.dataset == "fno4co2":
        assert args.train_x and args.train_y and args.test_x and args.test_y, \
            "fno4co2 needs --train-x/y and --test-x/y (paths to a/u .pt files)"
        x_tr, y_tr = load_fno4co2_finetune(args.train_x, args.train_y,
                                           n_samples=args.n_labeled, resize_to=resize)
        x_te, y_te = load_fno4co2_finetune(args.test_x, args.test_y,
                                           n_samples=args.n_test, resize_to=resize)
    else:
        raise ValueError(args.dataset)

    print(f"train shapes: x={tuple(x_tr.shape)}, y={tuple(y_tr.shape)}")
    print(f"test  shapes: x={tuple(x_te.shape)}, y={tuple(y_te.shape)}")

    if args.from_scratch:
        pijepa, decoders = build_pijepa_random_init(config, device)
        print("[from_scratch] PI-JEPA initialized randomly (no pretrain)")
    else:
        if args.pretrain_checkpoint is None:
            raise SystemExit("--pretrain-checkpoint required unless --from-scratch is set")
        pijepa, decoders = restore_pijepa_from_checkpoint(
            args.pretrain_checkpoint, config, device
        )
    model = PIJEPAFinetuner(pijepa, decoders)
    print(f"PI-JEPA finetuner params: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = DataLoader(
        TensorDataset(x_tr, y_tr),
        batch_size=args.batch_size, shuffle=True, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(x_te, y_te),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    t0 = time.time()
    metrics = train_finetune(
        model, train_loader, test_loader, device,
        epochs=args.epochs, lr_head=args.lr_head, lr_encoder=args.lr_encoder,
        freeze_encoder=args.freeze_encoder,
    )
    dt = time.time() - t0
    metrics["wall_clock_seconds"] = dt
    metrics["method"] = "pi_jepa_from_scratch" if args.from_scratch else "pi_jepa_finetuned"
    metrics["dataset"] = args.dataset
    metrics["n_labeled"] = args.n_labeled
    metrics["epochs"] = args.epochs
    metrics["seed"] = args.seed
    metrics["pretrain_checkpoint"] = None if args.from_scratch else args.pretrain_checkpoint

    out_json = os.path.join(args.output, "pijepa_result.json")
    with open(out_json, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nWrote {out_json}")
    print(f"rel_L2: {metrics['relative_l2_mean']:.4f} "
          f"[{metrics['relative_l2_ci_low']:.4f}, {metrics['relative_l2_ci_high']:.4f}] "
          f"(n_test={metrics['n_test']}, wall={dt:.1f}s)")


if __name__ == "__main__":
    main()
