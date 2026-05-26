"""
FNO4CO2 / U-FNO data loader.

The U-FNO dataset (Wen et al. 2022, Advances in Water Resources) ships PyTorch
.pt tensors:
    - `dP_<split>_a.pt` -> (N, H, W, T, C_in)   multi-channel input
    - `dP_<split>_u.pt` -> (N, H, W, T)         pressure-buildup output
    - `sg_<split>_a.pt` -> (N, H, W, T, C_in)   same input format
    - `sg_<split>_u.pt` -> (N, H, W, T)         gas-saturation output

C_in is typically 12 channels (permeability, porosity, depth, injection
schedule, BC indicators, etc.). H=96, W=200, T=24, N up to 4500 for train.

This module mirrors PI-JEPA/data/ccsnet_loader.py: yields {"x": tensor}
for unlabeled pretraining or (x, y) tuples for supervised fine-tuning.
Supports both:
    - 2D collapsed-time mode (t_index=<int>): emits (C, H, W)
    - 3D time-preserving mode (t_index=None, layout="ctxy"): emits (C, T, H, W)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


def _load_pt(path: str) -> torch.Tensor:
    """Load a .pt file. The U-FNO files are raw tensors, not state dicts."""
    blob = torch.load(path, weights_only=False, map_location="cpu")
    if not isinstance(blob, torch.Tensor):
        raise TypeError(
            f"{path}: expected torch.Tensor, got {type(blob).__name__}"
        )
    return blob.float()


class FNO4CO2UnlabeledDataset(Dataset):
    """U-FNO input fields, optionally with time preserved.

    The U-FNO 'a' tensor is shape (N, H, W, T, C). We permute to PyTorch
    channel-first layout: either (N, C, H, W) for time-collapsed mode or
    (N, C, T, H, W) for time-preserving mode.
    """

    def __init__(
        self,
        a_path: str,
        n_samples: Optional[int] = None,
        t_index: Optional[int] = None,
        normalize: bool = True,
        resize_to: Optional[Tuple[int, int]] = None,
        layout: str = "ctxy",
        keep_channels: Optional[Tuple[int, ...]] = None,
    ):
        """
        Args:
            a_path: path to dP_train_a.pt / sg_test_a.pt / etc.
            n_samples: cap on number of samples (None = all)
            t_index: if int, take that single timestep and emit 4D (C, H, W).
                If None (default), keep the full time axis and emit 5D per
                `layout`.
            normalize: zero-mean, unit-std per channel across the dataset
            resize_to: optional (H, W) resize via bilinear interpolation;
                the time and channel axes are preserved.
            layout: 5D output ordering. "ctxy" -> (C, T, H, W) (U-FNO 3D-conv
                convention); "tchw" -> (T, C, H, W).
            keep_channels: optional list of channel indices to retain from
                the input C dim. Default keeps all C channels.
        """
        super().__init__()
        if not Path(a_path).exists():
            raise FileNotFoundError(f"FNO4CO2 a file not found: {a_path}")

        a = _load_pt(a_path)  # (N, H, W, T, C)
        if a.dim() != 5:
            raise ValueError(
                f"{a_path}: expected 5D (N,H,W,T,C); got {tuple(a.shape)}"
            )

        if keep_channels is not None:
            a = a[..., list(keep_channels)]

        if n_samples is not None:
            a = a[:n_samples]

        if t_index is not None:
            # (N, H, W, T, C) -> take timestep -> (N, H, W, C) -> (N, C, H, W)
            a = a[:, :, :, t_index, :]
            a = a.permute(0, 3, 1, 2).contiguous()
            if resize_to is not None:
                a = torch.nn.functional.interpolate(
                    a, size=resize_to, mode="bilinear", align_corners=False
                )
            if normalize:
                mean = a.mean(dim=(0, 2, 3), keepdim=True)
                std = a.std(dim=(0, 2, 3), keepdim=True) + 1e-8
                a = (a - mean) / std
        else:
            # Keep time. Permute to the chosen layout.
            if layout == "ctxy":
                # (N, H, W, T, C) -> (N, C, T, H, W)
                a = a.permute(0, 4, 3, 1, 2).contiguous()
            elif layout == "tchw":
                # (N, H, W, T, C) -> (N, T, C, H, W)
                a = a.permute(0, 3, 4, 1, 2).contiguous()
            else:
                raise ValueError(f"Unknown layout: {layout!r}")

            if resize_to is not None:
                if a.dim() == 5:
                    N, C, T, H, W = a.shape
                    a_flat = a.reshape(N, C * T, H, W)
                    a_flat = torch.nn.functional.interpolate(
                        a_flat, size=resize_to, mode="bilinear", align_corners=False
                    )
                    a = a_flat.reshape(N, C, T, *resize_to)

            if normalize:
                # Normalize across (N, T, H, W); preserve channel separation
                # because different channels have very different scales (e.g.,
                # permeability vs. boundary indicator).
                if a.dim() == 5:
                    mean = a.mean(dim=(0, 2, 3, 4), keepdim=True)
                    std = a.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
                else:
                    mean = a.mean(dim=(0, 2, 3), keepdim=True)
                    std = a.std(dim=(0, 2, 3), keepdim=True) + 1e-8
                a = (a - mean) / std

        self.x = a

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return {"x": self.x[i]}


class FNO4CO2LabeledDataset(Dataset):
    """Paired (input, target) U-FNO samples for supervised fine-tuning."""

    def __init__(
        self,
        a_path: str,
        u_path: str,
        n_samples: Optional[int] = None,
        t_index: Optional[int] = None,
        normalize: bool = True,
        layout: str = "ctxy",
        keep_channels: Optional[Tuple[int, ...]] = None,
    ):
        super().__init__()
        if not Path(a_path).exists():
            raise FileNotFoundError(f"FNO4CO2 a file not found: {a_path}")
        if not Path(u_path).exists():
            raise FileNotFoundError(f"FNO4CO2 u file not found: {u_path}")

        a = _load_pt(a_path)  # (N, H, W, T, C)
        u = _load_pt(u_path)  # (N, H, W, T)

        if a.dim() != 5 or u.dim() != 4:
            raise ValueError(
                f"Expected a (N,H,W,T,C) and u (N,H,W,T); got a={a.shape}, u={u.shape}"
            )

        if keep_channels is not None:
            a = a[..., list(keep_channels)]

        if n_samples is not None:
            a = a[:n_samples]
            u = u[:n_samples]

        if t_index is not None:
            # Collapse time
            a = a[:, :, :, t_index, :].permute(0, 3, 1, 2).contiguous()       # (N, C, H, W)
            u = u[:, :, :, t_index].unsqueeze(1).contiguous()                  # (N, 1, H, W)
        else:
            if layout == "ctxy":
                a = a.permute(0, 4, 3, 1, 2).contiguous()  # (N, C, T, H, W)
                u = u.permute(0, 3, 1, 2).unsqueeze(1).contiguous()  # (N, 1, T, H, W)
            elif layout == "tchw":
                a = a.permute(0, 3, 4, 1, 2).contiguous()  # (N, T, C, H, W)
                u = u.permute(0, 3, 1, 2).unsqueeze(2).contiguous()  # (N, T, 1, H, W)
            else:
                raise ValueError(f"Unknown layout: {layout!r}")

        if normalize:
            self.a_mean = a.mean(dim=tuple(d for d in range(a.dim()) if d != 1), keepdim=True)
            self.a_std = a.std(dim=tuple(d for d in range(a.dim()) if d != 1), keepdim=True) + 1e-8
            self.u_mean = u.mean(dim=tuple(d for d in range(u.dim()) if d != 1), keepdim=True)
            self.u_std = u.std(dim=tuple(d for d in range(u.dim()) if d != 1), keepdim=True) + 1e-8
            a = (a - self.a_mean) / self.a_std
            u = (u - self.u_mean) / self.u_std

        self.x = a
        self.y = u

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_fno4co2_unlabeled(
    root: str,
    variant: str = "dP",
    split: str = "test",
    n_samples: Optional[int] = None,
    t_index: Optional[int] = None,
    normalize: bool = True,
    batch_size: int = 4,
    shuffle: bool = True,
    resize_to: Optional[Tuple[int, int]] = None,
    layout: str = "ctxy",
    keep_channels: Optional[Tuple[int, ...]] = None,
) -> DataLoader:
    """Convenience factory.

    Args:
        root: e.g. "data/fno4co2/dataset"
        variant: "dP" (pressure) or "sg" (saturation)
        split: "test", "train", "val"
    """
    a_path = os.path.join(root, f"{variant}_{split}_a.pt")
    ds = FNO4CO2UnlabeledDataset(
        a_path=a_path,
        n_samples=n_samples,
        t_index=t_index,
        normalize=normalize,
        resize_to=resize_to,
        layout=layout,
        keep_channels=keep_channels,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def load_fno4co2_labeled(
    root: str,
    variant: str = "dP",
    split: str = "test",
    n_samples: Optional[int] = None,
    t_index: Optional[int] = None,
    normalize: bool = True,
    batch_size: int = 4,
    shuffle: bool = True,
    layout: str = "ctxy",
    keep_channels: Optional[Tuple[int, ...]] = None,
) -> DataLoader:
    a_path = os.path.join(root, f"{variant}_{split}_a.pt")
    u_path = os.path.join(root, f"{variant}_{split}_u.pt")
    ds = FNO4CO2LabeledDataset(
        a_path=a_path,
        u_path=u_path,
        n_samples=n_samples,
        t_index=t_index,
        normalize=normalize,
        layout=layout,
        keep_channels=keep_channels,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
