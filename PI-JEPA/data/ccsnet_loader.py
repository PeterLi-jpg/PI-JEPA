"""
CCSNet data loader.

CCSNet (Wen, Hay, Benson 2021) ships HDF5 files with shape
    (N, H=96, W=200, T=24, C)
covering 2D radial-grid CO2 injection simulations. The input file
test_x.hdf5 has C=1 (permeability-like conditioning channel) and various
output files (dP, SG, BXMF, BYMF, BDENW, BDENG) hold one solver field each.

This module exposes two utilities for PI-JEPA:

1. `load_ccsnet_unlabeled(...)` -> torch DataLoader yielding {"x": tensor}
   suitable for self-supervised pretraining on the input (permeability) field
   only. By default we collapse the time axis by taking timestep `t_index`
   (default 0, the initial condition) so the resulting tensor matches the
   existing 2D PI-JEPA encoder (B, C, H, W).

2. `load_ccsnet_labeled(...)` -> torch DataLoader yielding (x, y) tuples,
   x = input field at chosen timestep, y = output field at chosen timestep
   (default: final timestep, T=-1). Use this for supervised fine-tuning.

These loaders are deliberately small and dependency-free. They are NOT a
full space-time loader — that requires the (B, C, T, H, W) variant of the
encoder which is on the roadmap but not yet implemented.

Tensor layout convention here: PyTorch wants channel-first
    (B, C, H, W)
while CCSNet's HDF5 layout is (N, H, W, T, C). We permute on read.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# Map of CCSNet output variable name -> filename pattern within the
# CCSNet_v1.0 release directory. Pattern: test_y_<NAME>.hdf5 / train_y_<NAME>.hdf5
CCSNET_OUTPUT_VARS = ("dP", "SG", "BXMF", "BYMF", "BDENW", "BDENG")


def _read_ccsnet_array(path: str, key: Optional[str] = None) -> np.ndarray:
    """Open an HDF5 file and return the single top-level dataset as a numpy array.

    CCSNet files typically have a single dataset named the same as the file's
    purpose ("test_x", "test_y", etc.). We auto-detect when key is None.
    """
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        if key is None:
            if len(keys) != 1:
                raise ValueError(
                    f"{path}: expected exactly one top-level dataset, got {keys}"
                )
            key = keys[0]
        return f[key][...]


class CCSNetUnlabeledDataset(Dataset):
    """Coefficient field only, with time axis collapsed to a single snapshot.

    Yields {"x": tensor of shape (C, H, W)} so that a standard 2D PI-JEPA
    pretrainer (which expects (B, C, H, W) batches) can consume it directly.
    """

    def __init__(
        self,
        x_path: str,
        n_samples: Optional[int] = None,
        t_index: Optional[int] = None,
        normalize: bool = True,
        cache_in_memory: bool = True,
        resize_to: Optional[Tuple[int, int]] = None,
        layout: str = "ctxy",
    ):
        """
        Args:
            t_index: if given (int), take only that single timestep and emit
                4D samples (C, H, W). If None (default), keep the full time
                axis and emit 5D samples per the `layout` argument.
            resize_to: optional spatial resize (H_target, W_target). Applied
                to the (H, W) axes only; the time axis is preserved.
            layout: when t_index is None, controls the channel/time/space
                ordering of returned samples. Supported:
                - "ctxy" (default): (C, T, H, W) — matches U-FNO / FNO4CO2
                  3D-Conv convention with time as the "depth" axis. Use this
                  with the FourierJEPAEncoder3D path.
                - "tchw": (T, C, H, W) — interpret each timestep as a separate
                  channel-first 2D sample (rarely useful).
        """
        super().__init__()
        if not Path(x_path).exists():
            raise FileNotFoundError(f"CCSNet x file not found: {x_path}")

        if cache_in_memory:
            arr = _read_ccsnet_array(x_path)
            # CCSNet native: (N, H, W, T, C).
            if arr.ndim != 5:
                raise ValueError(
                    f"{x_path}: expected 5D array (N,H,W,T,C); got shape {arr.shape}"
                )

            if t_index is not None:
                # Collapse time: (N, H, W, T, C) -> (N, H, W, C) -> (N, C, H, W)
                arr = arr[:, :, :, t_index, :]
                arr = np.transpose(arr, (0, 3, 1, 2))
                if n_samples is not None:
                    arr = arr[:n_samples]
                t = torch.from_numpy(arr).float()
                if resize_to is not None:
                    t = torch.nn.functional.interpolate(
                        t, size=resize_to, mode="bilinear", align_corners=False
                    )
                if normalize:
                    mean = t.mean(dim=(0, 2, 3), keepdim=True)
                    std = t.std(dim=(0, 2, 3), keepdim=True) + 1e-8
                    t = (t - mean) / std
            else:
                # Keep time. Permute to layout.
                if layout == "ctxy":
                    # (N, H, W, T, C) -> (N, C, T, H, W)
                    arr = np.transpose(arr, (0, 4, 3, 1, 2))
                elif layout == "tchw":
                    # (N, H, W, T, C) -> (N, T, C, H, W)
                    arr = np.transpose(arr, (0, 3, 4, 1, 2))
                else:
                    raise ValueError(f"Unknown layout: {layout!r}")
                if n_samples is not None:
                    arr = arr[:n_samples]
                t = torch.from_numpy(arr).float()
                if resize_to is not None:
                    # Resize the LAST two axes (H, W). Use trilinear if 5D.
                    if t.dim() == 5:
                        # (N, C, T, H, W) — interpolate spatial only by
                        # reshaping (N, C*T, H, W), resize, reshape back.
                        N, C, T, H, W = t.shape
                        t_flat = t.reshape(N, C * T, H, W)
                        t_flat = torch.nn.functional.interpolate(
                            t_flat, size=resize_to, mode="bilinear", align_corners=False
                        )
                        t = t_flat.reshape(N, C, T, *resize_to)
                if normalize:
                    # Normalize over batch + spatial + temporal axes, leave channels.
                    if t.dim() == 5:
                        mean = t.mean(dim=(0, 2, 3, 4), keepdim=True)
                        std = t.std(dim=(0, 2, 3, 4), keepdim=True) + 1e-8
                    else:
                        mean = t.mean(dim=(0, 2, 3), keepdim=True)
                        std = t.std(dim=(0, 2, 3), keepdim=True) + 1e-8
                    t = (t - mean) / std

            self.x = t
            self._lazy = False
            self._x_path = x_path
            self._t_index = t_index
        else:
            # Lazy mode: open per __getitem__. Slower but doesn't blow up RAM
            # for very large CCSNet variants.
            self._lazy = True
            self._x_path = x_path
            self._t_index = t_index
            with h5py.File(x_path, "r") as f:
                key = list(f.keys())[0]
                self._n = f[key].shape[0] if n_samples is None else min(n_samples, f[key].shape[0])
                self._dataset_key = key
            self._normalize = normalize
            # We don't compute per-channel stats in lazy mode; user should
            # pre-normalize the file or accept raw values.
            self._lazy_mean = None
            self._lazy_std = None

    def __len__(self):
        return self.x.shape[0] if not self._lazy else self._n

    def __getitem__(self, i):
        if not self._lazy:
            return {"x": self.x[i]}
        # Lazy read
        with h5py.File(self._x_path, "r") as f:
            arr = f[self._dataset_key][i, :, :, self._t_index, :]  # (H, W, C)
        arr = np.transpose(arr, (2, 0, 1))  # (C, H, W)
        return {"x": torch.from_numpy(arr).float()}


class CCSNetLabeledDataset(Dataset):
    """Yields (input_field, output_field) pairs from a chosen variable.

    output_var picks which CCSNet target to load (dP, SG, ...). The dataset
    pairs the input at `t_in_index` with the output at `t_out_index`.
    """

    def __init__(
        self,
        x_path: str,
        y_path: str,
        n_samples: Optional[int] = None,
        t_in_index: int = 0,
        t_out_index: int = -1,
        normalize: bool = True,
    ):
        super().__init__()
        if not Path(x_path).exists():
            raise FileNotFoundError(f"CCSNet x file not found: {x_path}")
        if not Path(y_path).exists():
            raise FileNotFoundError(f"CCSNet y file not found: {y_path}")

        x = _read_ccsnet_array(x_path)
        y = _read_ccsnet_array(y_path)
        if x.ndim != 5 or y.ndim != 5:
            raise ValueError(
                f"Expected 5D arrays; got x={x.shape}, y={y.shape}"
            )

        x = x[:, :, :, t_in_index, :]      # (N, H, W, C_x)
        y = y[:, :, :, t_out_index, :]     # (N, H, W, C_y)
        x = np.transpose(x, (0, 3, 1, 2))   # (N, C_x, H, W)
        y = np.transpose(y, (0, 3, 1, 2))   # (N, C_y, H, W)

        if n_samples is not None:
            x = x[:n_samples]
            y = y[:n_samples]

        x_t = torch.from_numpy(x).float()
        y_t = torch.from_numpy(y).float()
        if normalize:
            self.x_mean = x_t.mean(dim=(0, 2, 3), keepdim=True)
            self.x_std = x_t.std(dim=(0, 2, 3), keepdim=True) + 1e-8
            self.y_mean = y_t.mean(dim=(0, 2, 3), keepdim=True)
            self.y_std = y_t.std(dim=(0, 2, 3), keepdim=True) + 1e-8
            x_t = (x_t - self.x_mean) / self.x_std
            y_t = (y_t - self.y_mean) / self.y_std
        self.x = x_t
        self.y = y_t

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def load_ccsnet_unlabeled(
    root: str,
    split: str = "test",
    n_samples: Optional[int] = None,
    t_index: Optional[int] = None,
    normalize: bool = True,
    batch_size: int = 8,
    shuffle: bool = True,
    resize_to: Optional[Tuple[int, int]] = None,
    layout: str = "ctxy",
) -> DataLoader:
    """Convenience factory for the unlabeled CCSNet loader.

    Args:
        root: e.g. "data/ccsnet/CCSNet_v1.0"
        split: "test" or "train" (file pattern is f"{split}_x.hdf5")
        n_samples: optional cap on the number of samples
        t_index: which timestep to take. If None (default), keep the full
            time axis (24 steps) and emit 5D (C, T, H, W) samples — this is
            the U-FNO / FourierJEPAEncoder3D convention. If an int, collapse
            to a single 2D snapshot.
        normalize: zero-mean, unit-std per channel
        batch_size: DataLoader batch size
        shuffle: shuffle indices
        resize_to: optional (H, W) to resize the spatial axes. The time axis
            is preserved.
        layout: when t_index is None, "ctxy" -> (C, T, H, W), "tchw" -> (T, C, H, W).
    """
    x_path = os.path.join(root, f"{split}_x.hdf5")
    ds = CCSNetUnlabeledDataset(
        x_path=x_path,
        n_samples=n_samples,
        t_index=t_index,
        normalize=normalize,
        resize_to=resize_to,
        layout=layout,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )


def load_ccsnet_labeled(
    root: str,
    output_var: str = "dP",
    split: str = "test",
    n_samples: Optional[int] = None,
    t_in_index: int = 0,
    t_out_index: int = -1,
    normalize: bool = True,
    batch_size: int = 8,
    shuffle: bool = True,
) -> DataLoader:
    """Convenience factory for the labeled CCSNet loader.

    Args:
        root: e.g. "data/ccsnet/CCSNet_v1.0"
        output_var: one of CCSNET_OUTPUT_VARS (dP, SG, BXMF, BYMF, BDENW, BDENG)
        split: "test" or "train"
    """
    if output_var not in CCSNET_OUTPUT_VARS:
        raise ValueError(
            f"output_var must be one of {CCSNET_OUTPUT_VARS}; got {output_var}"
        )
    x_path = os.path.join(root, f"{split}_x.hdf5")
    y_path = os.path.join(root, f"{split}_y_{output_var}.hdf5")
    ds = CCSNetLabeledDataset(
        x_path=x_path,
        y_path=y_path,
        n_samples=n_samples,
        t_in_index=t_in_index,
        t_out_index=t_out_index,
        normalize=normalize,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=False,
    )
