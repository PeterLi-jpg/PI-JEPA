"""Combined-pool unlabeled pretrain loader.

Wires Brandon's `IrregularGridProcessor` (shape harmonization) and
`MultiFidelityPretrainer` (tier-weighted sampling) into a single torch
Dataset that yields `{"x": tensor_at_common_shape}` dicts for
cross-dataset self-supervised pretraining.

Goal: instead of pretraining one encoder per dataset, pretrain a single
encoder on a TIER-WEIGHTED POOL of unlabeled parameter fields drawn
from multiple datasets at different native shapes.

How it composes Brandon's tools:
- `IrregularGridProcessor.handle_nans` is used to sanitize NaN holes
  per-tier sample (his class is 2D-only, so we call it on each 2D
  slice for 3D inputs). We add a small `_resize_3d_trilinear` helper
  for 3D shape harmonization since his `process()` asserts 2D output.
- `MultiFidelityPretrainer` is used to schedule which tier each sample
  is drawn from at each epoch. By default we run it in non-progressive
  mode (fixed tier weights) so we don't need to patch the existing
  pretrainer's epoch loop. To use the progressive intro schedule,
  callers must call `set_current_epoch(epoch)` on the returned dataset
  at the top of each epoch.

Returned DataLoader yields `{"x": (B, C, D, H, W) tensor}` so it is a
drop-in replacement for the per-domain unlabeled loaders called by
`scripts/run_full_benchmarks.py::pretrain_on_domain`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from ..training.multi_fidelity import MultiFidelityPretrainer
    from .irregular_grid import IrregularGridProcessor
except (ImportError, ValueError):
    from training.multi_fidelity import MultiFidelityPretrainer
    from data.irregular_grid import IrregularGridProcessor


# --------------------------------------------------------------------------
# 3D shape harmonization (Brandon's IrregularGridProcessor is 2D-only;
# we add a 3D path here using trilinear interpolation).
# --------------------------------------------------------------------------

def _resize_3d_trilinear(
    x: torch.Tensor, target_shape: Tuple[int, int, int]
) -> torch.Tensor:
    """Trilinear-resize a 3D field to a common (D, H, W).

    Input can be (D, H, W), (C, D, H, W), or (B, C, D, H, W). Output keeps
    the same channel/batch structure, with spatial dims resized.

    NaN/Inf are sanitized to 0 before interpolation.
    """
    x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    if x.dim() == 3:
        x = x.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        squeeze_axes: Tuple[int, ...] = (0, 0)
    elif x.dim() == 4:
        x = x.unsqueeze(0)  # (1, C, D, H, W)
        squeeze_axes = (0,)
    elif x.dim() == 5:
        squeeze_axes = ()
    else:
        raise ValueError(
            f"_resize_3d_trilinear expects 3D/4D/5D input; got {tuple(x.shape)}"
        )
    if x.shape[-3:] != target_shape:
        x = F.interpolate(x, size=target_shape, mode="trilinear", align_corners=False)
    for _ in squeeze_axes:
        x = x.squeeze(0)
    return x


def _sanitize_3d_with_brandon_2d_processor(
    x: torch.Tensor, processor: IrregularGridProcessor
) -> torch.Tensor:
    """Use Brandon's 2D `handle_nans` on each 2D slice of a 3D volume.

    Lets us actually USE his code on 3D data even though it was written
    for 2D. We slice along D, run his NaN-fill on each (H, W) slice,
    then stack back to 3D.
    """
    if x.dim() == 3:
        # (D, H, W) — process each (H, W) slice
        return torch.stack(
            [processor.handle_nans(slice_2d, method="interpolate") for slice_2d in x],
            dim=0,
        )
    if x.dim() == 4:
        # (C, D, H, W)
        out = torch.empty_like(x)
        for c in range(x.shape[0]):
            for d in range(x.shape[1]):
                out[c, d] = processor.handle_nans(x[c, d], method="interpolate")
        return out
    return x


# --------------------------------------------------------------------------
# Dataset that draws from tier datasets at sampler-determined weights
# --------------------------------------------------------------------------

class CombinedPoolDataset(Dataset):
    """Resampling-on-the-fly Dataset over tier datasets.

    Each `__getitem__`:
      1. Asks `MultiFidelityPretrainer` which tier to sample from at the
         current epoch.
      2. Draws a random sample from that tier's underlying Dataset.
      3. Sanitizes NaNs via Brandon's `IrregularGridProcessor` (2D path,
         applied slice-wise on 3D volumes).
      4. Trilinear-resizes to the common `target_shape`.
      5. Returns `{"x": (C, D, H, W) tensor}`.

    Set `dataset.current_epoch` before each epoch (only needed when
    `progressive=True` on the MF pretrainer).
    """

    def __init__(
        self,
        tier_datasets: List[Dataset],
        mf_pretrainer: MultiFidelityPretrainer,
        target_shape: Tuple[int, int, int],
        samples_per_epoch: int,
        irregular_processor: Optional[IrregularGridProcessor] = None,
    ):
        if len(tier_datasets) != mf_pretrainer.n_tiers:
            raise ValueError(
                f"tier_datasets ({len(tier_datasets)}) must match "
                f"mf_pretrainer.n_tiers ({mf_pretrainer.n_tiers})"
            )
        non_empty = [i for i, d in enumerate(tier_datasets) if len(d) > 0]
        if not non_empty:
            raise ValueError("All tier_datasets are empty — nothing to sample from.")
        self.tier_datasets = tier_datasets
        self.mf = mf_pretrainer
        self.target_shape = tuple(target_shape)
        self.samples_per_epoch = int(samples_per_epoch)
        self.current_epoch = 0
        self.irregular_processor = irregular_processor or IrregularGridProcessor(
            target_resolution=max(target_shape[-2:]), interpolation_method="bilinear"
        )
        self._non_empty_tiers = non_empty

    def __len__(self) -> int:
        return self.samples_per_epoch

    def set_current_epoch(self, epoch: int) -> None:
        """Update the epoch counter so progressive-mode sampling sees it."""
        self.current_epoch = int(epoch)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Pick a tier per the MF schedule. If the chosen tier happens to be
        # empty, fall back to a non-empty one (defensive — __init__ already
        # rejected the all-empty case).
        tier = self.mf.sample_tier(self.current_epoch)
        if tier >= len(self.tier_datasets) or len(self.tier_datasets[tier]) == 0:
            tier = self._non_empty_tiers[idx % len(self._non_empty_tiers)]
        ds = self.tier_datasets[tier]
        sample_idx = int(torch.randint(0, len(ds), (1,)).item())
        sample = ds[sample_idx]

        # Underlying dataset items may be tensors, tuples, or dicts.
        if isinstance(sample, dict) and "x" in sample:
            x = sample["x"]
        elif isinstance(sample, (tuple, list)):
            x = sample[0]
        else:
            x = sample
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)

        # Sanitize NaNs using Brandon's 2D nan-fill applied slice-wise
        x = _sanitize_3d_with_brandon_2d_processor(x, self.irregular_processor)

        # Harmonize 3D shape via trilinear resize
        x = _resize_3d_trilinear(x, self.target_shape)

        # Ensure leading channel dim (C, D, H, W) for stacking downstream
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return {"x": x.float()}


# --------------------------------------------------------------------------
# Public builder
# --------------------------------------------------------------------------

def build_combined_pool_loader(
    tier_specs: List[Dict[str, Any]],
    target_shape: Tuple[int, int, int] = (64, 64, 64),
    batch_size: int = 8,
    samples_per_epoch: int = 1024,
    progressive: bool = False,
    tier_introduction_epochs: Optional[List[int]] = None,
    num_workers: int = 0,
) -> DataLoader:
    """Construct a DataLoader over a tier-weighted combined unlabeled pool.

    Args:
        tier_specs: list of dicts, one per tier, ordered from
            lowest-fidelity to highest-fidelity. Each dict needs:
              - "build_fn":    zero-arg callable returning a torch Dataset
                               of parameter-field tensors at the dataset's
                               NATIVE shape (3D: (C,D,H,W) or (D,H,W)).
              - "weight":      relative sampling weight (>= 0).
              - "name" (optional): label for logging.
        target_shape: common (D, H, W) every sample is trilinear-resized to.
            Default (64,64,64) is a CUBE because Brandon's fourier_encoder_3d
            silently squashes rectangular inputs via adaptive_avg_pool3d
            (see audit). If you change this to a non-cubic shape, also
            extend `models/fourier_encoder_3d.py` to accept rectangular
            volume_size, or the encoder will produce wrong embeddings
            without raising.
        batch_size: DataLoader batch size.
        samples_per_epoch: how many samples one "epoch" through the combined
            pool draws. Pick something comparable to a single-dataset epoch.
        progressive: pass-through to `MultiFidelityPretrainer.progressive`.
            If True, callers MUST call `loader.dataset.set_current_epoch(e)`
            at the top of each epoch.
        tier_introduction_epochs: pass-through; epoch at which each tier
            becomes active under `progressive=True`. Default `[0]*n_tiers`.
        num_workers: DataLoader workers. Default 0 for safety with the
            tensor-based tier datasets.

    Returns:
        A torch DataLoader yielding `{"x": (B, C, D, H, W)}` dicts.
    """
    if not tier_specs:
        raise ValueError("tier_specs must be non-empty.")
    weights = [float(t["weight"]) for t in tier_specs]
    if any(w < 0 for w in weights):
        raise ValueError("All tier weights must be >= 0.")
    if sum(weights) == 0:
        raise ValueError("At least one tier must have a positive weight.")

    n_tiers = len(tier_specs)
    if tier_introduction_epochs is None:
        tier_introduction_epochs = [0] * n_tiers
    if len(tier_introduction_epochs) != n_tiers:
        raise ValueError(
            f"tier_introduction_epochs ({len(tier_introduction_epochs)}) "
            f"must match number of tiers ({n_tiers})."
        )

    mf = MultiFidelityPretrainer(
        tier_weights=weights,
        progressive=progressive,
        tier_introduction_epochs=tier_introduction_epochs,
        n_tiers=n_tiers,
    )

    tier_datasets: List[Dataset] = []
    for spec in tier_specs:
        if "build_fn" not in spec:
            raise KeyError("Each tier_spec needs a 'build_fn' callable.")
        ds = spec["build_fn"]()
        if not hasattr(ds, "__len__") or not hasattr(ds, "__getitem__"):
            raise TypeError(
                f"build_fn for tier '{spec.get('name', '?')}' must return a "
                f"torch Dataset; got {type(ds).__name__}."
            )
        tier_datasets.append(ds)

    pool_ds = CombinedPoolDataset(
        tier_datasets=tier_datasets,
        mf_pretrainer=mf,
        target_shape=target_shape,
        samples_per_epoch=samples_per_epoch,
    )

    return DataLoader(
        pool_ds,
        batch_size=batch_size,
        shuffle=False,  # sampling is internal to CombinedPoolDataset
        num_workers=num_workers,
        drop_last=False,
    )
