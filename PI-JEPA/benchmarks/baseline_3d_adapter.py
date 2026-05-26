"""Uniform 3D baseline adapter.

Wraps any 3D nn.Module (FNO3D, PINO3D, U-FNO3D, PI-DeepONet3D, ...) into
a `*Wrapper`-style object with the same `train_model(loader, epochs, lr)`
and `predict(x)` interface that Brandon's 2D wrappers expose. This lets
`train_eval_baseline` dispatch 2D and 3D baselines without separate code
paths.

If the wrapped model has a `physics_loss(x, pred)` method (PINO3D,
PI-DeepONet3D), the adapter automatically adds a physics-residual term
to the training loss with the given weight.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


class BaselineAdapter3D:
    """Uniform wrapper around any 3D baseline `nn.Module`.

    Implements `train_model(loader, epochs, lr)` and `predict(x)` so it
    plugs into `scripts/run_full_benchmarks.py::train_eval_baseline`.

    If `model` defines a callable attribute `physics_loss(x, pred)`, the
    adapter mixes it into the supervised MSE with weight
    `physics_weight`. This matches PINO2D's pattern.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        in_channels: int = 1,
        out_channels: int = 1,
        physics_weight: float = 0.0,
        lr: float = 1e-3,
    ):
        self.device = device
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.model = model.to(device)
        self.physics_weight = float(physics_weight)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self._has_physics = (
            hasattr(self.model, "physics_loss")
            and callable(getattr(self.model, "physics_loss"))
        )

    @staticmethod
    def _to_5d(x: torch.Tensor) -> torch.Tensor:
        """Promote (B,D,H,W) → (B,1,D,H,W); pass (B,C,D,H,W) through."""
        if x.dim() == 4:
            return x.unsqueeze(1)
        if x.dim() == 5:
            return x
        raise ValueError(
            f"BaselineAdapter3D expects 4D or 5D input; got {tuple(x.shape)}"
        )

    def train_model(self, loader: Any, epochs: int, lr: float) -> None:
        # Reset optimizer LR in case the caller wants a different rate
        # than what was set at construction time.
        for g in self.optimizer.param_groups:
            g["lr"] = float(lr)

        self.model.train()
        for epoch in range(epochs):
            total = 0.0
            n_batches = 0
            for batch in loader:
                x = self._to_5d(batch["x"].to(self.device).float())
                y = self._to_5d(batch["y"].to(self.device).float())

                # Channel slicing to match declared in/out_channels.
                x_in = x[:, : self.in_channels]
                y_tgt = y[:, : self.out_channels]

                pred = self.model(x_in)
                loss = self.loss_fn(pred, y_tgt)

                if self._has_physics and self.physics_weight > 0.0:
                    phys = self.model.physics_loss(x_in, pred)
                    loss = loss + self.physics_weight * phys

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

                total += float(loss.item())
                n_batches += 1
            avg = total / max(1, n_batches)
            tag = type(self.model).__name__
            print(f"[{tag}] epoch {epoch+1}/{epochs} loss={avg:.6f}")

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = self._to_5d(x.to(self.device).float())
        return self.model(x[:, : self.in_channels])
