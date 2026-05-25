"""
PVT Conditioner: FiLM-based conditioning on fluid PVT properties.

Applies Feature-wise Linear Modulation (FiLM) to encoder hidden features
at each Fourier block:

    h_out = γ(pvt) * h + β(pvt)

This enables the encoder to generalize across different fluid systems
(varying viscosity ratio, compressibility, gas-oil ratio) without
retraining.
"""

import torch
import torch.nn as nn
from typing import List, Tuple

from torch import Tensor


class PVTConditioner(nn.Module):
    """FiLM-based conditioning on fluid PVT properties.

    Applies Feature-wise Linear Modulation: γ(pvt) * h + β(pvt)
    to encoder hidden features at each Fourier block.

    Architecture:
        1. Shared MLP processes the PVT vector into a latent representation
        2. Per-layer linear heads produce (gamma, beta) pairs for each Fourier block

    Initialization:
        gamma is initialized to 1 and beta to 0 (identity initialization),
        so the conditioner starts as a no-op and gradually learns modulation.

    Args:
        pvt_dim: Dimension of the PVT input vector (default 3 for
                 [viscosity_ratio, compressibility, GOR]).
        hidden_channels: Number of channels in the encoder's Fourier blocks.
                         Each gamma/beta will have this many elements.
        n_layers: Number of Fourier blocks in the encoder. One (gamma, beta)
                  pair is produced per layer.
    """

    def __init__(
        self,
        pvt_dim: int = 3,
        hidden_channels: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.pvt_dim = pvt_dim
        self.hidden_channels = hidden_channels
        self.n_layers = n_layers

        # Shared MLP to process PVT vector into a latent representation
        mlp_hidden = hidden_channels * 2
        self.shared_mlp = nn.Sequential(
            nn.Linear(pvt_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
        )

        # Per-layer linear heads producing (gamma, beta) pairs
        # Each head outputs 2 * hidden_channels: first half is gamma, second is beta
        self.layer_heads = nn.ModuleList([
            nn.Linear(mlp_hidden, 2 * hidden_channels)
            for _ in range(n_layers)
        ])

        # Identity initialization: gamma=1, beta=0
        self._init_identity()

    def _init_identity(self) -> None:
        """Initialize so that gamma=1 and beta=0 (no-op at start)."""
        for head in self.layer_heads:
            # Zero out the weight so initial output is just the bias
            nn.init.zeros_(head.weight)
            # Bias: first half (gamma) = 1, second half (beta) = 0
            with torch.no_grad():
                head.bias[:self.hidden_channels].fill_(1.0)
                head.bias[self.hidden_channels:].fill_(0.0)

    def forward(self, pvt_params: Tensor) -> List[Tuple[Tensor, Tensor]]:
        """Produce (gamma, beta) pairs for each encoder layer.

        Args:
            pvt_params: PVT parameter vector of shape (B, pvt_dim).

        Returns:
            List of (gamma, beta) tuples, one per encoder layer.
            Each gamma and beta has shape (B, hidden_channels).
        """
        # Shared representation
        h = self.shared_mlp(pvt_params)  # (B, mlp_hidden)

        # Per-layer (gamma, beta) pairs
        film_params: List[Tuple[Tensor, Tensor]] = []
        for head in self.layer_heads:
            out = head(h)  # (B, 2 * hidden_channels)
            gamma = out[:, :self.hidden_channels]  # (B, hidden_channels)
            beta = out[:, self.hidden_channels:]   # (B, hidden_channels)
            film_params.append((gamma, beta))

        return film_params
