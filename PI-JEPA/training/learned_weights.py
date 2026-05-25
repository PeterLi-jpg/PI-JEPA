"""Learned per-sub-operator loss weights on log scale.

Each weight λ_k = exp(log_λ_k) ensures positivity.
Optimized in a separate AdamW parameter group.
"""

import torch
import torch.nn as nn
from torch import Tensor


class LearnedLossWeights(nn.Module):
    """Learned per-sub-operator loss weights on log scale.

    Each weight λ_k = exp(log_λ_k) ensures positivity.
    Optimized in a separate AdamW parameter group with its own learning rate.

    Parameters
    ----------
    num_operators : int
        Number of sub-operators (loss terms) to weight. Default is 2
        (pressure, saturation).
    """

    def __init__(self, num_operators: int = 2):
        super().__init__()
        # Initialize log_weights to zeros so all weights start at exp(0) = 1.0
        self.log_weights = nn.Parameter(torch.zeros(num_operators))

    @property
    def weights(self) -> Tensor:
        """Return positive weights λ_k = exp(clamped log_λ_k).

        Clamps log_weights to [-10, 10] before exponentiation to prevent
        numerical overflow (exp(10) ≈ 22026) or underflow (exp(-10) ≈ 4.5e-5).
        """
        clamped = self.log_weights.clamp(-10.0, 10.0)
        return clamped.exp()

    def get_parameter_group(self, lr: float = 1e-3) -> dict:
        """Return optimizer parameter group for these weights.

        Uses AdamW with no weight decay so the log-scale weights are
        free to move without regularization bias.

        Parameters
        ----------
        lr : float
            Learning rate for the weight parameters. Default 1e-3.

        Returns
        -------
        dict
            Parameter group dict suitable for passing to an optimizer.
        """
        return {"params": [self.log_weights], "lr": lr, "weight_decay": 0.0}
