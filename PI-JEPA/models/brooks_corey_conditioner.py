"""
Brooks-Corey Conditioner: Conditioning module for variable Brooks-Corey parameters.

Supports both scalar (per-sample) and spatially varying (2D field)
Brooks-Corey parameters (λ, S_wr, S_nr) as conditioning inputs for the encoder.

For scalar parameters:
    An MLP maps the parameter vector to conditioning features that are
    concatenated with the encoder input channels.

For spatial parameters:
    A small CNN processes the 2D parameter fields into conditioning feature
    maps that are concatenated with the encoder input channels.

This enables the encoder to generalize across different rock-fluid
interaction models without retraining.
"""

import torch
import torch.nn as nn
from torch import Tensor


class BrooksCoreyConditioner(nn.Module):
    """Conditioning module for variable Brooks-Corey parameters.

    Supports both scalar (per-sample) and spatially varying (2D field)
    Brooks-Corey parameters (λ, S_wr, S_nr).

    For scalar mode:
        Input (B, n_params) → MLP → (B, hidden_channels, H, W) conditioning features
        The MLP output is spatially broadcast to match the encoder input resolution.

    For spatial mode:
        Input (B, n_params, H, W) → CNN → (B, hidden_channels, H, W) conditioning features
        A small CNN processes the spatially varying parameter fields.

    The output conditioning features are intended to be concatenated with the
    encoder's input channels, increasing the effective in_channels by hidden_channels.

    Args:
        n_params: Number of Brooks-Corey parameters (default 3 for λ, S_wr, S_nr).
        hidden_channels: Number of output conditioning channels.
        spatial: If True, expect spatially varying 2D field inputs (B, n_params, H, W).
                 If False, expect scalar per-sample inputs (B, n_params).
        image_size: Spatial resolution for broadcasting scalar params (default 64).
    """

    def __init__(
        self,
        n_params: int = 3,
        hidden_channels: int = 64,
        spatial: bool = False,
        image_size: int = 64,
    ):
        super().__init__()
        self.n_params = n_params
        self.hidden_channels = hidden_channels
        self.spatial = spatial
        self.image_size = image_size

        if spatial:
            # Small CNN for spatially varying parameters
            self.net = nn.Sequential(
                nn.Conv2d(n_params, hidden_channels // 2, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(min(8, hidden_channels // 2), hidden_channels // 2),
                nn.Conv2d(hidden_channels // 2, hidden_channels, kernel_size=3, padding=1),
                nn.GELU(),
                nn.GroupNorm(min(8, hidden_channels), hidden_channels),
            )
        else:
            # MLP for scalar per-sample parameters
            mlp_hidden = hidden_channels * 2
            self.net = nn.Sequential(
                nn.Linear(n_params, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, hidden_channels),
                nn.GELU(),
            )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with small values so conditioning starts near zero."""
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                # Scale down initial weights so conditioning starts small
                with torch.no_grad():
                    m.weight.mul_(0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, bc_params: Tensor) -> Tensor:
        """Process Brooks-Corey parameters into conditioning features.

        Args:
            bc_params: Either (B, n_params) for scalar or (B, n_params, H, W) for spatial.

        Returns:
            Conditioning features of shape (B, hidden_channels, H, W) to concatenate
            with encoder input.
        """
        if self.spatial:
            # Spatial mode: bc_params is (B, n_params, H, W)
            assert bc_params.dim() == 4, (
                f"Spatial mode expects 4D input (B, n_params, H, W), got {bc_params.dim()}D"
            )
            features = self.net(bc_params)  # (B, hidden_channels, H, W)
        else:
            # Scalar mode: bc_params is (B, n_params)
            assert bc_params.dim() == 2, (
                f"Scalar mode expects 2D input (B, n_params), got {bc_params.dim()}D"
            )
            B = bc_params.shape[0]
            h = self.net(bc_params)  # (B, hidden_channels)
            # Broadcast spatially to (B, hidden_channels, H, W)
            features = h.unsqueeze(-1).unsqueeze(-1).expand(
                B, self.hidden_channels, self.image_size, self.image_size
            )

        return features
