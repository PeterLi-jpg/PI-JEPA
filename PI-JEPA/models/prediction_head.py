"""Prediction head for mapping encoder embeddings to solution fields during finetuning."""

import math
import torch
import torch.nn as nn
from torch import Tensor


class PredictionHead(nn.Module):
    """Prediction head that unpatchifies encoder embeddings then refines with a CNN.

    Architecture:
    1. Linear projection: (B, N, D) -> (B, N, C*P*P)  per-patch pixel prediction
    2. Unpatchify: fold patches back to (B, C_mid, H, W)
    3. Lightweight refinement CNN at full resolution
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 512,
        output_channels: int = 1,
        image_size: int = 64,
        patch_size: int = 8,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.patch_size = patch_size

        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size ** 2

        # Number of intermediate channels produced per patch
        mid_channels = 16

        # Step 1: project each patch embedding to a patch of pixels
        self.patch_proj = nn.Linear(embed_dim, mid_channels * patch_size * patch_size)

        # Step 2: lightweight refinement at full resolution
        self.refine = nn.Sequential(
            nn.Conv2d(mid_channels, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, output_channels, 1),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        for m in self.refine.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: Tensor) -> Tensor:
        """Map encoder patch embeddings to a full-resolution solution field.

        Args:
            z: (B, N, D) encoder output

        Returns:
            (B, output_channels, H, W) predicted solution field
        """
        B, N, D = z.shape
        G = self.grid_size
        P = self.patch_size
        C = self.refine[-1].out_channels  # output_channels
        mid = self.patch_proj.out_features // (P * P)

        # Per-patch pixel prediction
        x = self.patch_proj(z)                       # (B, N, mid*P*P)
        x = x.view(B, G, G, mid, P, P)              # (B, G, G, mid, P, P)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()  # (B, mid, G, P, G, P)
        x = x.view(B, mid, G * P, G * P)            # (B, mid, H, W)

        # Refine at full resolution
        y_pred = self.refine(x)                      # (B, C, H, W)
        return y_pred
