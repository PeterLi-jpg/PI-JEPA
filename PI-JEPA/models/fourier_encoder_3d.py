"""
Fourier-JEPA 3D Encoder: Volumetric Physics-Aware Encoder

This module extends the 2D Fourier-JEPA encoder to 3D volumetric data:
1. SpectralConv3d: 3D spectral convolution using rfftn/irfftn
2. FourierBlock3D: Combined spectral + local convolution block
3. FourierJEPAEncoder3D: Full 3D encoder with 8×8×8 patchification

Designed for 32×32×32 grids (e.g., SPE10 Tarbert formation layers),
producing 64 tokens (4×4×4 patches) compatible with the JEPA framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectralConv3d(nn.Module):
    """3D Spectral convolution layer operating in Fourier space.

    Uses torch.fft.rfftn / torch.fft.irfftn for efficient 3D spectral
    convolution with learnable complex weights in the frequency domain.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int,
        modes3: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3

        # Learnable Fourier weights for 4 quadrants of the 3D spectrum
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights3 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )
        self.weights4 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, modes3, dtype=torch.cfloat)
        )

    def compl_mul3d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """Complex multiplication in 3D Fourier space."""
        return torch.einsum("bixyz,ioxyz->boxyz", input, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, D, H, W) input volume
        Returns:
            (B, C_out, D, H, W) output volume
        """
        B, C, D, H, W = x.shape

        # Clamp modes to input resolution
        modes1 = min(self.modes1, D)
        modes2 = min(self.modes2, H)
        modes3 = min(self.modes3, W // 2 + 1)

        # 3D FFT (real-valued input → half-complex output along last dim)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Allocate output in frequency domain
        out_ft = torch.zeros(
            B, self.out_channels, D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        # Multiply relevant Fourier modes in 4 quadrants
        # Quadrant 1: [:modes1, :modes2, :modes3]
        out_ft[:, :, :modes1, :modes2, :modes3] = self.compl_mul3d(
            x_ft[:, :, :modes1, :modes2, :modes3],
            self.weights1[:, :, :modes1, :modes2, :modes3]
        )
        # Quadrant 2: [-modes1:, :modes2, :modes3]
        out_ft[:, :, -modes1:, :modes2, :modes3] = self.compl_mul3d(
            x_ft[:, :, -modes1:, :modes2, :modes3],
            self.weights2[:, :, :modes1, :modes2, :modes3]
        )
        # Quadrant 3: [:modes1, -modes2:, :modes3]
        out_ft[:, :, :modes1, -modes2:, :modes3] = self.compl_mul3d(
            x_ft[:, :, :modes1, -modes2:, :modes3],
            self.weights3[:, :, :modes1, :modes2, :modes3]
        )
        # Quadrant 4: [-modes1:, -modes2:, :modes3]
        out_ft[:, :, -modes1:, -modes2:, :modes3] = self.compl_mul3d(
            x_ft[:, :, -modes1:, -modes2:, :modes3],
            self.weights4[:, :, :modes1, :modes2, :modes3]
        )

        # Inverse 3D FFT
        return torch.fft.irfftn(out_ft, s=(D, H, W))


class FourierBlock3D(nn.Module):
    """Combined 3D Fourier + local convolution block with residual connection."""

    def __init__(
        self,
        channels: int,
        modes: Tuple[int, int, int] = (8, 8, 8),
        mlp_ratio: float = 2.0,
    ):
        super().__init__()

        # Spectral path
        self.spectral = SpectralConv3d(channels, channels, modes[0], modes[1], modes[2])

        # Local path (captures high-frequency details)
        self.local = nn.Conv3d(channels, channels, kernel_size=3, padding=1)

        # Combine and normalize
        self.norm1 = nn.GroupNorm(min(8, channels), channels)

        # MLP with expansion
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
        )
        self.norm2 = nn.GroupNorm(min(8, channels), channels)

        # Learnable residual scale
        self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            (B, C, D, H, W)
        """
        # Spectral + local paths with residual
        residual = x
        x = self.norm1(self.spectral(x) + self.local(x))
        x = residual + self.gamma1 * x

        # MLP with residual
        residual = x
        x = self.norm2(self.mlp(x))
        x = residual + self.gamma2 * x

        return x


class FourierJEPAEncoder3D(nn.Module):
    """3D volumetric encoder for 32×32×32 grids.

    Architecture:
    1. Lift input to hidden dimension via 3D convolutions
    2. Stack of 3D Fourier blocks (spectral + local convolutions)
    3. Patchify into 8×8×8 patches → 64 tokens
    4. Project patches to embed_dim
    5. Add 3D positional embeddings
    6. Transformer attention layers for global reasoning

    Uses 8×8×8 patches → 64 tokens, 3D Fourier blocks,
    and 3D positional embeddings.
    """

    def __init__(self, config: dict, in_channels: int = 1):
        super().__init__()

        enc_cfg = config.get("model", {}).get("encoder", {})

        self.in_channels = in_channels
        self.embed_dim = enc_cfg.get("embed_dim", 384)
        self.patch_size = enc_cfg.get("patch_size", 8)
        self.volume_size = enc_cfg.get("volume_size", 32)

        # Fourier-specific config
        fourier_cfg = enc_cfg.get("fourier", {})
        self.hidden_channels = fourier_cfg.get("hidden_channels", 64)
        self.n_fourier_layers = fourier_cfg.get("n_layers", 4)
        self.modes = tuple(fourier_cfg.get("modes_3d", fourier_cfg.get("modes", [8, 8, 8])))
        if len(self.modes) == 2:
            # If only 2D modes provided, extend to 3D
            self.modes = (self.modes[0], self.modes[1], self.modes[0])
        self.n_attention_layers = fourier_cfg.get("n_attention_layers", 2)

        # Compute grid and patch dimensions
        self.grid_size = self.volume_size // self.patch_size  # 32/8 = 4
        self.n_patches = self.grid_size ** 3  # 4*4*4 = 64
        self.patch_dim = in_channels * (self.patch_size ** 3)  # C * 8*8*8

        # 1. Lift to hidden dimension with 3D convolutions
        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(min(8, self.hidden_channels // 2), self.hidden_channels // 2),
            nn.Conv3d(self.hidden_channels // 2, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(min(8, self.hidden_channels), self.hidden_channels),
        )

        # 2. 3D Fourier blocks
        self.fourier_layers = nn.ModuleList([
            FourierBlock3D(self.hidden_channels, self.modes)
            for _ in range(self.n_fourier_layers)
        ])

        # 3. Project to embed_dim before patchifying
        self.pre_patch_proj = nn.Conv3d(
            self.hidden_channels, self.embed_dim, kernel_size=1
        )

        # 4. Learnable 3D positional embeddings (1, 64, embed_dim)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # 5. Transformer attention layers (shared architecture with 2D encoder)
        self.attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=enc_cfg.get("heads", 8),
                dim_feedforward=int(self.embed_dim * enc_cfg.get("mlp_ratio", 4.0)),
                dropout=enc_cfg.get("dropout", 0.1),
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            for _ in range(self.n_attention_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(self.embed_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def patchify(self, x: torch.Tensor) -> torch.Tensor:
        """Convert 3D volume to patch tokens via adaptive average pooling.

        Args:
            x: (B, embed_dim, D, H, W) projected features
        Returns:
            (B, n_patches, embed_dim) patch embeddings
        """
        # Pool to grid_size × grid_size × grid_size
        x = F.adaptive_avg_pool3d(x, (self.grid_size, self.grid_size, self.grid_size))
        # Flatten spatial dims: (B, embed_dim, G, G, G) → (B, embed_dim, G^3)
        x = x.flatten(2)  # (B, embed_dim, n_patches)
        # Transpose to (B, n_patches, embed_dim)
        x = x.transpose(1, 2)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) input volume (e.g., 3D permeability field)
               Default: (B, 1, 32, 32, 32)
        Returns:
            (B, 64, embed_dim) patch embeddings compatible with JEPA
        """
        B = x.shape[0]

        # 1. Lift to hidden dimension
        x = self.lift(x)  # (B, hidden_channels, D, H, W)

        # 2. 3D Fourier blocks — capture spectral structure
        for layer in self.fourier_layers:
            x = layer(x)

        # 3. Project to embed_dim
        x = self.pre_patch_proj(x)  # (B, embed_dim, D, H, W)

        # 4. Patchify: pool to grid_size^3 tokens
        x = self.patchify(x)  # (B, 64, embed_dim)

        # 5. Add 3D positional embeddings
        x = x + self.pos_embed

        # 6. Transformer attention layers
        for layer in self.attention_layers:
            x = layer(x)

        # 7. Final norm
        x = self.norm(x)

        return x
