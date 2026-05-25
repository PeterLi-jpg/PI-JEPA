"""
Fourier-JEPA Encoder: Physics-Aware Encoder for Subsurface Flow

This module implements a novel encoder that combines:
1. Fourier Neural Operator layers (captures spectral PDE structure)
2. Transformer attention (captures long-range spatial dependencies)
3. JEPA-compatible output format (patch embeddings)

Key innovations:
- Spectral convolutions preserve physical frequency content
- Multi-scale feature extraction via different Fourier modes
- Physics-informed positional encoding based on spatial coordinates
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union


def _as_triple(value: Union[int, Tuple[int, int, int], list]) -> Tuple[int, int, int]:
    """Coerce a scalar or 3-element iterable into a 3-tuple of ints.

    Used to interpret image_size / patch_size config values: a scalar means
    cubic, a 3-list means rectangular (d, h, w).
    """
    if isinstance(value, int):
        return (value, value, value)
    if hasattr(value, "__len__") and len(value) == 3:
        return tuple(int(v) for v in value)
    raise ValueError(f"Expected int or 3-tuple; got {value!r}")


def _safe_num_groups(channels: int, max_groups: int = 8) -> int:
    """Return the largest divisor of `channels` that is <= max_groups.

    GroupNorm requires num_channels % num_groups == 0. Naive `min(8, ch)`
    breaks when channels is e.g. 12 (12 % 8 != 0). This helper picks the
    largest valid divisor instead.
    """
    if channels < 1:
        return 1
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


class SpectralConv2d(nn.Module):
    """2D Spectral convolution layer operating in Fourier space."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes1: int,
        modes2: int
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        
        # Learnable Fourier weights
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Clamp modes to input resolution
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W // 2 + 1)
        
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(B, self.out_channels, H, W // 2 + 1, 
                           dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2], 
            self.weights1[:, :, :modes1, :modes2]
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2], 
            self.weights2[:, :, :modes1, :modes2]
        )
        
        # Inverse FFT
        return torch.fft.irfft2(out_ft, s=(H, W))


class FourierBlock(nn.Module):
    """Combined Fourier + local convolution block with residual connection."""
    
    def __init__(
        self,
        channels: int,
        modes: Tuple[int, int] = (16, 16),
        mlp_ratio: float = 2.0
    ):
        super().__init__()
        
        # Spectral path
        self.spectral = SpectralConv2d(channels, channels, modes[0], modes[1])
        
        # Local path (captures high-frequency details)
        self.local = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # Combine and normalize
        self.norm1 = nn.GroupNorm(min(8, channels), channels)
        
        # MLP with larger expansion
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, channels, 1)
        )
        self.norm2 = nn.GroupNorm(min(8, channels), channels)
        
        # Learnable residual scale
        self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spectral + local paths with residual
        residual = x
        x = self.norm1(self.spectral(x) + self.local(x))
        x = residual + self.gamma1 * x
        
        # MLP with residual
        residual = x
        x = self.norm2(self.mlp(x))
        x = residual + self.gamma2 * x
        
        return x


class PatchifyFourier(nn.Module):
    """Convert spatial features to patch embeddings for JEPA compatibility."""
    
    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        patch_size: int = 8
    ):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial features
        Returns:
            (B, N, D) patch embeddings where N = (H/patch_size) * (W/patch_size)
        """
        x = self.proj(x)  # (B, D, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class FourierJEPAEncoder(nn.Module):
    """
    Fourier-JEPA Encoder for subsurface flow simulation.
    
    Architecture:
    1. Lift input to hidden dimension
    2. Stack of Fourier blocks (spectral + local convolutions)
    3. Optional transformer layers for global attention
    4. Patchify to JEPA-compatible format
    
    This design captures:
    - Global spectral structure (Fourier layers)
    - Local heterogeneity (local convolutions)  
    - Long-range dependencies (attention)
    - JEPA compatibility (patch embeddings)
    """
    
    def __init__(
        self,
        config: dict,
        in_channels: int = 1
    ):
        super().__init__()
        
        enc_cfg = config.get("model", {}).get("encoder", {})
        
        self.in_channels = in_channels
        self.embed_dim = enc_cfg.get("embed_dim", 384)
        self.patch_size = enc_cfg.get("patch_size", 8)
        self.image_size = enc_cfg.get("image_size", 64)
        
        # Fourier-specific config
        fourier_cfg = enc_cfg.get("fourier", {})
        self.hidden_channels = fourier_cfg.get("hidden_channels", 64)
        self.n_fourier_layers = fourier_cfg.get("n_layers", 4)
        self.modes = tuple(fourier_cfg.get("modes", [16, 16]))
        self.use_attention = fourier_cfg.get("use_attention", True)
        self.n_attention_layers = fourier_cfg.get("n_attention_layers", 2)
        
        # 1. Lift to hidden dimension with better initialization
        self.lift = nn.Sequential(
            nn.Conv2d(in_channels, self.hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(min(8, self.hidden_channels // 2), self.hidden_channels // 2),
            nn.Conv2d(self.hidden_channels // 2, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(min(8, self.hidden_channels), self.hidden_channels)
        )
        
        # 2. Fourier blocks
        self.fourier_layers = nn.ModuleList([
            FourierBlock(self.hidden_channels, self.modes)
            for _ in range(self.n_fourier_layers)
        ])
        
        # 3. Project to embed_dim before patchifying
        self.pre_patch_proj = nn.Conv2d(
            self.hidden_channels, self.embed_dim, kernel_size=1
        )
        
        # 4. Patchify for JEPA compatibility
        # Note: We patchify AFTER Fourier processing, so patch_size=1 here
        # The "patches" are just spatial positions
        self.grid_size = self.image_size // self.patch_size
        self.n_patches = self.grid_size ** 2
        
        # Learnable position embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 5. Optional attention layers for global reasoning
        if self.use_attention:
            self.attention_layers = nn.ModuleList([
                nn.TransformerEncoderLayer(
                    d_model=self.embed_dim,
                    nhead=enc_cfg.get("heads", 8),
                    dim_feedforward=int(self.embed_dim * enc_cfg.get("mlp_ratio", 4.0)),
                    dropout=enc_cfg.get("dropout", 0.1),
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input field (e.g., permeability)
        Returns:
            (B, N, D) patch embeddings compatible with JEPA
        """
        B = x.shape[0]
        
        # 1. Lift to hidden dimension
        x = self.lift(x)  # (B, hidden, H, W)
        
        # 2. Fourier blocks - capture spectral structure
        for layer in self.fourier_layers:
            x = layer(x)
        
        # 3. Project to embed_dim
        x = self.pre_patch_proj(x)  # (B, embed_dim, H, W)
        
        # 4. Patchify: pool to grid_size x grid_size
        x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        
        # 5. Add position embeddings
        x = x + self.pos_embed
        
        # 6. Optional attention layers
        if self.use_attention:
            for layer in self.attention_layers:
                x = layer(x)
        
        # 7. Final norm
        x = self.norm(x)
        
        return x
    
    def get_intermediate_features(self, x: torch.Tensor) -> dict:
        """Get intermediate features for analysis/visualization."""
        features = {}
        
        x = self.lift(x)
        features['after_lift'] = x.clone()
        
        for i, layer in enumerate(self.fourier_layers):
            x = layer(x)
            features[f'fourier_{i}'] = x.clone()
        
        x = self.pre_patch_proj(x)
        features['after_proj'] = x.clone()
        
        return features


class MultiScaleFourierEncoder(nn.Module):
    """
    Multi-scale Fourier encoder that processes at multiple resolutions.
    
    This captures both fine-grained heterogeneity and large-scale flow patterns.
    """
    
    def __init__(
        self,
        config: dict,
        in_channels: int = 1
    ):
        super().__init__()
        
        enc_cfg = config.get("model", {}).get("encoder", {})
        self.embed_dim = enc_cfg.get("embed_dim", 384)
        self.patch_size = enc_cfg.get("patch_size", 8)
        self.image_size = enc_cfg.get("image_size", 64)
        
        # Multi-scale branches
        self.scales = [1, 2, 4]  # Process at 1x, 1/2x, 1/4x resolution
        hidden = 64
        
        self.branches = nn.ModuleList()
        for scale in self.scales:
            modes = (16 // scale, 16 // scale)
            branch = nn.Sequential(
                nn.Conv2d(in_channels, hidden, 3, padding=1),
                nn.GELU(),
                FourierBlock(hidden, modes),
                FourierBlock(hidden, modes),
            )
            self.branches.append(branch)
        
        # Fusion
        self.fusion = nn.Conv2d(hidden * len(self.scales), self.embed_dim, 1)
        
        # Patchify
        self.grid_size = self.image_size // self.patch_size
        self.n_patches = self.grid_size ** 2
        
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, self.embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.norm = nn.LayerNorm(self.embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Process at multiple scales
        features = []
        for scale, branch in zip(self.scales, self.branches):
            if scale > 1:
                x_scaled = F.avg_pool2d(x, scale)
            else:
                x_scaled = x
            
            feat = branch(x_scaled)
            
            # Upsample back to original resolution
            if scale > 1:
                feat = F.interpolate(feat, size=(H, W), mode='bilinear', align_corners=False)
            
            features.append(feat)
        
        # Concatenate and fuse
        x = torch.cat(features, dim=1)
        x = self.fusion(x)
        
        # Patchify
        x = F.adaptive_avg_pool2d(x, (self.grid_size, self.grid_size))
        x = x.flatten(2).transpose(1, 2)

        x = x + self.pos_embed
        x = self.norm(x)

        return x


# =============================================================================
# 3D variants for volumetric subsurface fields (D, H, W) — added for the 3D
# follow-up paper. The 2D code above is preserved unchanged so that both
# regimes can be benchmarked side by side.
# =============================================================================


class SpectralConv3d(nn.Module):
    """3D spectral convolution operating in (k_d, k_h, k_w) Fourier space.

    Mirrors SpectralConv2d but with three frequency axes. We keep four weight
    tensors covering the (+,+,+), (+,+,-), (+,-,+), (-,+,+) corner blocks of
    the rFFT spectrum (the last dim is single-sided via rfftn; negative-w
    content is conjugate of positive-w content by Hermitian symmetry).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        modes_d: int,
        modes_h: int,
        modes_w: int,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_d = modes_d
        self.modes_h = modes_h
        self.modes_w = modes_w

        scale = 1 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_d, modes_h, modes_w)
        self.w_ppp = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w_pmp = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w_mpp = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))
        self.w_mmp = nn.Parameter(scale * torch.rand(*shape, dtype=torch.cfloat))

    @staticmethod
    def _cmul(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: (B, in, d, h, w)  w: (in, out, d, h, w)
        return torch.einsum("bidhw,iodhw->bodhw", x, w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        md = min(self.modes_d, D)
        mh = min(self.modes_h, H)
        mw = min(self.modes_w, W // 2 + 1)

        x_ft = torch.fft.rfftn(x, dim=(-3, -2, -1))
        out_ft = torch.zeros(
            B, self.out_channels, D, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device,
        )

        # +d +h +w corner
        out_ft[:, :, :md, :mh, :mw] = self._cmul(
            x_ft[:, :, :md, :mh, :mw], self.w_ppp[:, :, :md, :mh, :mw]
        )
        # +d -h +w corner
        out_ft[:, :, :md, -mh:, :mw] = self._cmul(
            x_ft[:, :, :md, -mh:, :mw], self.w_pmp[:, :, :md, :mh, :mw]
        )
        # -d +h +w corner
        out_ft[:, :, -md:, :mh, :mw] = self._cmul(
            x_ft[:, :, -md:, :mh, :mw], self.w_mpp[:, :, :md, :mh, :mw]
        )
        # -d -h +w corner
        out_ft[:, :, -md:, -mh:, :mw] = self._cmul(
            x_ft[:, :, -md:, -mh:, :mw], self.w_mmp[:, :, :md, :mh, :mw]
        )

        return torch.fft.irfftn(out_ft, s=(D, H, W), dim=(-3, -2, -1))


class FourierBlock3D(nn.Module):
    """3D Fourier + local conv block, residual."""

    def __init__(
        self,
        channels: int,
        modes: Tuple[int, int, int] = (8, 8, 8),
        mlp_ratio: float = 2.0,
    ):
        super().__init__()
        self.spectral = SpectralConv3d(channels, channels, *modes)
        self.local = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(_safe_num_groups(channels), channels)
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv3d(channels, hidden, 1),
            nn.GELU(),
            nn.Conv3d(hidden, channels, 1),
        )
        self.norm2 = nn.GroupNorm(_safe_num_groups(channels), channels)
        self.gamma1 = nn.Parameter(torch.ones(1) * 0.1)
        self.gamma2 = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(self.spectral(x) + self.local(x))
        x = residual + self.gamma1 * x

        residual = x
        x = self.norm2(self.mlp(x))
        x = residual + self.gamma2 * x
        return x


class FourierJEPAEncoder3D(nn.Module):
    """3D Fourier + attention encoder producing JEPA patch tokens.

    Input:  (B, C_in, D, H, W).
    Output: (B, N, embed_dim) with N = (D/P) * (H/P) * (W/P), P = patch_size.

    Defaults assume small smoke-test grids (32^3, patch=4 → 8^3 = 512 tokens).
    """

    def __init__(self, config: dict, in_channels: int = 1):
        super().__init__()

        enc_cfg = config.get("model", {}).get("encoder", {})

        self.in_channels = in_channels
        self.embed_dim = enc_cfg.get("embed_dim", 256)

        # image_size and patch_size may be scalars (cubic) or 3-tuples
        # (rectangular). The rectangular path supports CCSNet / FNO4CO2
        # native shapes like (T=24, H=96, W=200) with patches like (4, 8, 8).
        image_size = enc_cfg.get("image_size", 32)
        patch_size = enc_cfg.get("patch_size", 4)
        self.image_size_dhw = _as_triple(image_size)
        self.patch_size_dhw = _as_triple(patch_size)
        # Keep scalar attrs for back-compat with pi_jepa.PIJEPA which reads
        # encoder.patch_size; pick the smallest axis to be conservative.
        self.patch_size = min(self.patch_size_dhw)
        self.image_size = max(self.image_size_dhw)

        fourier_cfg = enc_cfg.get("fourier", {})
        self.hidden_channels = fourier_cfg.get("hidden_channels", 32)
        self.n_fourier_layers = fourier_cfg.get("n_layers", 3)
        default_modes = [8, 8, 8]
        modes = fourier_cfg.get("modes", default_modes)
        if len(modes) == 2:
            modes = list(modes) + [modes[-1]]  # pad 2D config to 3D
        self.modes = tuple(modes)
        self.use_attention = fourier_cfg.get("use_attention", True)
        self.n_attention_layers = fourier_cfg.get("n_attention_layers", 2)

        self.lift = nn.Sequential(
            nn.Conv3d(in_channels, self.hidden_channels // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(_safe_num_groups(self.hidden_channels // 2), self.hidden_channels // 2),
            nn.Conv3d(self.hidden_channels // 2, self.hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.GroupNorm(_safe_num_groups(self.hidden_channels), self.hidden_channels),
        )

        self.fourier_layers = nn.ModuleList([
            FourierBlock3D(self.hidden_channels, self.modes)
            for _ in range(self.n_fourier_layers)
        ])

        self.pre_patch_proj = nn.Conv3d(
            self.hidden_channels, self.embed_dim, kernel_size=1
        )

        # Token grid is now (gd, gh, gw) — possibly rectangular
        gd = self.image_size_dhw[0] // self.patch_size_dhw[0]
        gh = self.image_size_dhw[1] // self.patch_size_dhw[1]
        gw = self.image_size_dhw[2] // self.patch_size_dhw[2]
        self.grid_size_dhw = (gd, gh, gw)
        self.grid_size = max(gd, gh, gw)  # legacy scalar — used only by some
                                          # downstream code that assumes cubic
        self.n_patches = gd * gh * gw

        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.n_patches, self.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if self.use_attention:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D, H, W) volumetric input
        Returns:
            (B, N, embed_dim) token sequence
        """
        if x.dim() != 5:
            raise ValueError(
                f"FourierJEPAEncoder3D expects 5D input (B, C, D, H, W); got shape {tuple(x.shape)}"
            )

        x = self.lift(x)
        for layer in self.fourier_layers:
            x = layer(x)
        x = self.pre_patch_proj(x)

        # Rectangular grid: pool along each axis to its own target size.
        # The Fourier layers preserve spatial dims; we compute per-axis
        # kernels from the configured patch sizes. avg_pool3d is MPS-safe;
        # adaptive_avg_pool3d is the fallback (CPU on MPS).
        gd, gh, gw = self.grid_size_dhw
        D_in, H_in, W_in = x.shape[-3], x.shape[-2], x.shape[-1]
        k_d = max(1, D_in // gd)
        k_h = max(1, H_in // gh)
        k_w = max(1, W_in // gw)
        if D_in == gd * k_d and H_in == gh * k_h and W_in == gw * k_w:
            x = F.avg_pool3d(x, kernel_size=(k_d, k_h, k_w), stride=(k_d, k_h, k_w))
        else:
            x = F.adaptive_avg_pool3d(x, (gd, gh, gw))
        x = x.flatten(2).transpose(1, 2)  # (B, N=gd*gh*gw, embed_dim)

        x = x + self.pos_embed

        if self.use_attention:
            for layer in self.attention_layers:
                x = layer(x)

        return self.norm(x)
