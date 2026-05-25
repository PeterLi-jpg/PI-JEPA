"""
Well Control Conditioner: Cross-attention conditioning on well control tokens.

Encodes well schedules (rates, pressures, on/off status, coordinates)
as tokens and injects via cross-attention into predictor bank patch embeddings.

Architecture:
    1. Well features (controls + spatial coords) are concatenated and projected
       to embed_dim via a linear layer to produce well tokens.
    2. Multi-head cross-attention: patch embeddings (queries) attend to
       well tokens (keys/values).
    3. Residual connection + LayerNorm for stable conditioning.

This enables the predictor to adapt predictions based on variable well
configurations, supporting production optimization workflows where well
schedules change between training and evaluation.
"""

import torch
import torch.nn as nn
from torch import Tensor


class WellControlConditioner(nn.Module):
    """Cross-attention conditioning on well control tokens.

    Encodes well schedules (rates, pressures, on/off, coordinates)
    as tokens and injects via cross-attention into predictor bank.

    Args:
        well_feature_dim: Total dimension of well features including spatial
                          coordinates. Default 6 = [x, y, rate, bhp, status, type].
                          The encode_wells method receives controls and coords
                          separately and concatenates them.
        embed_dim: Embedding dimension matching the predictor bank output.
        n_heads: Number of attention heads for cross-attention.
        max_wells: Maximum number of wells supported per sample.
    """

    def __init__(
        self,
        well_feature_dim: int = 6,  # [x, y, rate, bhp, status, type]
        embed_dim: int = 384,
        n_heads: int = 8,
        max_wells: int = 20,
    ):
        super().__init__()
        self.well_feature_dim = well_feature_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.max_wells = max_wells

        # Total input dim equals well_feature_dim (coords + controls combined)
        total_input_dim = well_feature_dim

        # Project concatenated [coords, controls] to embed_dim
        self.well_encoder = nn.Sequential(
            nn.Linear(total_input_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Layer norms for pre-normalization
        self.norm_patches = nn.LayerNorm(embed_dim)
        self.norm_wells = nn.LayerNorm(embed_dim)

        # Multi-head cross-attention: patches attend to well tokens
        # PyTorch MHA expects (seq_len, batch, embed_dim) by default,
        # but we use batch_first=True for clarity.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )

        # Post-attention LayerNorm
        self.norm_out = nn.LayerNorm(embed_dim)

    def encode_wells(self, well_controls: Tensor, well_coords: Tensor) -> Tensor:
        """Encode variable-length well controls to fixed-dim tokens.

        Spatial coordinates are concatenated with well features before
        projection to embed_dim, ensuring spatial information is included
        in the token representation.

        Args:
            well_controls: (B, N_wells, well_feature_dim) well control features
                           (rate, bhp, status, type, etc.)
            well_coords: (B, N_wells, 2) spatial coordinates (x, y)

        Returns:
            well_tokens: (B, N_wells, embed_dim) encoded well tokens
        """
        # Concatenate spatial coords with control features: [x, y, rate, bhp, status, type]
        combined = torch.cat([well_coords, well_controls], dim=-1)  # (B, N_wells, total_input_dim)

        # Project to embedding dimension
        well_tokens = self.well_encoder(combined)  # (B, N_wells, embed_dim)

        return well_tokens

    def forward(self, z_patches: Tensor, well_tokens: Tensor) -> Tensor:
        """Cross-attend patch embeddings to well control tokens.

        Uses multi-head cross-attention where patches are queries and
        well tokens are keys/values. Includes a residual connection
        so that z_out = z_patches + cross_attn(z_patches, well_tokens).

        Args:
            z_patches: (B, N_patches, embed_dim) patch embeddings from predictor
            well_tokens: (B, N_wells, embed_dim) encoded well tokens

        Returns:
            z_conditioned: (B, N_patches, embed_dim) conditioned patch embeddings
        """
        # Pre-normalization
        q = self.norm_patches(z_patches)  # (B, N_patches, embed_dim)
        kv = self.norm_wells(well_tokens)  # (B, N_wells, embed_dim)

        # Cross-attention: patches attend to well tokens
        attn_out, _ = self.cross_attn(
            query=q,
            key=kv,
            value=kv,
        )  # (B, N_patches, embed_dim)

        # Residual connection
        z_conditioned = z_patches + attn_out

        # Post-normalization
        z_conditioned = self.norm_out(z_conditioned)

        return z_conditioned
