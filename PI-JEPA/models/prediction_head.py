"""Prediction head for mapping encoder embeddings to solution fields during finetuning."""

import math
import torch
import torch.nn as nn
from torch import Tensor


class PredictionHead(nn.Module):
    """Prediction head for mapping coefficient embeddings to solutions.
    
    This module takes encoder output embeddings (B, N, D) where N is the number
    of patches and D is the embedding dimension, and maps them to a full-resolution
    solution field (B, C, H, W).
    
    The architecture consists of:
    1. Linear projection from embed_dim to hidden_dim
    2. Reshape to spatial grid (B, hidden_dim, H_patches, W_patches)
    3. Transposed convolution upsampling to full resolution
    """
    
    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 512,
        output_channels: int = 1,
        image_size: int = 64,
        patch_size: int = 8
    ):
        """Initialize prediction head.
        
        Args:
            embed_dim: Input embedding dimension from encoder
            hidden_dim: Hidden dimension for upsampling network
            output_channels: Number of output channels (1 for pressure)
            image_size: Output spatial resolution
            patch_size: Patch size used by encoder
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.output_channels = output_channels
        self.image_size = image_size
        self.patch_size = patch_size
        
        # Number of patches per side and total
        self.grid_size = image_size // patch_size  # 8 for 64/8
        self.n_patches = self.grid_size ** 2  # 64 patches
        
        # Project embeddings from embed_dim to hidden_dim
        self.proj = nn.Linear(embed_dim, hidden_dim)
        
        # Build upsampling CNN dynamically based on patch_size
        # We need to upsample from grid_size to image_size, which is patch_size times
        # Each transposed conv with stride=2 doubles the spatial size
        # So we need log2(patch_size) stages
        upsample_factor = patch_size  # e.g., 8 for 64/8
        num_stages = int(math.log2(upsample_factor))
        
        layers = []
        in_channels = hidden_dim
        
        # Channel progression: hidden_dim -> 256 -> 128 -> 64 -> ... -> output_channels
        channel_schedule = [256, 128, 64, 32, 16]
        
        for i in range(num_stages):
            out_channels = channel_schedule[min(i, len(channel_schedule) - 1)]
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.GELU(),
                nn.InstanceNorm2d(out_channels),
            ])
            in_channels = out_channels
        
        # Final conv to get output channels
        layers.append(nn.Conv2d(in_channels, output_channels, kernel_size=3, padding=1))
        
        self.upsample = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Kaiming initialization."""
        # Initialize linear projection
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        
        # Initialize conv layers
        for m in self.upsample.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, z: Tensor) -> Tensor:
        """Map encoder embeddings to solution field.
        
        Args:
            z: (B, N, D) encoder output embeddings where:
               - B is batch size
               - N is number of patches (should be grid_size^2)
               - D is embedding dimension (should be embed_dim)
            
        Returns:
            y_pred: (B, C, H, W) predicted solution field where:
                    - C is output_channels
                    - H, W are image_size
        """
        B, N, D = z.shape
        
        # Validate input dimensions
        if N != self.n_patches:
            raise ValueError(
                f"Expected {self.n_patches} patches, got {N}. "
                f"Check that image_size ({self.image_size}) and "
                f"patch_size ({self.patch_size}) match the encoder."
            )
        
        if D != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, got {D}."
            )
        
        # Project embeddings: (B, N, D) -> (B, N, hidden_dim)
        z = self.proj(z)
        
        # Reshape to spatial grid: (B, N, hidden_dim) -> (B, hidden_dim, grid_size, grid_size)
        # First reshape to (B, grid_size, grid_size, hidden_dim)
        z = z.view(B, self.grid_size, self.grid_size, self.hidden_dim)
        # Then permute to (B, hidden_dim, grid_size, grid_size)
        z = z.permute(0, 3, 1, 2).contiguous()
        
        # Upsample to full resolution: (B, hidden_dim, 8, 8) -> (B, C, 64, 64)
        y_pred = self.upsample(z)
        
        return y_pred
