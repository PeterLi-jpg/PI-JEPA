"""FNO (Fourier Neural Operator) baseline."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class SpectralConv2d(nn.Module):
    """2D spectral convolution in Fourier space."""
    
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
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
    
    def compl_mul2d(self, input: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input, weights)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Clamp modes to input resolution
        modes1 = min(self.modes1, H)
        modes2 = min(self.modes2, W // 2 + 1)
        
        # Compute FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            B, self.out_channels, H, W // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :modes1, :modes2] = self.compl_mul2d(
            x_ft[:, :, :modes1, :modes2], self.weights1[:, :, :modes1, :modes2]
        )
        out_ft[:, :, -modes1:, :modes2] = self.compl_mul2d(
            x_ft[:, :, -modes1:, :modes2], self.weights2[:, :, :modes1, :modes2]
        )
        
        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(H, W))
        
        return x


class FNOBlock(nn.Module):
    """FNO block with spectral convolution and skip connection."""
    
    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.conv = nn.Conv2d(channels, channels, 1)
        self.norm = nn.InstanceNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.spectral_conv(x)
        x2 = self.conv(x)
        out = self.norm(x1 + x2)
        return F.gelu(out)


class FNO2d(nn.Module):
    """2D Fourier Neural Operator."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        modes: Tuple[int, int] = (16, 16),
        hidden_channels: int = 64,
        n_layers: int = 4
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.hidden_channels = hidden_channels
        
        self.lift = nn.Conv2d(in_channels, hidden_channels, 1)
        self.layers = nn.ModuleList([
            FNOBlock(hidden_channels, modes[0], modes[1])
            for _ in range(n_layers)
        ])
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, out_channels, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Lift to hidden dimension
        x = self.lift(x)
        
        # FNO layers
        for layer in self.layers:
            x = layer(x)
        
        # Project to output
        x = self.proj(x)
        
        return x


def fix_shape(x):
    """Ensure input tensor has shape (B, C, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4:
        pass  # already (B, C, H, W)
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.shape}")
    return x.contiguous()


class FNOWrapper:
    """Wrapper for FNO model."""
    
    def __init__(
        self,
        device: torch.device,
        modes: Tuple[int, int] = (16, 16),
        hidden_channels: int = 64,
        in_channels: int = 1,
        out_channels: int = 1,
        n_layers: int = 4
    ):
        self.device = device
        self.model = FNO2d(
            in_channels=in_channels,
            out_channels=out_channels,
            modes=modes,
            hidden_channels=hidden_channels,
            n_layers=n_layers
        ).to(device)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train_model(self, loader, epochs: int, lr: float):
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                k = fix_shape(batch["x"].to(self.device).float())
                u = fix_shape(batch["y"].to(self.device).float())
                
                pred = self.model(k)
                loss = self.loss_fn(pred, u)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"[FNO] Epoch {epoch+1}/{epochs} Loss: {total_loss:.6f}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        x = fix_shape(x)
        with torch.no_grad():
            return self.model(x)
    
    def eval(self):
        self.model.eval()
