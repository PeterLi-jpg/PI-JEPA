"""DeepONet and U-DeepONet baselines with branch-trunk framework."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


def fix_shape(x):
    """Ensure input tensor has shape (B, C, H, W)."""
    if x.ndim == 3:
        x = x.unsqueeze(1)
    elif x.ndim == 4:
        pass
    else:
        raise ValueError(f"Expected 3D or 4D tensor, got {x.shape}")
    return x.contiguous()


class DeepONet(nn.Module):
    """Standard DeepONet with branch-trunk architecture."""
    
    def __init__(
        self,
        branch_input_dim: int = 64 * 64,
        trunk_input_dim: int = 2,
        hidden_dim: int = 256,
        latent_dim: int = 128
    ):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Linear(branch_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (B, C, H, W) -> output (B, 1, H, W)."""
        B, C, H, W = x.shape
        branch = self.branch(x.view(B, -1))
        
        grid = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, H, device=x.device),
            torch.linspace(0, 1, W, device=x.device),
            indexing='ij'
        ), dim=-1).view(-1, 2)
        
        trunk = self.trunk(grid)
        out = torch.einsum("bi,ni->bn", branch, trunk)
        return out.view(B, 1, H, W)


class DeepONetWrapper:
    """Wrapper for standard DeepONet."""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.model = None  # Created lazily on first batch
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
    
    def _ensure_model(self, input_dim):
        if self.model is None:
            self.model = DeepONet(branch_input_dim=input_dim).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train_model(self, loader, epochs: int, lr: float):
        """Train DeepONet model."""
        for epoch in range(epochs):
            for batch in loader:
                k = fix_shape(batch["x"].to(self.device).float())
                u = fix_shape(batch["y"].to(self.device).float())
                
                B = k.shape[0]
                self._ensure_model(k.view(B, -1).shape[1])
                self.model.train()
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) if epoch == 0 else self.optimizer
                
                pred = self.model(k)
                loss = self.loss_fn(pred, u)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction with trained model."""
        self.model.eval()
        x = fix_shape(x)
        with torch.no_grad():
            return self.model(x)


class UNetEncoderBranch(nn.Module):
    """U-Net encoder for branch network in U-DeepONet."""
    
    def __init__(
        self,
        in_channels: int = 1,
        channels: Optional[List[int]] = None,
        latent_dim: int = 128
    ):
        super().__init__()
        
        if channels is None:
            channels = [64, 128, 256]
        
        self.channels = channels
        
        # Encoder blocks
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        
        prev_ch = in_channels
        for ch in channels:
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU(),
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU()
                )
            )
            self.pools.append(nn.MaxPool2d(2))
            prev_ch = ch
        
        # Global pooling and projection to latent space
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(channels[-1], latent_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Input (B, C, H, W) -> latent (B, latent_dim)."""
        for block, pool in zip(self.blocks, self.pools):
            x = block(x)
            x = pool(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.proj(x)
        
        return x


class TrunkMLP(nn.Module):
    """MLP trunk network mapping coordinates to latent space."""
    
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        n_layers: int = 3
    ):
        super().__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, latent_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """Forward pass. Coords (N, input_dim) -> latent (N, latent_dim)."""
        return self.net(coords)


class UDeepONet(nn.Module):
    """U-Net enhanced DeepONet with branch-trunk framework."""
    
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        branch_channels: Optional[List[int]] = None,
        trunk_hidden: int = 128,
        latent_dim: int = 128,
        trunk_layers: int = 3
    ):
        super().__init__()
        
        if branch_channels is None:
            branch_channels = [64, 128, 256]
        
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        
        # Branch network: U-Net encoder
        self.branch = UNetEncoderBranch(
            in_channels=in_channels,
            channels=branch_channels,
            latent_dim=latent_dim * out_channels  # Multiple outputs
        )
        
        # Trunk network: MLP for coordinates
        self.trunk = TrunkMLP(
            input_dim=2,
            hidden_dim=trunk_hidden,
            latent_dim=latent_dim,
            n_layers=trunk_layers
        )
        
        # Bias term for each output channel
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def forward(
        self,
        x: torch.Tensor,
        coords: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass. Input (B, C, H, W) -> output (B, out_channels, H, W)."""
        B, C, H, W = x.shape
        
        # Branch: encode input function
        branch_out = self.branch(x)  # (B, latent_dim * out_channels)
        branch_out = branch_out.view(B, self.out_channels, self.latent_dim)  # (B, out_channels, latent_dim)
        
        # Create coordinate grid if not provided
        if coords is None:
            y_coords = torch.linspace(0, 1, H, device=x.device)
            x_coords = torch.linspace(0, 1, W, device=x.device)
            yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
            coords = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (H*W, 2)
        
        # Trunk: encode coordinates
        trunk_out = self.trunk(coords)  # (H*W, latent_dim)
        
        # Compute output: dot product of branch and trunk
        # branch_out: (B, out_channels, latent_dim)
        # trunk_out: (H*W, latent_dim)
        # output: (B, out_channels, H*W)
        output = torch.einsum('bcd,nd->bcn', branch_out, trunk_out)
        
        # Add bias
        output = output + self.bias.view(1, -1, 1)
        
        # Reshape to spatial dimensions
        output = output.view(B, self.out_channels, H, W)
        
        return output


class UDeepONetWrapper:
    """U-DeepONet wrapper for training and prediction."""
    
    def __init__(
        self,
        device: torch.device,
        in_channels: int = 1,
        out_channels: int = 1,
        branch_channels: Optional[List[int]] = None,
        trunk_hidden: int = 128,
        latent_dim: int = 128,
        trunk_layers: int = 3
    ):
        self.device = device
        
        if branch_channels is None:
            branch_channels = [64, 128, 256]
        
        self.model = UDeepONet(
            in_channels=in_channels,
            out_channels=out_channels,
            branch_channels=branch_channels,
            trunk_hidden=trunk_hidden,
            latent_dim=latent_dim,
            trunk_layers=trunk_layers
        ).to(device)
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def train_model(
        self,
        loader,
        epochs: int,
        lr: float
    ):
        """Train U-DeepONet model."""
        self.model.train()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for batch in loader:
                x = fix_shape(batch["x"].to(self.device).float())
                y = fix_shape(batch["y"].to(self.device).float())
                
                pred = self.model(x)
                loss = self.loss_fn(pred, y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            print(f"[U-DeepONet] Epoch {epoch+1}/{epochs} Loss: {total_loss:.6f}")
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Make prediction with trained model."""
        self.model.eval()
        x = fix_shape(x)
        with torch.no_grad():
            return self.model(x)
    
    def eval(self):
        """Set model to evaluation mode."""
        self.model.eval()
