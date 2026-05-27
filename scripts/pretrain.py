#!/usr/bin/env python
"""
Self-Supervised Pretraining Script for PI-JEPA.

This script implements the self-supervised pretraining phase using only
unlabeled coefficient fields (permeability K) without requiring solution
fields (pressure/saturation).

Paper specifications:
- 500 epochs pretraining with batch size 64
- AdamW: lr=1.5×10^-4, weight_decay=5×10^-2
- EMA momentum annealing: τ from 0.99 to 0.999 over first 10% epochs
- Physics weight ramping over first 200 steps
- VICReg regularization to prevent embedding collapse

Validates: Requirements 1, 2 (Self-Supervised Pretraining, Physics Regularization)
- AC 1.1: Load only coefficient fields x without requiring solution fields y
- AC 1.3: Apply spatial masking to partition coefficient field into context/target
- AC 1.4: Context_Encoder processes only context patches
- AC 1.5: Target_Encoder processes full coefficient field
- AC 1.6: Latent_Predictor predicts target embeddings from context embeddings
- AC 1.7: Compute JEPA_Objective as L2 distance with stop-gradient on target
- AC 1.8: Update Target_Encoder via EMA with momentum τ annealed 0.99→0.999
- AC 2.1: Decode predicted embeddings to physical space
- AC 2.3: Ramp physics weight λ_p from 0 to 0.1 over first 200 steps
- AC 2.5: Apply VICReg regularization to prevent collapse
"""

import os
import sys
import argparse
import math
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

# Add PI-JEPA directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PI-JEPA"))

from models import ViTEncoder, Predictor, PIJEPA, Decoder, build_encoder
from training import (
    SpatialBlockMasker,
    build_spatial_block_masker,
    EMAMomentumSchedule,
    PhysicsWeightSchedule,
    build_ema_schedule,
    build_physics_weight_schedule,
    update_ema,
)
from data.loaders import UnlabeledDarcyDataset, DatasetFactory
from utils import load_config


# ============================================================================
# VICReg Regularization
# ============================================================================

class VICRegLoss(nn.Module):
    """
    VICReg-style regularization to prevent embedding collapse.
    
    Implements variance and covariance regularization terms:
    - Variance: Encourages embeddings to have unit variance
    - Covariance: Encourages decorrelated embedding dimensions
    
    Reference: Bardes et al., "VICReg: Variance-Invariance-Covariance 
    Regularization for Self-Supervised Learning", ICLR 2022
    """
    
    def __init__(
        self,
        variance_weight: float = 0.05,
        covariance_weight: float = 0.01,
        variance_gamma: float = 1.0,
        eps: float = 1e-4
    ):
        """
        Initialize VICReg loss.
        
        Args:
            variance_weight: Weight for variance loss term
            covariance_weight: Weight for covariance loss term
            variance_gamma: Target standard deviation (default 1.0)
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.variance_weight = variance_weight
        self.covariance_weight = covariance_weight
        self.variance_gamma = variance_gamma
        self.eps = eps
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VICReg regularization losses.
        
        Args:
            z: (B, N, D) latent embeddings
            
        Returns:
            Dict with 'variance' and 'covariance' loss components
        """
        # Flatten to (B*N, D)
        z_flat = z.reshape(-1, z.shape[-1])
        
        # Center the embeddings
        z_centered = z_flat - z_flat.mean(dim=0, keepdim=True)
        
        # Variance loss: encourage std >= gamma
        std = torch.sqrt(z_centered.var(dim=0, unbiased=False) + self.eps)
        variance_loss = torch.mean(F.relu(self.variance_gamma - std) ** 2)
        
        # Covariance loss: encourage decorrelated dimensions
        N, D = z_centered.shape
        z_norm = z_centered / (std + self.eps)
        cov = (z_norm.T @ z_norm) / (N - 1 + self.eps)
        
        # Off-diagonal elements should be zero
        off_diag_mask = ~torch.eye(D, dtype=torch.bool, device=z.device)
        off_diag = cov[off_diag_mask]
        covariance_loss = (off_diag ** 2).mean()
        
        return {
            'variance': variance_loss,
            'covariance': covariance_loss,
            'total': self.variance_weight * variance_loss + self.covariance_weight * covariance_loss
        }


# ============================================================================
# JEPA Loss Computation
# ============================================================================

def compute_jepa_loss(
    z_pred: torch.Tensor,
    z_target: torch.Tensor,
    normalize: bool = True
) -> torch.Tensor:
    """
    Compute JEPA prediction loss (L2 distance in latent space).
    
    The target embeddings should have stop-gradient applied before
    calling this function.
    
    Args:
        z_pred: (B, N_t, D) predicted target embeddings
        z_target: (B, N_t, D) EMA target embeddings (detached)
        normalize: Whether to L2-normalize embeddings before computing loss
        
    Returns:
        Scalar loss tensor (mean L2 distance)
    """
    if normalize:
        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target, dim=-1)
    
    # L2 distance: ||z_pred - z_target||^2
    # For normalized vectors: 2 - 2 * cos_sim = ||z_pred - z_target||^2
    loss = F.mse_loss(z_pred, z_target)
    
    return loss


# ============================================================================
# Self-Supervised Pretrainer
# ============================================================================

class SelfSupervisedPretrainer:
    """
    Self-supervised pretraining pipeline for PI-JEPA.
    
    Implements the pretraining phase using only unlabeled coefficient fields
    with spatial masking, JEPA objective, and optional physics regularization.
    """
    
    def __init__(
        self,
        model: PIJEPA,
        decoder: nn.Module,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize self-supervised pretrainer.
        
        Args:
            model: PI-JEPA model with encoder, target_encoder, predictors
            decoder: Decoder for physics residual (optional)
            config: Training configuration
            device: Training device
        """
        self.model = model.to(device)
        self.decoder = decoder.to(device)
        self.config = config
        self.device = device
        
        # Build masker
        self.masker = build_spatial_block_masker(config)
        
        # Build schedules
        self.ema_schedule = self._build_ema_schedule()
        self.physics_schedule = self._build_physics_schedule()
        
        # Build VICReg loss
        vicreg_cfg = config.get("pretraining", {}).get("vicreg", {})
        self.vicreg_loss = VICRegLoss(
            variance_weight=vicreg_cfg.get("variance_weight", 0.05),
            covariance_weight=vicreg_cfg.get("covariance_weight", 0.01)
        )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def _build_ema_schedule(self) -> EMAMomentumSchedule:
        """Build EMA momentum schedule from config."""
        pretraining_cfg = self.config.get("pretraining", {})
        ema_cfg = pretraining_cfg.get("ema", self.config.get("ema", {}).get("schedule", {}))
        
        return EMAMomentumSchedule(
            tau_start=ema_cfg.get("tau_start", 0.99),
            tau_end=ema_cfg.get("tau_end", 0.999),
            warmup_fraction=ema_cfg.get("warmup_fraction", 0.1),
            total_epochs=pretraining_cfg.get("epochs", self.config.get("training", {}).get("epochs", 500))
        )
    
    def _build_physics_schedule(self) -> PhysicsWeightSchedule:
        """Build physics weight ramping schedule from config."""
        pretraining_cfg = self.config.get("pretraining", {})
        physics_cfg = pretraining_cfg.get("physics", self.config.get("loss", {}).get("physics", {}))
        
        return PhysicsWeightSchedule(
            target_weight=physics_cfg.get("weight", 0.1),
            ramp_steps=physics_cfg.get("ramp_steps", 200)
        )
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build AdamW optimizer with paper specifications."""
        pretraining_cfg = self.config.get("pretraining", {})
        optim_cfg = pretraining_cfg.get("optim", self.config.get("training", {}).get("optim", {}))
        
        lr = float(optim_cfg.get("lr", 1.5e-4))
        weight_decay = float(optim_cfg.get("weight_decay", 5e-2))
        betas = tuple(optim_cfg.get("betas", [0.9, 0.95]))
        
        # Only train encoder, predictors, and decoder (not target_encoder)
        params = (
            list(self.model.encoder.parameters()) +
            list(self.model.predictors.parameters()) +
            list(self.decoder.parameters())
        )
        
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas
        )
    
    def _forward_pretraining(
        self,
        x: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-supervised pretraining.
        
        Args:
            x: (B, 1, H, W) coefficient field (single channel)
            context_idx: (B, N_c) context patch indices
            target_idx: (B, N_t) target patch indices
            
        Returns:
            z_pred: (B, N_t, D) predicted target embeddings
            z_target: (B, N_t, D) EMA target embeddings (detached)
            z_full: (B, N, D) full encoded representation
        """
        B = x.shape[0]
        
        # Encode full field with target encoder (EMA, stop gradient)
        # AC 1.5: Target_Encoder processes full coefficient field
        with torch.no_grad():
            z_target_full = self.model.target_encoder(x)
        
        # Get number of patches from encoder output
        num_patches = z_target_full.shape[1]
        
        # AC 1.3: Apply spatial masking to partition coefficient field
        # Handle padded indices (-1) and clamp to valid range
        target_idx_safe = target_idx.clamp(min=0, max=num_patches - 1)
        x_masked = self.model.mask_input(x, target_idx_safe)
        
        # Encode masked input with context encoder
        # AC 1.4: Context_Encoder processes only context patches
        z_full = self.model.encoder(x_masked)
        
        # Verify encoder output matches expected number of patches
        assert z_full.shape[1] == num_patches, f"Encoder output mismatch: {z_full.shape[1]} vs {num_patches}"
        
        # Replace target positions with mask tokens
        z = z_full.clone()
        mask_tokens = self.model.mask_token.expand(
            B, target_idx.shape[1], self.model.embed_dim
        )
        
        # Handle padded indices (-1) and clamp to valid range
        valid_mask = target_idx >= 0
        
        # Scatter mask tokens to target positions using safe indexing
        for b in range(B):
            valid_targets = target_idx[b][valid_mask[b]]
            if len(valid_targets) > 0:
                # Clamp indices to valid range for this encoder output
                valid_targets_clamped = valid_targets.clamp(0, num_patches - 1)
                z[b, valid_targets_clamped] = mask_tokens[b, :len(valid_targets)]
        
        # AC 1.6: Latent_Predictor predicts target embeddings from context
        # Use safe indices for predictor operations
        context_idx_safe = context_idx.clamp(min=0, max=num_patches - 1)
        
        for predictor in self.model.predictors:
            z_delta, _ = predictor(z, context_idx_safe, target_idx_safe)
            
            # Gather old values at target positions
            z_old = torch.gather(
                z, 1,
                target_idx_safe.unsqueeze(-1).expand(-1, -1, self.model.embed_dim)
            )
            
            # Residual update
            z_new = z_old + 0.5 * z_delta
            
            # Scatter back
            z = z.scatter(
                1,
                target_idx_safe.unsqueeze(-1).expand(-1, -1, self.model.embed_dim),
                z_new
            )
        
        # Gather predicted target embeddings
        z_pred = torch.gather(
            z, 1,
            target_idx_safe.unsqueeze(-1).expand(-1, -1, self.model.embed_dim)
        )
        
        # Gather target embeddings from EMA encoder
        z_target = torch.gather(
            z_target_full, 1,
            target_idx_safe.unsqueeze(-1).expand(-1, -1, self.model.embed_dim)
        )
        
        # Normalize embeddings
        z_pred = F.layer_norm(z_pred, (self.model.embed_dim,))
        z_target = F.layer_norm(z_target, (self.model.embed_dim,))
        
        return z_pred, z_target.detach(), z_full
    
    def _compute_physics_residual(
        self,
        z_decoded: torch.Tensor,
        coefficient_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Darcy PDE residual: -∇·(K ∇p) ≈ 0.

        The decoder output is treated as a candidate pressure field p.
        The coefficient field is the permeability K.  Penalising the
        residual forces the latent space to encode information that is
        physically consistent with the governing PDE.

        Args:
            z_decoded: (B, C, H, W) decoded field — treated as pressure p
            coefficient_field: (B, 1, H, W) permeability K

        Returns:
            Scalar physics residual loss
        """
        from physics.darcy import grad_x, grad_y, divergence

        # Use first channel of decoded output as pressure
        p = z_decoded[:, 0:1]

        # Match spatial resolution
        if p.shape[-2:] != coefficient_field.shape[-2:]:
            p = F.interpolate(
                p, size=coefficient_field.shape[-2:],
                mode='bilinear', align_corners=False
            )

        K = coefficient_field  # (B, 1, H, W)

        physics_cfg = self.config.get("physics", {})
        dx = float(physics_cfg.get("dx", 1.0))
        dy = float(physics_cfg.get("dy", 1.0))

        # Darcy flux: q = -K ∇p
        dp_dx = grad_x(p, dx)
        dp_dy = grad_y(p, dy)

        flux_x = -K * dp_dx
        flux_y = -K * dp_dy

        # Residual: -∇·(K ∇p) = ∇·q  (should be ≈ 0 for steady-state)
        residual = divergence(flux_x, flux_y, dx, dy)

        pde_loss = (residual ** 2).mean()

        # Mild gradient norm penalty to keep decoded field smooth
        # (replaces the old recon_loss that incorrectly pushed p toward K)
        grad_norm = (dp_dx ** 2 + dp_dy ** 2).mean()
        smoothness_loss = 0.01 * grad_norm

        return pde_loss + smoothness_loss
    
    def pretrain(
        self,
        data_loader: DataLoader,
        n_epochs: int = 500,
        checkpoint_dir: str = "outputs/pretrain"
    ) -> Dict[str, Any]:
        """
        Run self-supervised pretraining.
        
        Args:
            data_loader: Unlabeled coefficient field loader
            n_epochs: Number of pretraining epochs (default 500)
            checkpoint_dir: Directory for saving checkpoints
            
        Returns:
            Dict with training metrics and final checkpoint path
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        optimizer = self._build_optimizer()
        
        # Get gradient clipping config
        grad_clip = self.config.get("training", {}).get("gradient", {}).get("clip_norm", 1.0)
        
        # Physics enabled?
        physics_enabled = self.config.get("pretraining", {}).get("physics", {}).get(
            "enabled", self.config.get("loss", {}).get("physics", {}).get("enabled", True)
        )
        
        # Training metrics
        all_losses = []
        best_loss = float('inf')
        
        print(f"Starting self-supervised pretraining for {n_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {data_loader.batch_size}")
        print(f"  EMA: tau {self.ema_schedule.tau_start} -> {self.ema_schedule.tau_end}")
        print(f"  Physics enabled: {physics_enabled}")
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            self.model.train()
            self.decoder.train()
            
            epoch_losses = {}
            num_batches = 0
            
            for batch in data_loader:
                # AC 1.1: Load only coefficient fields x
                x = batch['x'].to(self.device).float()
                
                # Ensure shape is (B, 1, H, W)
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                
                B = x.shape[0]
                
                # AC 1.3: Sample spatial block mask
                context_idx, target_idx = self.masker.sample_mask(B, self.device)
                
                # Forward pass
                z_pred, z_target, z_full = self._forward_pretraining(
                    x, context_idx, target_idx
                )
                
                # AC 1.7: Compute JEPA loss (L2 with stop-gradient on target)
                jepa_loss = compute_jepa_loss(z_pred, z_target, normalize=True)
                
                # AC 2.5: VICReg regularization
                vicreg_losses = self.vicreg_loss(z_pred)
                
                # Total loss
                total_loss = jepa_loss + vicreg_losses['total']
                
                # AC 2.1, 2.3: Optional physics residual with ramping
                if physics_enabled:
                    # Decode predicted embeddings
                    z_recon = z_full.clone()
                    z_recon = z_recon.scatter(
                        1,
                        target_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.model.embed_dim),
                        z_pred
                    )
                    x_decoded = self.decoder(z_recon)
                    
                    physics_loss = self._compute_physics_residual(x_decoded, x)
                    physics_weight = self.physics_schedule.get_weight(self.global_step)
                    total_loss = total_loss + physics_weight * physics_loss
                    
                    epoch_losses['physics'] = epoch_losses.get('physics', 0) + physics_loss.item()
                    epoch_losses['physics_weight'] = physics_weight
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.encoder.parameters()) +
                        list(self.model.predictors.parameters()) +
                        list(self.decoder.parameters()),
                        grad_clip
                    )
                
                optimizer.step()
                
                # AC 1.8: Update EMA with annealed momentum
                tau = self.ema_schedule.get_tau(epoch)
                update_ema(self.model.encoder, self.model.target_encoder, tau=tau)
                
                # Accumulate losses
                epoch_losses['jepa'] = epoch_losses.get('jepa', 0) + jepa_loss.item()
                epoch_losses['variance'] = epoch_losses.get('variance', 0) + vicreg_losses['variance'].item()
                epoch_losses['covariance'] = epoch_losses.get('covariance', 0) + vicreg_losses['covariance'].item()
                epoch_losses['total'] = epoch_losses.get('total', 0) + total_loss.item()
                
                self.global_step += 1
                num_batches += 1
            
            # Average losses
            for k in epoch_losses:
                if k != 'physics_weight':
                    epoch_losses[k] /= max(num_batches, 1)
            
            epoch_losses['tau'] = tau
            all_losses.append(epoch_losses)
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = (
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Loss: {epoch_losses['total']:.4f} | "
                    f"JEPA: {epoch_losses['jepa']:.4f} | "
                    f"tau: {tau:.4f}"
                )
                if physics_enabled:
                    log_msg += f" | lambda_p: {epoch_losses.get('physics_weight', 0):.4f}"
                print(log_msg)
            
            # Save best checkpoint
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, "checkpoint_best.pt"),
                    optimizer, epoch_losses
                )
            
            # Periodic checkpoint — only save best to conserve disk
            # (epoch checkpoints disabled to avoid filling disk during ablations)
        
        # Final checkpoint — skip if best already saved (save disk)
        print(f"Pretraining complete. Best checkpoint in {checkpoint_dir}")
        
        best_path = os.path.join(checkpoint_dir, "checkpoint_best.pt")
        return {
            'losses': all_losses,
            'final_loss': epoch_losses['total'],
            'best_loss': best_loss,
            'checkpoint_path': best_path,
            'n_epochs': n_epochs,
            'global_step': self.global_step
        }
    
    def _save_checkpoint(
        self,
        path: str,
        optimizer: optim.Optimizer,
        metrics: Dict[str, float]
    ) -> None:
        """Save pretraining checkpoint."""
        checkpoint = {
            'checkpoint_type': 'pretraining',
            'encoder_state_dict': self.model.encoder.state_dict(),
            'target_encoder_state_dict': self.model.target_encoder.state_dict(),
            'predictor_state_dicts': [p.state_dict() for p in self.model.predictors],
            'decoder_state_dict': self.decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'ema_tau': self.ema_schedule.get_tau(self.epoch),
            'config': self.config,
            'metrics': metrics
        }
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(checkpoint, path)


# ============================================================================
# Model Building
# ============================================================================

def build_model_for_pretraining(
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[PIJEPA, Decoder]:
    """
    Build PI-JEPA model configured for self-supervised pretraining.
    
    Uses single-channel encoder for coefficient-only input.
    
    Args:
        config: Configuration dictionary
        device: Training device
        
    Returns:
        Tuple of (PIJEPA model, Decoder)
    """
    # Build encoder using factory (supports vit, fourier, multiscale_fourier)
    encoder = build_encoder(config, in_channels=1).to(device)
    target_encoder = build_encoder(config, in_channels=1).to(device)
    
    # Build predictors
    predictors = [
        Predictor(config).to(device)
        for _ in range(config["model"]["num_predictors"])
    ]
    
    # Build PIJEPA model
    model = PIJEPA(
        encoder=encoder,
        target_encoder=target_encoder,
        predictors=predictors,
        embed_dim=config["model"]["encoder"]["embed_dim"],
        num_patches=None,
        patch_size=config["model"]["encoder"]["patch_size"],
    ).to(device)
    
    # Initialize target encoder from encoder
    for p in target_encoder.parameters():
        p.requires_grad = False
    target_encoder.load_state_dict(encoder.state_dict())
    
    # Build decoder
    decoder_cfg = config.get("decoder", {})
    decoder = Decoder(
        embed_dim=decoder_cfg.get("embed_dim", config["model"]["encoder"]["embed_dim"]),
        out_channels=decoder_cfg.get("out_channels", 1),  # Single channel for pretraining
        image_size=decoder_cfg.get("image_size", config["model"]["encoder"]["image_size"]),
        patch_size=decoder_cfg.get("patch_size", config["model"]["encoder"]["patch_size"])
    ).to(device)
    
    return model, decoder


class _Simple3DUnlabeledDataset(torch.utils.data.Dataset):
    """Minimal 3D unlabeled-fields dataset that reads our `.pt` format.

    Expects `<path>/darcy3d_train.pt` (or `_pretrain.pt`) with a dict
    `{"x": (N, C, D, H, W), "y": ...}`. Returns `{"x": (C, D, H, W)}` dicts
    so the pretrainer's existing 5D-aware forward path consumes it.

    Brandon's `UnlabeledDarcyDataset` is hardcoded 2D (returns `(1, H, W)`)
    and silently mis-stacks our 3D `.pt` files. We dispatch to this loader
    whenever `config.data.ndim == 3` or `config.model.encoder.type == 'fourier_3d'`.
    """

    def __init__(self, path: str, split: str = "pretrain",
                 n_samples: Optional[int] = None, normalize: bool = True,
                 resize_cube: Optional[int] = None):
        import os
        candidates = [
            os.path.join(path, "darcy3d_train.pt"),
            os.path.join(path, "darcy3d_pretrain.pt"),
            os.path.join(path, "train.pt"),
        ]
        chosen = None
        for c in candidates:
            if os.path.exists(c):
                chosen = c
                break
        if chosen is None:
            raise FileNotFoundError(
                f"_Simple3DUnlabeledDataset: no recognizable 3D .pt under {path}; "
                f"tried: {candidates}"
            )
        blob = torch.load(chosen, weights_only=False, map_location="cpu")
        x = blob["x"].float()  # (N, C, D, H, W)
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D (N, C, D, H, W) tensor in {chosen}; got {tuple(x.shape)}"
            )
        if n_samples is not None and n_samples > 0:
            x = x[:n_samples]
        if resize_cube and x.shape[-3:] != (resize_cube,) * 3:
            x = torch.nn.functional.interpolate(
                x, size=(resize_cube,) * 3, mode="trilinear", align_corners=False
            )
        if normalize:
            mean = x.mean()
            std = x.std() + 1e-8
            x = (x - mean) / std
        self.x = x
        self.split = split

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {"x": self.x[idx]}


def build_unlabeled_dataloader(
    config: Dict[str, Any],
    split: str = "pretrain"
) -> DataLoader:
    """
    Build data loader for unlabeled coefficient fields.

    Dispatches to a 3D-aware dataset when `config.data.ndim == 3` or when
    the encoder is `fourier_3d`. Brandon's original `UnlabeledDarcyDataset`
    is 2D-only and silently mis-stacks 3D `.pt` files, producing
    `(B, 4, 1, 16, 16)`-shaped tensors that crash the 3D encoder.

    Args:
        config: Configuration dictionary
        split: Data split ('train', 'pretrain', 'test')

    Returns:
        DataLoader for unlabeled coefficient fields
    """
    pretraining_cfg = config.get("pretraining", {})
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    enc_cfg = model_cfg.get("encoder", {})

    is_3d = (data_cfg.get("ndim") == 3) or (enc_cfg.get("type") == "fourier_3d")
    batch_size = pretraining_cfg.get("batch_size",
                                     config.get("training", {}).get("batch_size", 64))

    if is_3d:
        # 3D: load our `.pt` blob directly and resize to encoder cube.
        vol_size = enc_cfg.get("volume_size", data_cfg.get("grid_size", 32))
        dataset = _Simple3DUnlabeledDataset(
            path=data_cfg.get("path", "data/darcy_3d"),
            split=split,
            n_samples=pretraining_cfg.get("n_unlabeled", None),
            normalize=data_cfg.get("normalize", True),
            resize_cube=int(vol_size),
        )
    else:
        dataset_config = {
            'path': data_cfg.get("path", ""),
            'n_samples': pretraining_cfg.get("n_unlabeled", 1000),
            'resolution': data_cfg.get("grid_size", 64),
            'normalize': data_cfg.get("normalize", True),
        }
        dataset = UnlabeledDarcyDataset(config=dataset_config, split=split)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )


# ============================================================================
# Main Entry Point
# ============================================================================

def pretrain(
    config_path: str = "configs/darcy.yaml",
    output_dir: str = "outputs/pretrain"
) -> str:
    """
    Run self-supervised pretraining.
    
    Args:
        config_path: Path to YAML configuration file
        output_dir: Directory to save checkpoints
        
    Returns:
        Path to the saved final checkpoint
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set device
    if config["experiment"].get("device") is not None:
        device = torch.device(config["experiment"]["device"])
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    
    # Set seed
    seed = config["experiment"].get("seed", 42)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Build model
    model, decoder = build_model_for_pretraining(config, device)
    
    # Build data loader
    data_loader = build_unlabeled_dataloader(config, split="pretrain")
    
    # Build pretrainer
    pretrainer = SelfSupervisedPretrainer(
        model=model,
        decoder=decoder,
        config=config,
        device=device
    )
    
    # Get number of epochs
    pretraining_cfg = config.get("pretraining", {})
    n_epochs = pretraining_cfg.get("epochs", config.get("training", {}).get("epochs", 500))
    
    # Run pretraining
    results = pretrainer.pretrain(
        data_loader=data_loader,
        n_epochs=n_epochs,
        checkpoint_dir=output_dir
    )
    
    return results['checkpoint_path']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-supervised pretraining for PI-JEPA")
    parser.add_argument(
        "--config",
        default="configs/darcy.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--output",
        default="outputs/pretrain",
        help="Output directory for checkpoints"
    )
    args = parser.parse_args()
    
    checkpoint_path = pretrain(args.config, args.output)
    print(f"Pretraining complete. Checkpoint saved to: {checkpoint_path}")
