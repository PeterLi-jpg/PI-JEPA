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
        # `decoder` may be a single Decoder/Decoder3D (legacy) or an
        # nn.ModuleList of K decoders (one per predictor sub-operator).
        if isinstance(decoder, nn.ModuleList):
            self.decoders = decoder.to(device)
            # Back-compat: also expose self.decoder pointing at the first one.
            self.decoder = self.decoders[0]
        else:
            self.decoder = decoder.to(device)
            # Build a length-K view onto the single decoder so the train loop
            # can iterate uniformly. The same decoder is shared across stages.
            K = len(self.model.predictors)
            self.decoders = nn.ModuleList([self.decoder] * K).to(device)
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
        
        # Only train encoder, predictors, and decoder(s) (not target_encoder).
        # Use self.decoders (always a ModuleList) so per-stage decoders are
        # all registered when per_stage=True.
        params = (
            list(self.model.encoder.parameters()) +
            list(self.model.predictors.parameters()) +
            list(self.decoders.parameters())
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
        target_idx: torch.Tensor,
    ):
        """
        Forward pass for self-supervised pretraining.

        Uses PIJEPA.forward_operator_split which implements the true
        Lie-Trotter chain: each predictor stage refines the previous stage's
        output (not a fresh mask token), so K predictors really do encode K
        sub-operators of the splitting decomposition.

        Args:
            x: (B, 1, H, W) or (B, 1, D, H, W) coefficient field
            context_idx: (B, N_c) context patch indices
            target_idx:  (B, N_t) target patch indices (may contain -1 padding)

        Returns:
            z_pred:        (B, N_t, D) — final stage output (for JEPA loss)
            z_target:      (B, N_t, D) — EMA target embedding (detached)
            z_full:        (B, N, D)   — full encoder output (for decoding)
            stage_outputs: list[(B, N_t, D)] of length K — per-sub-operator predictions
        """
        # The target/context indices may contain -1 padding sentinels.
        # Determine num_patches from the encoder by encoding once.
        # Defer that to PIJEPA.forward_operator_split, which will (a) encode
        # via target_encoder under no_grad, (b) build z_context, (c) chain
        # the predictors with mask-token init.
        # Pre-clamp -1 padding to a valid range.
        # (The clamp is conservative: max index across both ctx and tgt.)
        max_idx = int(max(
            int(target_idx.max().item()) if target_idx.numel() > 0 else -1,
            int(context_idx.max().item()) if context_idx.numel() > 0 else -1,
        ))
        # We don't yet know num_patches; we'll just clamp negatives to 0 for
        # safety. The model's own ValueError check inside mask_input will
        # catch true mismatches.
        target_idx_safe = target_idx.clamp(min=0)
        context_idx_safe = context_idx.clamp(min=0)

        splitting = (
            self.config.get("model", {}).get("predictor", {}).get("splitting", "lie_trotter")
        )
        z_pred, z_target, stage_outputs, z_full = self.model.forward_operator_split(
            x, context_idx_safe, target_idx_safe, splitting=splitting
        )

        return z_pred, z_target, z_full, stage_outputs
    
    def _compute_physics_residual(
        self,
        z_decoded: torch.Tensor,
        coefficient_field: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the steady-state single-phase Darcy PDE residual:
            -∇·(K ∇p) ≈ 0
        for both 2D (4D tensors) and 3D (5D tensors).

        The decoder output is treated as a candidate pressure field p.
        The coefficient field is the permeability K. Penalising the residual
        forces the latent space to encode information that is physically
        consistent with the governing PDE.

        Args:
            z_decoded:        (B, C, H, W) or (B, C, D, H, W) decoded field
            coefficient_field: (B, 1, H, W) or (B, 1, D, H, W) permeability K

        Returns:
            Scalar physics residual loss (PDE residual + small smoothness term).
        """
        # First channel of decoded output = candidate pressure
        p = z_decoded[:, 0:1]

        # Resample p to the coefficient grid if shapes differ (decoder may
        # over-resolve via patchwise unprojection).
        if p.shape[-len(coefficient_field.shape[2:]):] != coefficient_field.shape[2:]:
            target_size = coefficient_field.shape[2:]
            mode = 'trilinear' if len(target_size) == 3 else 'bilinear'
            p = F.interpolate(p, size=target_size, mode=mode, align_corners=False)

        K = coefficient_field
        physics_cfg = self.config.get("physics", {})
        dx = float(physics_cfg.get("dx", 1.0))
        dy = float(physics_cfg.get("dy", 1.0))

        # Choose finite-difference vs spectral derivatives based on config.
        # Default: "fd" — matches the original PI-JEPA paper's behavior (which
        # the paper itself found neutral-to-harmful). "spectral" uses exact
        # FFT-based derivatives, which is the paper-contribution-(iii) fix.
        residual_type = str(physics_cfg.get("residual_type", "fd")).lower()

        if p.dim() == 4:
            # 2D path
            if residual_type == "spectral":
                from physics.darcy import _spectral_grad_2d, spectral_darcy_residual_2d
                dp_dx, dp_dy = _spectral_grad_2d(p, dx, dy)
                residual = spectral_darcy_residual_2d(p, K, q=None, dx=dx, dy=dy)
            else:
                from physics.darcy import grad_x, grad_y, divergence
                dp_dx = grad_x(p, dx)
                dp_dy = grad_y(p, dy)
                flux_x = -K * dp_dx
                flux_y = -K * dp_dy
                residual = divergence(flux_x, flux_y, dx, dy)
            grad_norm = (dp_dx ** 2 + dp_dy ** 2).mean()

        elif p.dim() == 5:
            # 3D path
            dz = float(physics_cfg.get("dz", 1.0))
            if residual_type == "spectral":
                from physics.darcy import _spectral_grad_3d, spectral_darcy_residual_3d
                dp_dx, dp_dy, dp_dz = _spectral_grad_3d(p, dx, dy, dz)
                residual = spectral_darcy_residual_3d(p, K, q=None, dx=dx, dy=dy, dz=dz)
            else:
                from physics.darcy import grad_x_3d, grad_y_3d, grad_z_3d, divergence_3d
                dp_dx = grad_x_3d(p, dx)
                dp_dy = grad_y_3d(p, dy)
                dp_dz = grad_z_3d(p, dz)
                flux_x = -K * dp_dx
                flux_y = -K * dp_dy
                flux_z = -K * dp_dz
                residual = divergence_3d(flux_x, flux_y, flux_z, dx, dy, dz)
            grad_norm = (dp_dx ** 2 + dp_dy ** 2 + dp_dz ** 2).mean()
        else:
            raise ValueError(
                f"_compute_physics_residual expects 4D or 5D decoded field; got dim={p.dim()}"
            )

        pde_loss = (residual ** 2).mean()
        # smoothness regularizer to prevent the decoder from over-fitting
        # high-frequency garbage; weight is config-driven so the ablation
        # can sweep it (paper's "no smoothness" row).
        smoothness_weight = float(physics_cfg.get("smoothness_weight", 0.01))
        smoothness_loss = smoothness_weight * grad_norm
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
                
                # Forward pass (true Lie-Trotter chain via forward_operator_split)
                z_pred, z_target, z_full, stage_outputs = self._forward_pretraining(
                    x, context_idx, target_idx
                )

                # AC 1.7: Compute JEPA loss (L2 with stop-gradient on target)
                jepa_loss = compute_jepa_loss(z_pred, z_target, normalize=True)

                # AC 2.5: VICReg regularization on final stage prediction
                vicreg_losses = self.vicreg_loss(z_pred)

                # Total loss
                total_loss = jepa_loss + vicreg_losses['total']

                # Multi-fidelity Tier-1 continuation: if this batch has a
                # coarse-solver target field, add a reconstruction loss
                # against the FINAL stage's decoded output. This gives the
                # encoder + decoders cheap weak supervision from coarse sims
                # without ever touching the expensive Tier-2 simulator.
                y_coarse = batch.get("y_coarse")
                if y_coarse is not None:
                    y_coarse = y_coarse.to(self.device).float()
                    # Decode the final stage at every target patch + leave
                    # context patches as encoded. This mirrors the physics
                    # decoding path below.
                    z_recon = z_full.clone()
                    target_idx_safe = target_idx.clamp(min=0)
                    expand_target = target_idx_safe.unsqueeze(-1).expand(-1, -1, self.model.embed_dim)
                    z_recon = z_recon.scatter(1, expand_target, z_pred)
                    # Final-stage decoder (always exists; per-stage or shared)
                    x_decoded_final = self.decoders[-1](z_recon)
                    # Match the spatial shape of y_coarse if encoder/decoder
                    # imply a different resolution.
                    if x_decoded_final.shape[-len(y_coarse.shape[2:]):] != y_coarse.shape[2:]:
                        x_decoded_final = F.interpolate(
                            x_decoded_final, size=y_coarse.shape[2:],
                            mode="trilinear" if y_coarse.dim() == 5 else "bilinear",
                            align_corners=False,
                        )
                    mf_weight = float(
                        self.config.get("pretraining", {}).get("multifidelity", {}).get("weight", 0.5)
                    )
                    mf_loss = F.mse_loss(x_decoded_final[:, :y_coarse.shape[1]], y_coarse)
                    total_loss = total_loss + mf_weight * mf_loss
                    epoch_losses['mf_recon'] = epoch_losses.get('mf_recon', 0) + mf_loss.item()
                    epoch_losses['mf_weight'] = mf_weight

                # AC 2.1, 2.3: Optional per-sub-operator physics residuals with ramping.
                # For each predictor stage k, decode ẑ^(k) back to physical space and
                # evaluate the k-th sub-operator residual. Sum over K stages (paper Eq. 6).
                if physics_enabled:
                    physics_weight = self.physics_schedule.get_weight(self.global_step)
                    decoders = getattr(self, 'decoders', None)
                    if decoders is None:
                        decoders = [self.decoder] * len(stage_outputs)

                    physics_loss_total = 0.0
                    target_idx_safe = target_idx.clamp(min=0)
                    expand_target = target_idx_safe.unsqueeze(-1).expand(-1, -1, self.model.embed_dim)

                    for k, (z_k, dec_k) in enumerate(zip(stage_outputs, decoders)):
                        z_recon = z_full.clone()
                        z_recon = z_recon.scatter(1, expand_target, z_k)
                        x_decoded_k = dec_k(z_recon)
                        physics_loss_k = self._compute_physics_residual(x_decoded_k, x)
                        physics_loss_total = physics_loss_total + physics_loss_k
                        epoch_losses[f'physics_k{k}'] = epoch_losses.get(f'physics_k{k}', 0) + physics_loss_k.item()

                    total_loss = total_loss + physics_weight * physics_loss_total
                    epoch_losses['physics'] = epoch_losses.get('physics', 0) + float(physics_loss_total) if not isinstance(physics_loss_total, torch.Tensor) else epoch_losses.get('physics', 0) + physics_loss_total.item()
                    epoch_losses['physics_weight'] = physics_weight
                
                # Backward pass
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.encoder.parameters()) +
                        list(self.model.predictors.parameters()) +
                        list(self.decoders.parameters()),
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
            
            # Periodic checkpoint
            save_interval = self.config.get("pretraining", {}).get("checkpoint", {}).get("save_interval", 50)
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                    optimizer, epoch_losses
                )
        
        # Save final checkpoint
        final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        self._save_checkpoint(final_path, optimizer, epoch_losses)
        print(f"Saved final checkpoint to {final_path}")
        
        return {
            'losses': all_losses,
            'final_loss': epoch_losses['total'],
            'best_loss': best_loss,
            'checkpoint_path': final_path,
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
            # Save BOTH the legacy single-decoder field (first decoder) and the
            # full per-stage list. Old loaders that only look at 'decoder_state_dict'
            # still work; new loaders pick up 'decoder_state_dicts'.
            'decoder_state_dict': self.decoders[0].state_dict(),
            'decoder_state_dicts': [d.state_dict() for d in self.decoders],
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': self.epoch,
            'global_step': self.global_step,
            'ema_tau': self.ema_schedule.get_tau(self.epoch),
            'config': self.config,
            'metrics': metrics,
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
    # Build encoder using factory (supports vit, fourier, multiscale_fourier).
    # Read in_channels from config (CCSNet=1, FNO4CO2=12, ADR=n_species, etc.)
    in_channels = int(
        config.get("model", {}).get("encoder", {}).get("in_channels", 1)
    )
    encoder = build_encoder(config, in_channels=in_channels).to(device)
    target_encoder = build_encoder(config, in_channels=in_channels).to(device)
    print(f"  [encoder] in_channels={in_channels}")
    
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
    
    # Build decoder(s). For operator-split pretraining we want K decoders
    # (one per predictor sub-operator). The K decoders are independent params:
    # each one maps the k-th stage's latent back to physical space so that
    # the k-th sub-operator's PDE residual can be evaluated on its own
    # decoded field. If decoder.per_stage is false, a single shared decoder
    # is used and the trainer broadcasts it across stages.
    decoder_cfg = config.get("decoder", {})
    encoder_type = config.get("model", {}).get("encoder", {}).get("type", "vit").lower()
    embed_dim = decoder_cfg.get("embed_dim", config["model"]["encoder"]["embed_dim"])
    out_channels = decoder_cfg.get("out_channels", 1)
    image_size = decoder_cfg.get("image_size", config["model"]["encoder"]["image_size"])
    patch_size = decoder_cfg.get("patch_size", config["model"]["encoder"]["patch_size"])
    per_stage = bool(decoder_cfg.get("per_stage", True))  # default to per-sub-op decoders
    K = int(config["model"]["num_predictors"])

    is_3d = encoder_type in ("fourier_3d", "fourier3d")

    def _make_one():
        if is_3d:
            from models import Decoder3D
            return Decoder3D(
                embed_dim=embed_dim,
                out_channels=out_channels,
                image_size=image_size,
                patch_size=patch_size,
            )
        return Decoder(
            embed_dim=embed_dim,
            out_channels=out_channels,
            image_size=image_size,
            patch_size=patch_size,
        )

    if per_stage and K > 1:
        decoder = nn.ModuleList([_make_one() for _ in range(K)]).to(device)
        print(f"  [decoder] Built {K} per-stage decoders (per_stage=True)")
    else:
        decoder = _make_one().to(device)
        print(f"  [decoder] Built 1 shared decoder (per_stage={per_stage}, K={K})")

    return model, decoder


def build_unlabeled_dataloader(
    config: Dict[str, Any],
    split: str = "pretrain"
) -> DataLoader:
    """
    Build data loader for unlabeled coefficient fields.
    
    Args:
        config: Configuration dictionary
        split: Data split ('train', 'pretrain', 'test')
        
    Returns:
        DataLoader for unlabeled coefficient fields
    """
    pretraining_cfg = config.get("pretraining", {})
    data_cfg = config.get("data", {})
    ndim = int(data_cfg.get("ndim", 2))
    dataset_name = data_cfg.get("dataset", "darcy_flow").lower()

    batch_size = pretraining_cfg.get("batch_size", config.get("training", {}).get("batch_size", 64))

    # Multi-fidelity Darcy branch — mixes Tier-0 (unlabeled fields) with
    # Tier-1 (coarse-grid simulations). Tier-1 batches carry y_coarse.
    if dataset_name in ("darcy_3d_mf", "darcy3d_mf", "multifidelity"):
        from data.multifidelity import (
            DarcyTier1Dataset,
            build_multifidelity_loader,
        )
        mf_cfg = data_cfg.get("multifidelity", {})
        tier0_path = mf_cfg.get("tier0_path", "data/darcy_3d/darcy3d_train.pt")
        tier1_path = mf_cfg.get("tier1_path", "data/darcy_3d_tier1/darcy3d_tier1.pt")
        # Reuse the simple dict-yielding Tier-0 dataset built inline below.
        tier0_blob = torch.load(tier0_path, weights_only=False)
        x0 = tier0_blob["x"]
        n_tier0 = pretraining_cfg.get("n_unlabeled", x0.shape[0])
        x0 = x0[:n_tier0].float()

        class _Tier0Dict(torch.utils.data.Dataset):
            def __init__(self, x):
                self.x = x
            def __len__(self):
                return self.x.shape[0]
            def __getitem__(self, i):
                return {"x": self.x[i]}

        tier0_ds = _Tier0Dict(x0)
        tier1_ds = DarcyTier1Dataset(
            pt_path=tier1_path,
            n_samples=mf_cfg.get("n_tier1", None),
        )
        loader = build_multifidelity_loader(
            tier0_dataset=tier0_ds,
            tier1_dataset=tier1_ds,
            batch_size=batch_size,
            shuffle=True,
            tier1_weight=mf_cfg.get("tier1_weight", 0.5),
        )
        print(f"  [unlabeled loader] Multi-fidelity Darcy: "
              f"Tier0={len(tier0_ds)} samples, Tier1={len(tier1_ds)} samples")
        return loader

    # FNO4CO2 / U-FNO branch
    if dataset_name in ("fno4co2", "ufno"):
        from data.fno4co2_loader import load_fno4co2_unlabeled

        root = data_cfg.get("path", "data/fno4co2/dataset")
        fno_cfg = data_cfg.get("fno4co2", {})
        resize_to = fno_cfg.get("resize_to")
        if resize_to is not None:
            resize_to = tuple(resize_to)
        if ndim == 3:
            t_index = fno_cfg.get("t_index", None)
        else:
            t_index = int(fno_cfg.get("t_index", 0))
        keep_channels = fno_cfg.get("keep_channels")
        if keep_channels is not None:
            keep_channels = tuple(int(c) for c in keep_channels)
        loader = load_fno4co2_unlabeled(
            root=root,
            variant=fno_cfg.get("variant", "dP"),
            split=fno_cfg.get("split", "test"),
            n_samples=pretraining_cfg.get("n_unlabeled", None),
            t_index=t_index,
            normalize=data_cfg.get("normalize", True),
            batch_size=batch_size,
            shuffle=True,
            resize_to=resize_to,
            layout=fno_cfg.get("layout", "ctxy"),
            keep_channels=keep_channels,
        )
        print(f"  [unlabeled loader] FNO4CO2: {len(loader.dataset)} samples (ndim={ndim}, t_index={t_index})")
        return loader

    # CCSNet branch — uses the dedicated loader in PI-JEPA/data/ccsnet_loader.py
    if dataset_name == "ccsnet":
        from data.ccsnet_loader import load_ccsnet_unlabeled

        root = data_cfg.get("path", "data/ccsnet/CCSNet_v1.0")
        ccs_cfg = data_cfg.get("ccsnet", {})
        resize_to = ccs_cfg.get("resize_to")
        if resize_to is not None:
            resize_to = tuple(resize_to)
        # If ndim==3 in data config, default to keeping the time axis so the
        # samples come out as (C, T, H, W) for the 3D encoder.
        if ndim == 3:
            t_index = ccs_cfg.get("t_index", None)   # None -> keep all timesteps
        else:
            t_index = int(ccs_cfg.get("t_index", 0))
        loader = load_ccsnet_unlabeled(
            root=root,
            split=ccs_cfg.get("split", "test"),
            n_samples=pretraining_cfg.get("n_unlabeled", None),
            t_index=t_index,
            normalize=data_cfg.get("normalize", True),
            batch_size=batch_size,
            shuffle=True,
            resize_to=resize_to,
            layout=ccs_cfg.get("layout", "ctxy"),
        )
        print(f"  [unlabeled loader] CCSNet: {len(loader.dataset)} samples (ndim={ndim}, t_index={t_index})")
        return loader

    if ndim == 3:
        # 3D path: load the .pt file produced by scripts/generate_darcy_data_3d.py
        # and wrap as a tiny dict-yielding dataset.
        data_path = data_cfg.get("path", "data/darcy_3d")
        pt_path = os.path.join(data_path, "darcy3d_train.pt")
        blob = torch.load(pt_path, weights_only=False)
        x = blob["x"]  # (N, 1, D, H, W)
        n_samples = pretraining_cfg.get("n_unlabeled", x.shape[0])
        x = x[:n_samples].float()

        class _Dict3DDataset(torch.utils.data.Dataset):
            def __init__(self, x):
                self.x = x
            def __len__(self):
                return self.x.shape[0]
            def __getitem__(self, i):
                return {"x": self.x[i]}

        dataset = _Dict3DDataset(x)
        print(f"  [unlabeled loader] 3D: {len(dataset)} samples, shape per item {tuple(x[0].shape)}")

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False,
        )

    # 2D default path (unchanged)
    dataset_config = {
        'path': data_cfg.get("path", ""),
        'n_samples': pretraining_cfg.get("n_unlabeled", 1000),
        'resolution': data_cfg.get("grid_size", 64),
        'normalize': data_cfg.get("normalize", True)
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
