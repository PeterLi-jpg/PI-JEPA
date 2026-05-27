"""
Self-Supervised Pretrainer for PI-JEPA.

This module implements the self-supervised pretraining pipeline using only
unlabeled coefficient fields with spatial masking, JEPA objective, and
optional physics regularization.

Supports multiple physics modes: spectral, tpfa, latent_flux, combined.
Integrates PhysicsCurriculum, LearnedLossWeights, and AdaptiveCollocationSampler.

Validates: Requirements 1, 2 (Self-Supervised Pretraining, Physics Regularization)
"""

import os
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .masking import SpatialBlockMasker, build_spatial_block_masker
from .schedules import EMAMomentumSchedule, PhysicsWeightSchedule
from .ema import update_ema
from .curriculum import PhysicsCurriculum
from .learned_weights import LearnedLossWeights
from .adaptive_collocation import AdaptiveCollocationSampler


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
    
    Validates: AC 2.5 - Apply VICReg regularization to prevent collapse
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
    
    Validates: AC 1.7 - Compute JEPA_Objective as L2 distance with stop-gradient on target
    
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
    
    Supports physics modes: spectral, tpfa, latent_flux, combined.
    Integrates PhysicsCurriculum, LearnedLossWeights, and AdaptiveCollocationSampler.
    
    Validates: Requirements 1, 2, 5, 6, 7, 9, 10
    """
    
    def __init__(
        self,
        model: nn.Module,
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
        
        # Determine physics mode and build appropriate components
        self._physics_mode = self._get_physics_mode()
        
        # Build legacy physics schedule (used only in legacy mode)
        self.physics_schedule = self._build_physics_schedule()
        
        # Build PhysicsCurriculum (used in new physics modes)
        self.curriculum = self._build_curriculum()
        
        # Build LearnedLossWeights
        self.learned_weights = self._build_learned_weights()
        
        # Build AdaptiveCollocationSampler
        self.collocation_sampler = self._build_collocation_sampler()
        
        # Build physics modules based on mode
        self._spectral_module = None
        self._tpfa_module = None
        self._latent_flux_module = None
        self._init_physics_modules()
        
        # Build VICReg loss
        vicreg_cfg = config.get("pretraining", {}).get("vicreg", {})
        self.vicreg_loss = VICRegLoss(
            variance_weight=vicreg_cfg.get("variance_weight", 0.05),
            covariance_weight=vicreg_cfg.get("covariance_weight", 0.01)
        )
        
        # Build conditioning modules (Phase 2 — Generalization)
        self._pvt_conditioner = None
        self._brooks_corey_conditioner = None
        self._well_control_conditioner = None
        self._init_conditioning_modules()
        
        # Check 3D mode
        self._dim_3d = config.get("model", {}).get("encoder", {}).get("dim_3d", False)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
    
    def _get_physics_mode(self) -> Optional[str]:
        """Determine physics mode from config. Returns None for legacy behavior."""
        physics_cfg = self.config.get("pretraining", {}).get("physics", {})
        mode = physics_cfg.get("mode", None)
        if mode is not None and mode not in ("spectral", "tpfa", "latent_flux", "combined"):
            raise ValueError(
                f"physics.mode must be one of 'spectral', 'tpfa', 'latent_flux', 'combined', got '{mode}'"
            )
        return mode
    
    def _init_physics_modules(self) -> None:
        """Initialize physics modules based on the configured mode."""
        if self._physics_mode is None:
            return  # Legacy mode — no new physics modules
        
        physics_cfg = self.config.get("pretraining", {}).get("physics", {})
        
        if self._physics_mode in ("spectral", "combined"):
            try:
                from ..physics.spectral_residual import SpectralResidualModule
            except (ImportError, ValueError):
                from physics.spectral_residual import SpectralResidualModule
            spectral_cfg = physics_cfg.get("spectral", {})
            self._spectral_module = SpectralResidualModule(
                resolution=spectral_cfg.get("resolution", 64),
                cutoff_ratio=spectral_cfg.get("cutoff_ratio", 2 / 3),
                dx=spectral_cfg.get("dx", 1.0),
                dy=spectral_cfg.get("dy", 1.0),
            ).to(self.device)
        
        if self._physics_mode == "tpfa":
            try:
                from ..physics.tpfa import TPFALoss
            except (ImportError, ValueError):
                from physics.tpfa import TPFALoss
            tpfa_cfg = physics_cfg.get("tpfa", {})
            self._tpfa_module = TPFALoss(
                dx=tpfa_cfg.get("dx", 1.0),
                dy=tpfa_cfg.get("dy", 1.0),
            ).to(self.device)
        
        if self._physics_mode in ("latent_flux", "combined"):
            try:
                from ..physics.latent_flux import LatentFluxModule
            except (ImportError, ValueError):
                from physics.latent_flux import LatentFluxModule
            latent_flux_cfg = physics_cfg.get("latent_flux", {})
            self._latent_flux_module = LatentFluxModule(
                embed_dim=self.config.get("model", {}).get("encoder", {}).get("embed_dim", 384),
                grid_size=latent_flux_cfg.get("grid_size", 8),
                n_flux_heads=latent_flux_cfg.get("n_flux_heads", 4),
            ).to(self.device)
    
    def _init_conditioning_modules(self) -> None:
        """Initialize conditioning modules based on config (Phase 2 — Generalization).
        
        Supports:
        - PVT Conditioner: FiLM-based conditioning on fluid PVT properties
        - Brooks-Corey Conditioner: Variable relative permeability parameters
        - Well Control Conditioner: Cross-attention on well schedule tokens
        """
        conditioning_cfg = self.config.get("pretraining", {}).get("conditioning", {})
        enc_cfg = self.config.get("model", {}).get("encoder", {})
        fourier_cfg = enc_cfg.get("fourier", {})
        
        # --- PVT Conditioner ---
        pvt_cfg = conditioning_cfg.get("pvt", {})
        if pvt_cfg.get("enabled", False):
            try:
                from ..models.pvt_conditioner import PVTConditioner
            except (ImportError, ValueError):
                from models.pvt_conditioner import PVTConditioner
            
            self._pvt_conditioner = PVTConditioner(
                pvt_dim=pvt_cfg.get("dim", 3),
                hidden_channels=fourier_cfg.get("hidden_channels", 64),
                n_layers=fourier_cfg.get("n_layers", 4),
            ).to(self.device)
        
        # --- Brooks-Corey Conditioner ---
        bc_cfg = conditioning_cfg.get("brooks_corey", {})
        if bc_cfg.get("enabled", False):
            try:
                from ..models.brooks_corey_conditioner import BrooksCoreyConditioner
            except (ImportError, ValueError):
                from models.brooks_corey_conditioner import BrooksCoreyConditioner
            
            self._brooks_corey_conditioner = BrooksCoreyConditioner(
                n_params=3,  # λ, S_wr, S_nr
                hidden_channels=fourier_cfg.get("hidden_channels", 64),
                spatial=bc_cfg.get("spatial", False),
                image_size=enc_cfg.get("image_size", 64),
            ).to(self.device)
        
        # --- Well Control Conditioner ---
        wc_cfg = conditioning_cfg.get("well_control", {})
        if wc_cfg.get("enabled", False):
            try:
                from ..models.well_conditioner import WellControlConditioner
            except (ImportError, ValueError):
                from models.well_conditioner import WellControlConditioner
            
            self._well_control_conditioner = WellControlConditioner(
                well_feature_dim=wc_cfg.get("well_feature_dim", 6),
                embed_dim=enc_cfg.get("embed_dim", 384),
                n_heads=enc_cfg.get("heads", 8),
                max_wells=wc_cfg.get("max_wells", 20),
            ).to(self.device)
    
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
        """Build legacy physics weight ramping schedule from config."""
        pretraining_cfg = self.config.get("pretraining", {})
        physics_cfg = pretraining_cfg.get("physics", self.config.get("loss", {}).get("physics", {}))
        
        return PhysicsWeightSchedule(
            target_weight=physics_cfg.get("weight", 0.1),
            ramp_steps=physics_cfg.get("ramp_steps", 200)
        )
    
    def _build_curriculum(self) -> PhysicsCurriculum:
        """Build PhysicsCurriculum from config."""
        physics_cfg = self.config.get("pretraining", {}).get("physics", {})
        curriculum_cfg = physics_cfg.get("curriculum", {})
        
        return PhysicsCurriculum(
            warmup_steps=curriculum_cfg.get("warmup_steps", 1000),
            pressure_ramp_steps=curriculum_cfg.get("pressure_ramp_steps", 500),
            saturation_ramp_steps=curriculum_cfg.get("saturation_ramp_steps", 500),
            ramp_type=curriculum_cfg.get("ramp_type", "cosine"),
        )
    
    def _build_learned_weights(self) -> Optional[LearnedLossWeights]:
        """Build LearnedLossWeights if enabled in config."""
        physics_cfg = self.config.get("pretraining", {}).get("physics", {})
        lw_cfg = physics_cfg.get("learned_weights", {})
        
        if not lw_cfg.get("enabled", False):
            return None
        
        # Determine number of operators based on physics mode
        if self._physics_mode == "combined":
            num_operators = 3  # spectral_pressure, spectral_saturation, latent_flux
        elif self._physics_mode in ("spectral", "tpfa"):
            num_operators = 2  # pressure, saturation
        elif self._physics_mode == "latent_flux":
            num_operators = 1  # latent_flux only
        else:
            num_operators = 2  # default
        
        learned_weights = LearnedLossWeights(num_operators=num_operators).to(self.device)
        return learned_weights
    
    def _build_collocation_sampler(self) -> Optional[AdaptiveCollocationSampler]:
        """Build AdaptiveCollocationSampler if enabled in config."""
        physics_cfg = self.config.get("pretraining", {}).get("physics", {})
        colloc_cfg = physics_cfg.get("adaptive_collocation", {})
        
        if not colloc_cfg.get("enabled", False):
            return None
        
        return AdaptiveCollocationSampler(
            resolution=colloc_cfg.get("resolution", 64),
            n_points=colloc_cfg.get("n_points", 1024),
            min_density=colloc_cfg.get("min_density", 0.1),
            update_interval=colloc_cfg.get("update_interval", 50),
        )
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build AdamW optimizer with paper specifications.
        
        Includes separate parameter group for learned loss weights if enabled.
        Also includes parameters from conditioning modules if enabled.
        """
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
        
        # Add latent flux module parameters if present
        if self._latent_flux_module is not None:
            params = params + list(self._latent_flux_module.parameters())
        
        # Add conditioning module parameters
        if self._pvt_conditioner is not None:
            params = params + list(self._pvt_conditioner.parameters())
        if self._brooks_corey_conditioner is not None:
            params = params + list(self._brooks_corey_conditioner.parameters())
        if self._well_control_conditioner is not None:
            params = params + list(self._well_control_conditioner.parameters())
        
        param_groups = [
            {"params": params, "lr": lr, "weight_decay": weight_decay, "betas": betas}
        ]
        
        # Add learned weights as separate parameter group
        if self.learned_weights is not None:
            lw_cfg = self.config.get("pretraining", {}).get("physics", {}).get("learned_weights", {})
            lw_lr = float(lw_cfg.get("lr", 1e-3))
            param_groups.append(self.learned_weights.get_parameter_group(lr=lw_lr))
        
        return optim.AdamW(param_groups)
    
    def _forward_pretraining(
        self,
        x: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor,
        pvt_params: Optional[torch.Tensor] = None,
        brooks_corey_params: Optional[torch.Tensor] = None,
        well_controls: Optional[torch.Tensor] = None,
        well_coords: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-supervised pretraining.
        
        Args:
            x: (B, 1, H, W) or (B, 1, D, H, W) coefficient field (single channel)
            context_idx: (B, N_c) context patch indices
            target_idx: (B, N_t) target patch indices
            pvt_params: Optional (B, pvt_dim) PVT parameters for FiLM conditioning
            brooks_corey_params: Optional (B, n_params) or (B, n_params, H, W) Brooks-Corey params
            well_controls: Optional (B, N_wells, well_feature_dim) well control features
            well_coords: Optional (B, N_wells, 2) well spatial coordinates
            
        Returns:
            z_pred: (B, N_t, D) predicted target embeddings
            z_target: (B, N_t, D) EMA target embeddings (detached)
            z_full: (B, N, D) full encoded representation
        """
        B = x.shape[0]
        embed_dim = self.model.embed_dim
        
        # --- Apply PVT conditioning (FiLM) ---
        # PVT conditioning produces (gamma, beta) pairs for each Fourier block.
        # We apply FiLM after the encoder forward pass at the patch embedding level.
        film_params = None
        if self._pvt_conditioner is not None and pvt_params is not None:
            film_params = self._pvt_conditioner(pvt_params)
        
        # --- Compute Brooks-Corey conditioning features ---
        # BC features are added to the encoder output embeddings (post-encoder modulation)
        # to avoid changing encoder in_channels and breaking target encoder compatibility.
        bc_features_raw = None
        if self._brooks_corey_conditioner is not None and brooks_corey_params is not None:
            bc_features_raw = self._brooks_corey_conditioner(brooks_corey_params)  # (B, C_bc, H, W)
        
        # AC 1.5: Target_Encoder processes full coefficient field
        # Note: Target encoder uses original x (no conditioning) for stable targets
        with torch.no_grad():
            z_target_full = self.model.target_encoder(x)
        
        # AC 1.3: Apply spatial masking to partition coefficient field
        # Handle padded indices (-1) by clamping for mask_input
        target_idx_safe = target_idx.clamp(min=0)
        x_masked = self.model.mask_input(x, target_idx_safe)
        
        # AC 1.4: Context_Encoder processes only context patches
        z_full = self.model.encoder(x_masked)
        
        # Apply PVT FiLM conditioning to encoder output embeddings
        if film_params is not None:
            # Apply the last layer's (gamma, beta) to the patch embeddings
            gamma, beta = film_params[-1]  # Use last layer's params
            # gamma, beta: (B, hidden_channels)
            hc = gamma.shape[-1]
            ed = z_full.shape[-1]
            if hc == ed:
                z_full = gamma.unsqueeze(1) * z_full + beta.unsqueeze(1)
            else:
                # Apply FiLM on the first hc dimensions
                z_full_mod = z_full.clone()
                z_full_mod[:, :, :hc] = gamma.unsqueeze(1) * z_full[:, :, :hc] + beta.unsqueeze(1)
                z_full = z_full_mod
        
        # Apply Brooks-Corey conditioning as additive bias to embeddings
        # Now that we have z_full, we know the actual number of patches (N)
        if bc_features_raw is not None:
            n_patches = z_full.shape[1]
            grid_size = int(n_patches ** 0.5)
            bc_pooled = torch.nn.functional.adaptive_avg_pool2d(
                bc_features_raw, (grid_size, grid_size)
            )
            # Flatten to (B, N, hidden_channels)
            bc_embedding_bias = bc_pooled.flatten(2).transpose(1, 2)
            hc = bc_embedding_bias.shape[-1]
            ed = z_full.shape[-1]
            if hc == ed:
                z_full = z_full + bc_embedding_bias
            else:
                # Add to first hc dimensions
                z_full = z_full.clone()
                z_full[:, :, :hc] = z_full[:, :, :hc] + bc_embedding_bias
        
        # Replace target positions with mask tokens
        z = z_full.clone()
        mask_tokens = self.model.mask_token.expand(
            B, target_idx.shape[1], embed_dim
        )
        
        # Handle padded indices (-1)
        valid_mask = target_idx >= 0
        
        # Scatter mask tokens to target positions
        for b in range(B):
            valid_targets = target_idx[b][valid_mask[b]]
            if len(valid_targets) > 0:
                z[b, valid_targets] = mask_tokens[b, :len(valid_targets)]
        
        # AC 1.6: Latent_Predictor predicts target embeddings from context
        # Use safe indices for predictor operations
        context_idx_safe = context_idx.clamp(min=0)
        
        for predictor in self.model.predictors:
            z_delta, _ = predictor(z, context_idx_safe, target_idx_safe)
            
            # Gather old values at target positions
            z_old = torch.gather(
                z, 1,
                target_idx_safe.unsqueeze(-1).expand(-1, -1, embed_dim)
            )
            
            # Residual update
            z_new = z_old + 0.5 * z_delta
            
            # Scatter back
            z = z.scatter(
                1,
                target_idx_safe.unsqueeze(-1).expand(-1, -1, embed_dim),
                z_new
            )
        
        # --- Apply Well Control Conditioning via cross-attention ---
        if self._well_control_conditioner is not None and well_controls is not None and well_coords is not None:
            well_tokens = self._well_control_conditioner.encode_wells(well_controls, well_coords)
            z = self._well_control_conditioner(z, well_tokens)
        
        # Gather predicted target embeddings
        z_pred = torch.gather(
            z, 1,
            target_idx_safe.unsqueeze(-1).expand(-1, -1, embed_dim)
        )
        
        # Gather target embeddings from EMA encoder
        z_target = torch.gather(
            z_target_full, 1,
            target_idx_safe.unsqueeze(-1).expand(-1, -1, embed_dim)
        )
        
        # Normalize embeddings
        z_pred = F.layer_norm(z_pred, (embed_dim,))
        z_target = F.layer_norm(z_target, (embed_dim,))
        
        return z_pred, z_target.detach(), z_full
    
    def _compute_physics_residual(
        self,
        z_decoded: torch.Tensor,
        coefficient_field: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute physics residual on decoded coefficient field (LEGACY mode).
        
        For pretraining, we enforce consistency between decoded and
        original coefficient field, plus smoothness constraints.
        
        Args:
            z_decoded: (B, C, H, W) decoded latent representation
            coefficient_field: (B, 1, H, W) original coefficient field
            
        Returns:
            Scalar physics residual loss
        """
        # Reconstruction consistency
        if z_decoded.shape[1] > 1:
            z_decoded_coeff = z_decoded[:, 0:1]
        else:
            z_decoded_coeff = z_decoded
        
        recon_loss = F.mse_loss(z_decoded_coeff, coefficient_field)
        
        # Smoothness constraint (Laplacian regularization)
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=z_decoded_coeff.dtype, device=z_decoded_coeff.device).view(1, 1, 3, 3)
        
        laplacian = F.conv2d(z_decoded_coeff, laplacian_kernel, padding=1)
        smoothness_loss = (laplacian ** 2).mean()
        
        return recon_loss + 0.01 * smoothness_loss
    
    def _compute_new_physics_loss(
        self,
        z_pred_full: torch.Tensor,
        x_decoded: torch.Tensor,
        coefficient_field: torch.Tensor,
        step: int,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics loss using new modules (spectral/tpfa/latent_flux/combined).
        
        Dispatches to the appropriate physics module based on self._physics_mode.
        Applies curriculum weights and learned weights.
        
        Args:
            z_pred_full: (B, N, D) full latent representation (for latent_flux)
            x_decoded: (B, C, H, W) decoded fields (for spectral/tpfa)
            coefficient_field: (B, 1, H, W) original K field (used as permeability)
            step: Current training step (for curriculum)
            
        Returns:
            total_physics_loss: Scalar loss tensor
            loss_details: Dict with individual loss component values for logging
        """
        # Get curriculum weights for this step
        curriculum_weights = self.curriculum.get_weights(step)
        pressure_w = curriculum_weights["pressure"]
        saturation_w = curriculum_weights["saturation"]
        
        loss_details: Dict[str, float] = {}
        physics_losses = []
        
        # --- Spectral residual (operates on decoded fields) ---
        if self._physics_mode in ("spectral", "combined") and self._spectral_module is not None:
            # Build params dict for spectral module
            params = {
                "mu_w": 1.0,
                "mu_o": 1.0,
                "phi": 0.2,
                "dt": 1.0,
                "Sw_prev": x_decoded[:, 1:2, :, :].detach() if x_decoded.shape[1] > 1 else torch.zeros_like(coefficient_field),
            }
            spectral_loss = self._spectral_module(x_decoded, coefficient_field, params)
            # Apply curriculum: spectral loss combines pressure + saturation
            # Weight by average of pressure and saturation curriculum weights
            spectral_curriculum_w = (pressure_w + saturation_w) / 2.0
            physics_losses.append(spectral_loss * spectral_curriculum_w)
            loss_details["spectral"] = spectral_loss.item()
        
        # --- TPFA loss (operates on decoded fields) ---
        if self._physics_mode == "tpfa" and self._tpfa_module is not None:
            params = {}
            tpfa_loss = self._tpfa_module(x_decoded, coefficient_field, params)
            # TPFA is pressure-only, use pressure curriculum weight
            physics_losses.append(tpfa_loss * pressure_w)
            loss_details["tpfa"] = tpfa_loss.item()
        
        # --- Latent flux (operates on z_pred directly, no decoder needed) ---
        if self._physics_mode in ("latent_flux", "combined") and self._latent_flux_module is not None:
            latent_flux_loss = self._latent_flux_module(z_pred_full)
            # Latent flux uses pressure curriculum weight (it's a global consistency constraint)
            physics_losses.append(latent_flux_loss * pressure_w)
            loss_details["latent_flux"] = latent_flux_loss.item()
        
        # If no physics losses computed (all curriculum weights are 0), return zero
        if not physics_losses:
            zero_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
            return zero_loss, loss_details
        
        # Apply learned weights if available
        if self.learned_weights is not None:
            weights = self.learned_weights.weights
            total_physics_loss = torch.tensor(0.0, device=self.device)
            for i, ploss in enumerate(physics_losses):
                if i < len(weights):
                    total_physics_loss = total_physics_loss + weights[i] * ploss
                else:
                    total_physics_loss = total_physics_loss + ploss
            loss_details["learned_weights"] = weights.detach().cpu().tolist()
        else:
            total_physics_loss = sum(physics_losses)
        
        loss_details["total_physics"] = total_physics_loss.item()
        loss_details["curriculum_pressure"] = pressure_w
        loss_details["curriculum_saturation"] = saturation_w
        
        return total_physics_loss, loss_details
    
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
        
        grad_clip = self.config.get("training", {}).get("gradient", {}).get("clip_norm", 1.0)
        
        physics_enabled = self.config.get("pretraining", {}).get("physics", {}).get(
            "enabled", self.config.get("loss", {}).get("physics", {}).get("enabled", True)
        )
        
        # Determine if we're using new physics mode or legacy
        use_new_physics = self._physics_mode is not None and physics_enabled
        
        all_losses = []
        best_loss = float('inf')
        
        print(f"Starting self-supervised pretraining for {n_epochs} epochs")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {data_loader.batch_size}")
        print(f"  EMA: tau {self.ema_schedule.tau_start} -> {self.ema_schedule.tau_end}")
        print(f"  Physics enabled: {physics_enabled}")
        print(f"  Physics mode: {self._physics_mode or 'legacy'}")
        print(f"  3D mode: {self._dim_3d}")
        if self._pvt_conditioner is not None:
            print(f"  PVT conditioning: enabled (dim={self._pvt_conditioner.pvt_dim})")
        if self._brooks_corey_conditioner is not None:
            print(f"  Brooks-Corey conditioning: enabled (spatial={self._brooks_corey_conditioner.spatial})")
        if self._well_control_conditioner is not None:
            print(f"  Well control conditioning: enabled (max_wells={self._well_control_conditioner.max_wells})")
        if self.learned_weights is not None:
            print(f"  Learned weights: enabled ({self.learned_weights.log_weights.shape[0]} operators)")
        if self.collocation_sampler is not None:
            print(f"  Adaptive collocation: enabled (n_points={self.collocation_sampler.n_points})")
        
        for epoch in range(n_epochs):
            self.epoch = epoch
            self.model.train()
            self.decoder.train()
            if self._latent_flux_module is not None:
                self._latent_flux_module.train()
            if self._pvt_conditioner is not None:
                self._pvt_conditioner.train()
            if self._brooks_corey_conditioner is not None:
                self._brooks_corey_conditioner.train()
            if self._well_control_conditioner is not None:
                self._well_control_conditioner.train()
            
            epoch_losses = {}
            num_batches = 0
            
            for batch in data_loader:
                # AC 1.1: Load only coefficient fields x
                x = batch['x'].to(self.device).float()
                
                if x.dim() == 3:
                    x = x.unsqueeze(1)
                
                B = x.shape[0]
                
                # Extract optional conditioning data from batch
                pvt_params = batch.get('pvt_params')
                if pvt_params is not None:
                    pvt_params = pvt_params.to(self.device).float()
                
                brooks_corey_params = batch.get('brooks_corey_params')
                if brooks_corey_params is not None:
                    brooks_corey_params = brooks_corey_params.to(self.device).float()
                
                well_controls = batch.get('well_controls')
                if well_controls is not None:
                    well_controls = well_controls.to(self.device).float()
                
                well_coords = batch.get('well_coords')
                if well_coords is not None:
                    well_coords = well_coords.to(self.device).float()
                
                # AC 1.3: Sample spatial block mask
                context_idx, target_idx = self.masker.sample_mask(B, self.device)
                
                # Forward pass (with optional conditioning)
                z_pred, z_target, z_full = self._forward_pretraining(
                    x, context_idx, target_idx,
                    pvt_params=pvt_params,
                    brooks_corey_params=brooks_corey_params,
                    well_controls=well_controls,
                    well_coords=well_coords,
                )
                
                # AC 1.7: Compute JEPA loss
                jepa_loss = compute_jepa_loss(z_pred, z_target, normalize=True)
                
                # AC 2.5: VICReg regularization
                vicreg_losses = self.vicreg_loss(z_pred)
                
                total_loss = jepa_loss + vicreg_losses['total']
                
                # Physics loss computation
                if use_new_physics:
                    # --- New physics mode (spectral/tpfa/latent_flux/combined) ---
                    # In 3D mode, disable spectral/tpfa physics (not extended to 3D)
                    # Only latent_flux works in 3D since it operates on embeddings
                    skip_decoded_physics = self._dim_3d
                    
                    # Build full z representation for latent flux
                    z_recon = z_full.clone()
                    z_recon = z_recon.scatter(
                        1,
                        target_idx.clamp(min=0).unsqueeze(-1).expand(-1, -1, self.model.embed_dim),
                        z_pred
                    )
                    
                    # Decode for spectral/tpfa (only if needed and not in 3D mode)
                    x_decoded = None
                    if self._physics_mode in ("spectral", "tpfa", "combined") and not skip_decoded_physics:
                        x_decoded = self.decoder(z_recon)
                    
                    # Compute new physics loss
                    if skip_decoded_physics and self._physics_mode in ("spectral", "tpfa"):
                        # 3D mode with spectral/tpfa only — skip physics entirely
                        physics_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
                        physics_details = {"skipped_3d": True}
                    else:
                        physics_loss, physics_details = self._compute_new_physics_loss(
                            z_pred_full=z_recon,
                            x_decoded=x_decoded if x_decoded is not None else torch.zeros(B, 2, 64, 64, device=self.device),
                            coefficient_field=x,
                            step=self.global_step,
                        )
                    
                    total_loss = total_loss + physics_loss
                    
                    # Log physics details
                    for k, v in physics_details.items():
                        if isinstance(v, (int, float)):
                            epoch_losses[f'physics_{k}'] = epoch_losses.get(f'physics_{k}', 0) + v
                    
                    # Update adaptive collocation distribution periodically
                    if (self.collocation_sampler is not None and 
                        x_decoded is not None and
                        self.global_step % self.collocation_sampler.update_interval == 0):
                        with torch.no_grad():
                            self.collocation_sampler.update_distribution(x_decoded.detach())
                    
                elif physics_enabled:
                    # --- Legacy physics mode ---
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
                
                optimizer.zero_grad()
                total_loss.backward()
                
                # Gradient clipping
                clip_params = (
                    list(self.model.encoder.parameters()) +
                    list(self.model.predictors.parameters()) +
                    list(self.decoder.parameters())
                )
                if self._latent_flux_module is not None:
                    clip_params = clip_params + list(self._latent_flux_module.parameters())
                if self.learned_weights is not None:
                    clip_params = clip_params + list(self.learned_weights.parameters())
                if self._pvt_conditioner is not None:
                    clip_params = clip_params + list(self._pvt_conditioner.parameters())
                if self._brooks_corey_conditioner is not None:
                    clip_params = clip_params + list(self._brooks_corey_conditioner.parameters())
                if self._well_control_conditioner is not None:
                    clip_params = clip_params + list(self._well_control_conditioner.parameters())
                
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(clip_params, grad_clip)
                
                optimizer.step()
                
                # AC 1.8: Update EMA with annealed momentum
                tau = self.ema_schedule.get_tau(epoch)
                update_ema(self.model.encoder, self.model.target_encoder, tau=tau)
                
                epoch_losses['jepa'] = epoch_losses.get('jepa', 0) + jepa_loss.item()
                epoch_losses['variance'] = epoch_losses.get('variance', 0) + vicreg_losses['variance'].item()
                epoch_losses['covariance'] = epoch_losses.get('covariance', 0) + vicreg_losses['covariance'].item()
                epoch_losses['total'] = epoch_losses.get('total', 0) + total_loss.item()
                
                self.global_step += 1
                num_batches += 1
            
            for k in epoch_losses:
                if k not in ('physics_weight', 'physics_curriculum_pressure', 'physics_curriculum_saturation'):
                    epoch_losses[k] /= max(num_batches, 1)
            
            epoch_losses['tau'] = tau
            all_losses.append(epoch_losses)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                log_msg = (
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Loss: {epoch_losses['total']:.4f} | "
                    f"JEPA: {epoch_losses['jepa']:.4f} | "
                    f"τ: {tau:.4f}"
                )
                if use_new_physics:
                    log_msg += f" | mode: {self._physics_mode}"
                elif physics_enabled:
                    log_msg += f" | λ_p: {epoch_losses.get('physics_weight', 0):.4f}"
                print(log_msg)
            
            if epoch_losses['total'] < best_loss:
                best_loss = epoch_losses['total']
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, "checkpoint_best.pt"),
                    optimizer, epoch_losses
                )
            
            save_interval = self.config.get("pretraining", {}).get("checkpoint", {}).get("save_interval", 50)
            if (epoch + 1) % save_interval == 0:
                self._save_checkpoint(
                    os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pt"),
                    optimizer, epoch_losses
                )
        
        # Log final learned weight values at training completion
        if self.learned_weights is not None:
            final_weights = self.learned_weights.weights.detach().cpu().tolist()
            print(f"Final learned loss weights: {final_weights}")
            # Store in results for downstream use
            weight_names = self._get_weight_names()
            for i, (name, val) in enumerate(zip(weight_names, final_weights)):
                print(f"  {name}: {val:.6f}")
        
        final_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")
        self._save_checkpoint(final_path, optimizer, epoch_losses)
        print(f"Saved final checkpoint to {final_path}")
        
        results = {
            'losses': all_losses,
            'final_loss': epoch_losses['total'],
            'best_loss': best_loss,
            'checkpoint_path': final_path,
            'n_epochs': n_epochs,
            'global_step': self.global_step
        }
        
        if self.learned_weights is not None:
            results['final_learned_weights'] = dict(
                zip(self._get_weight_names(), self.learned_weights.weights.detach().cpu().tolist())
            )
        
        return results
    
    def _get_weight_names(self) -> list:
        """Get descriptive names for each learned weight based on physics mode."""
        if self._physics_mode == "combined":
            return ["spectral", "latent_flux", "extra"][:self.learned_weights.log_weights.shape[0]]
        elif self._physics_mode in ("spectral", "tpfa"):
            return ["pressure", "saturation"][:self.learned_weights.log_weights.shape[0]]
        elif self._physics_mode == "latent_flux":
            return ["latent_flux"][:self.learned_weights.log_weights.shape[0]]
        else:
            return [f"weight_{i}" for i in range(self.learned_weights.log_weights.shape[0])]
    
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
            'metrics': metrics,
            'physics_mode': self._physics_mode,
        }
        
        # Save learned weights state if present
        if self.learned_weights is not None:
            checkpoint['learned_weights_state_dict'] = self.learned_weights.state_dict()
        
        # Save latent flux module state if present
        if self._latent_flux_module is not None:
            checkpoint['latent_flux_state_dict'] = self._latent_flux_module.state_dict()
        
        # Save conditioning module states if present
        if self._pvt_conditioner is not None:
            checkpoint['pvt_conditioner_state_dict'] = self._pvt_conditioner.state_dict()
        if self._brooks_corey_conditioner is not None:
            checkpoint['brooks_corey_conditioner_state_dict'] = self._brooks_corey_conditioner.state_dict()
        if self._well_control_conditioner is not None:
            checkpoint['well_control_conditioner_state_dict'] = self._well_control_conditioner.state_dict()
        
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(checkpoint, path)


# ============================================================================
# Model Building Utilities
# ============================================================================

def build_model_for_pretraining(
    config: Dict[str, Any],
    device: torch.device
) -> Tuple[nn.Module, nn.Module]:
    """
    Build PI-JEPA model configured for self-supervised pretraining.
    
    Uses single-channel encoder for coefficient-only input.
    Supports 3D mode (FourierJEPAEncoder3D) and Brooks-Corey extra channels.
    
    Args:
        config: Configuration dictionary
        device: Training device
        
    Returns:
        Tuple of (PIJEPA model, Decoder)
    """
    # Import here to avoid circular imports
    from ..models import ViTEncoder, Predictor, PIJEPA, Decoder
    
    enc_cfg = config.get("model", {}).get("encoder", {})
    dim_3d = enc_cfg.get("dim_3d", False)
    
    # Encoder always uses in_channels=1 for coefficient field input.
    # Conditioning (PVT, Brooks-Corey) is applied post-encoder as embedding modulation.
    in_channels = 1
    
    # Validate 3D config consistency
    if dim_3d:
        volume_size = enc_cfg.get("volume_size", 32)
        patch_size = enc_cfg.get("patch_size", 8)
        if volume_size % patch_size != 0:
            raise ValueError(
                f"volume_size ({volume_size}) must be divisible by patch_size ({patch_size})"
            )
    
    # Build encoder based on mode
    if dim_3d:
        from ..models import FourierJEPAEncoder3D
        encoder = FourierJEPAEncoder3D(config, in_channels=in_channels).to(device)
        target_encoder = FourierJEPAEncoder3D(config, in_channels=in_channels).to(device)
    else:
        encoder = ViTEncoder(config, in_channels=in_channels).to(device)
        target_encoder = ViTEncoder(config, in_channels=in_channels).to(device)
    
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
    
    # Build decoder — dispatch 3D when encoder is 3D, else 2D Decoder.
    decoder_cfg = config.get("decoder", {})
    enc_image_size = config["model"]["encoder"]["image_size"]
    enc_patch_size = config["model"]["encoder"]["patch_size"]
    if dim_3d:
        from ..models import Decoder3D
        # 3D image_size: prefer encoder.volume_size if set, else fall back to
        # image_size (cubic). Decoder3D handles rectangular lists too.
        vol_size = enc_cfg.get("volume_size", enc_image_size)
        decoder = Decoder3D(
            embed_dim=decoder_cfg.get("embed_dim",
                                      config["model"]["encoder"]["embed_dim"]),
            out_channels=decoder_cfg.get("out_channels", 1),
            image_size=decoder_cfg.get("image_size", vol_size),
            patch_size=decoder_cfg.get("patch_size", enc_patch_size),
        ).to(device)
    else:
        decoder = Decoder(
            embed_dim=decoder_cfg.get("embed_dim",
                                      config["model"]["encoder"]["embed_dim"]),
            out_channels=decoder_cfg.get("out_channels", 1),
            image_size=decoder_cfg.get("image_size", enc_image_size),
            patch_size=decoder_cfg.get("patch_size", enc_patch_size),
        ).to(device)

    return model, decoder


def validate_conditioning_config(config: Dict[str, Any]) -> None:
    """Validate conditioning and 3D configuration consistency.
    
    Raises ValueError if configuration is inconsistent.
    
    Checks:
    - volume_size must be divisible by patch_size when dim_3d is True
    - physics.mode must be valid
    - curriculum.ramp_type must be valid
    """
    enc_cfg = config.get("model", {}).get("encoder", {})
    dim_3d = enc_cfg.get("dim_3d", False)
    
    if dim_3d:
        volume_size = enc_cfg.get("volume_size", 32)
        patch_size = enc_cfg.get("patch_size", 8)
        if volume_size % patch_size != 0:
            raise ValueError(
                f"model.encoder.volume_size ({volume_size}) must be divisible by "
                f"model.encoder.patch_size ({patch_size})"
            )
    
    # Validate physics mode
    physics_cfg = config.get("pretraining", {}).get("physics", {})
    mode = physics_cfg.get("mode", None)
    valid_modes = (None, "spectral", "tpfa", "latent_flux", "combined")
    if mode not in valid_modes:
        raise ValueError(
            f"pretraining.physics.mode must be one of {valid_modes}, got '{mode}'"
        )
    
    # Validate curriculum ramp type
    curriculum_cfg = physics_cfg.get("curriculum", {})
    ramp_type = curriculum_cfg.get("ramp_type", "cosine")
    valid_ramp_types = ("linear", "cosine", "step")
    if ramp_type not in valid_ramp_types:
        raise ValueError(
            f"pretraining.physics.curriculum.ramp_type must be one of {valid_ramp_types}, "
            f"got '{ramp_type}'"
        )
    
    # Validate conditioning config
    conditioning_cfg = config.get("pretraining", {}).get("conditioning", {})
    
    pvt_cfg = conditioning_cfg.get("pvt", {})
    if pvt_cfg.get("enabled", False):
        pvt_dim = pvt_cfg.get("dim", 3)
        if pvt_dim < 1:
            raise ValueError(f"conditioning.pvt.dim must be >= 1, got {pvt_dim}")
    
    wc_cfg = conditioning_cfg.get("well_control", {})
    if wc_cfg.get("enabled", False):
        max_wells = wc_cfg.get("max_wells", 20)
        if max_wells < 1:
            raise ValueError(f"conditioning.well_control.max_wells must be >= 1, got {max_wells}")


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
    from ..data.loaders import UnlabeledDarcyDataset
    
    pretraining_cfg = config.get("pretraining", {})
    data_cfg = config.get("data", {})
    
    dataset_config = {
        'path': data_cfg.get("path", ""),
        'n_samples': pretraining_cfg.get("n_unlabeled", 1000),
        'resolution': data_cfg.get("grid_size", 64),
        'normalize': data_cfg.get("normalize", True)
    }
    
    dataset = UnlabeledDarcyDataset(config=dataset_config, split=split)
    
    batch_size = pretraining_cfg.get("batch_size", config.get("training", {}).get("batch_size", 64))
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
