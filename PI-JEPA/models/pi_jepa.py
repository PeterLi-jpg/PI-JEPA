import torch
import torch.nn as nn
import torch.nn.functional as F


class PIJEPA(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        target_encoder: nn.Module,
        predictors: list,
        embed_dim: int,
        num_patches: int = None,
        patch_size: int = 16,
    ):
        super().__init__()

        self.encoder = encoder
        self.target_encoder = target_encoder
        self.predictors = nn.ModuleList(predictors)

        self.embed_dim = embed_dim
        self.num_patches = num_patches
        self.patch_size = patch_size

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def mask_input(self, x, target_indices):
        """Zero out target-patch regions of x.

        Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W) inputs.
        target_indices is (B, N_t) of patch indices in row-major (or
        depth-major-row-major for 3D) order.
        """
        if x.dim() == 4:
            return self._mask_input_2d(x, target_indices)
        elif x.dim() == 5:
            return self._mask_input_3d(x, target_indices)
        else:
            raise ValueError(
                f"mask_input expects 4D (B,C,H,W) or 5D (B,C,D,H,W); got shape {tuple(x.shape)}"
            )

    def _get_patch_size(self):
        if hasattr(self.encoder, 'patch_size'):
            return self.encoder.patch_size
        return self.patch_size

    def _mask_input_2d(self, x, target_indices):
        B, C, H, W = x.shape
        patch_size = self._get_patch_size()

        grid_size = H // patch_size
        num_patches = grid_size * grid_size

        max_idx = target_indices.max().item() if target_indices.numel() > 0 else -1
        if max_idx >= num_patches:
            raise ValueError(
                f"target_indices max={max_idx} exceeds encoder's num_patches={num_patches} "
                f"(image_size={H}x{W}, patch_size={patch_size}). "
                f"Masker grid_size likely doesn't match encoder grid_size."
            )
        target_indices_clamped = target_indices.clamp(min=0)

        mask = torch.ones(B, num_patches, device=x.device)
        mask = mask.scatter(
            1,
            target_indices_clamped,
            torch.zeros_like(target_indices_clamped, dtype=mask.dtype),
        )

        mask = mask.view(B, grid_size, grid_size)
        mask = mask.repeat_interleave(patch_size, dim=1)
        mask = mask.repeat_interleave(patch_size, dim=2)
        mask = mask.unsqueeze(1)
        return x * mask

    def _mask_input_3d(self, x, target_indices):
        B, C, D, H, W = x.shape

        # Prefer the encoder's authoritative (grid_dhw, patch_dhw) — needed
        # for rectangular grids. Fall back to legacy cubic computation.
        enc = self.encoder
        if hasattr(enc, "grid_size_dhw") and hasattr(enc, "patch_size_dhw"):
            gd, gh, gw = enc.grid_size_dhw
            pd, ph, pw = enc.patch_size_dhw
        else:
            patch = self._get_patch_size()
            gd = D // patch
            gh = H // patch
            gw = W // patch
            pd = ph = pw = patch

        num_patches = gd * gh * gw

        max_idx = target_indices.max().item() if target_indices.numel() > 0 else -1
        if max_idx >= num_patches:
            raise ValueError(
                f"target_indices max={max_idx} exceeds encoder's num_patches={num_patches} "
                f"(volume={D}x{H}x{W}, patch=({pd},{ph},{pw}), grid=({gd},{gh},{gw})). "
                f"Masker grid_size likely doesn't match encoder grid_size."
            )
        target_indices_clamped = target_indices.clamp(min=0)

        mask = torch.ones(B, num_patches, device=x.device)
        mask = mask.scatter(
            1,
            target_indices_clamped,
            torch.zeros_like(target_indices_clamped, dtype=mask.dtype),
        )

        mask = mask.view(B, gd, gh, gw)
        mask = mask.repeat_interleave(pd, dim=1)
        mask = mask.repeat_interleave(ph, dim=2)
        mask = mask.repeat_interleave(pw, dim=3)
        mask = mask.unsqueeze(1)  # (B, 1, D, H, W)
        return x * mask

    def get_num_patches(self, image_size: int, ndim: int = 2) -> int:
        """Get the number of patches for a given image size and dimensionality.

        Args:
            image_size: side length of the input field
            ndim: 2 for (H, W) inputs, 3 for (D, H, W) inputs
        """
        patch_size = self._get_patch_size()
        grid_size = image_size // patch_size
        return grid_size ** ndim

    def forward(self, x, context_indices, target_indices, return_stage_outputs: bool = False):
        """Original 2D-compatible forward (kept for backward compat).

        Uses the broken additive-ensemble chain. Prefer forward_operator_split
        for true Lie-Trotter operator splitting across multiple predictors.
        """
        B = x.shape[0]

        x_masked = self.mask_input(x, target_indices)

        with torch.no_grad():
            z_target_full = self.target_encoder(x)

        z_full = self.encoder(x_masked)

        z = z_full.clone()

        mask_tokens = self.mask_token.expand(
            B, target_indices.shape[1], self.embed_dim
        )

        z = z.scatter(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            mask_tokens
        )

        for predictor in self.predictors:
            z_delta, _ = predictor(z, context_indices, target_indices)

            z_old = torch.gather(
                z,
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )

            z_new = z_old + 0.5 * z_delta

            z = z.scatter(
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
                z_new
            )

        z_pred = torch.gather(
            z,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        z_target = torch.gather(
            z_target_full,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        z_pred = F.layer_norm(z_pred, (self.embed_dim,))
        z_target = F.layer_norm(z_target, (self.embed_dim,))

        z_pred = F.normalize(z_pred, dim=-1)
        z_target = F.normalize(z_target, dim=-1)

        return z_pred, z_target

    def forward_operator_split(self, x, context_indices, target_indices, splitting: str = "lie_trotter"):
        """True operator-splitting forward in latent space.

        splitting:
          - "lie_trotter" (default): 1st-order Lie-Trotter chain
                ẑ^(k) = g_{φ_k}(ẑ^(k-1), z_context)  for k = 1..K
            One pass through each predictor in order.

          - "strang": 2nd-order Strang splitting (for K=2 only)
                ẑ^(0.5) = g_{φ_1}^{1/2}(ẑ^(0), z_context)
                ẑ^(1.5) = g_{φ_2}(ẑ^(0.5), z_context)
                ẑ^(2)   = g_{φ_1}^{1/2}(ẑ^(1.5), z_context)
            We approximate the "half step" by sharing g_{φ_1} across both
            half passes (the predictor itself is fixed; only its application
            count changes). Higher-order accuracy in the same training budget.

          - "monolithic": all predictors collapsed into a single composed
            pass (acts as an ablation against the per-sub-operator structure).

        Returns:
            z_pred, z_target, stage_outputs, z_full
        """
        B = x.shape[0]

        x_masked = self.mask_input(x, target_indices)

        with torch.no_grad():
            z_target_full = self.target_encoder(x)

        z_full = self.encoder(x_masked)

        # Gather context once (constant input to every predictor in the chain)
        D = self.embed_dim
        z_context = torch.gather(
            z_full,
            1,
            context_indices.unsqueeze(-1).expand(-1, -1, D),
        )

        # Initialize chain at target positions from this model's mask_token
        N_t = target_indices.shape[1]
        z_t = self.mask_token.expand(B, N_t, D).contiguous()

        stage_outputs = []
        K = len(self.predictors)
        if splitting == "strang" and K == 2:
            # Strang: g_φ1^{1/2} → g_φ2 → g_φ1^{1/2}
            # Implemented as 3 passes using 2 distinct predictor params.
            z_t = self.predictors[0].forward_chained(z_t, z_context)
            stage_outputs.append(z_t)  # ẑ^(0.5) — "half" first pass
            z_t = self.predictors[1].forward_chained(z_t, z_context)
            stage_outputs.append(z_t)  # ẑ^(1.5) — middle full pass
            z_t = self.predictors[0].forward_chained(z_t, z_context)
            stage_outputs.append(z_t)  # ẑ^(2) — closing half pass
        elif splitting == "monolithic":
            # Compose all predictors into a single pass each (no chain).
            # All predictors operate on the SAME input (mask token), outputs
            # are averaged. This mirrors the original PI-JEPA paper's
            # broken behavior — useful as the "no operator splitting" ablation.
            outputs = []
            for predictor in self.predictors:
                outputs.append(predictor.forward_chained(z_t, z_context))
            z_t = torch.stack(outputs, dim=0).mean(dim=0)
            stage_outputs.append(z_t)
        else:
            # Default Lie-Trotter (1st order). Falls back here for any K.
            for k, predictor in enumerate(self.predictors):
                z_t = predictor.forward_chained(z_t, z_context)
                stage_outputs.append(z_t)

        z_pred = z_t  # final stage output

        z_target = torch.gather(
            z_target_full,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, D),
        )

        # Match the BYOL/I-JEPA stability convention (LayerNorm + L2 normalize)
        z_pred = F.normalize(F.layer_norm(z_pred, (D,)), dim=-1)
        z_target = F.normalize(F.layer_norm(z_target, (D,)), dim=-1)

        return z_pred, z_target.detach(), stage_outputs, z_full

    def encode(self, x):
        return self.encoder(x)

    def encode_target(self, x):
        with torch.no_grad():
            return self.target_encoder(x)

    def predict_latent(self, z_full, context_indices, target_indices):
        B = z_full.shape[0]

        z = z_full.clone()

        mask_tokens = self.mask_token.expand(
            B, target_indices.shape[1], self.embed_dim
        )

        z = z.scatter(
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
            mask_tokens
        )

        for predictor in self.predictors:
            z_delta, _ = predictor(z, context_indices, target_indices)

            z_old = torch.gather(
                z,
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
            )

            z_new = z_old + 0.5 * z_delta

            z = z.scatter(
                1,
                target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim),
                z_new
            )

        z_out = torch.gather(
            z,
            1,
            target_indices.unsqueeze(-1).expand(-1, -1, self.embed_dim)
        )

        return F.normalize(z_out, dim=-1)

    def rollout(self, x_init, steps, decoder=None, noise_std=0.0):
        x = x_init
        outputs = []

        for _ in range(steps):
            z_full = self.encoder(x)

            B, N, D = z_full.shape
            idx = torch.arange(N, device=x.device).unsqueeze(0).repeat(B, 1)

            z_pred = self.predict_latent(z_full, idx, idx)

            z_recon = z_full.clone()
            z_recon = z_recon.scatter(
                1,
                idx.unsqueeze(-1).expand(-1, -1, D),
                z_pred
            )

            if noise_std > 0.0:
                z_recon = z_recon + noise_std * torch.randn_like(z_recon)

            if decoder is not None:
                x = decoder(z_recon)
            else:
                x = z_recon

            outputs.append(x)

        return torch.stack(outputs, dim=1)
