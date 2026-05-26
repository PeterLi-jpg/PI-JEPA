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
        B, C, H, W = x.shape
        
        # Get patch_size from encoder if available, otherwise use self.patch_size
        if hasattr(self.encoder, 'patch_size'):
            patch_size = self.encoder.patch_size
        else:
            patch_size = self.patch_size
        
        grid_size = H // patch_size
        num_patches = grid_size * grid_size
        
        # Clamp target_indices to valid range to prevent out-of-bounds errors
        # This handles cases where masker grid_size doesn't match encoder grid_size
        target_indices_clamped = target_indices.clamp(0, num_patches - 1)

        mask = torch.ones(B, num_patches, device=x.device)

        mask = mask.scatter(
            1,
            target_indices_clamped,
            torch.zeros_like(target_indices_clamped, dtype=mask.dtype)
        )

        mask = mask.view(B, grid_size, grid_size)
        mask = mask.repeat_interleave(patch_size, dim=1)
        mask = mask.repeat_interleave(patch_size, dim=2)
        mask = mask.unsqueeze(1)

        return x * mask
    
    def get_num_patches(self, image_size: int) -> int:
        """Get the number of patches for a given image size."""
        if hasattr(self.encoder, 'patch_size'):
            patch_size = self.encoder.patch_size
        else:
            patch_size = self.patch_size
        grid_size = image_size // patch_size
        return grid_size * grid_size

    def forward(self, x, context_indices, target_indices):
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

    # ------------------------------------------------------------------
    # Operator-split additions (additive: existing forward / mask_input
    # behavior unchanged for 2D callers).
    # ------------------------------------------------------------------

    def _mask_input_3d(self, x, target_indices):
        """5D (B, C, D, H, W) variant of mask_input.

        Uses the encoder's grid_size_dhw / patch_size_dhw when available
        (works for cubic OR rectangular FourierJEPAEncoder3D). Otherwise
        falls back to assuming cubic patches.
        """
        B, C, D, H, W = x.shape
        enc = self.encoder
        if hasattr(enc, "grid_size_dhw") and hasattr(enc, "patch_size_dhw"):
            gd, gh, gw = enc.grid_size_dhw
            pd, ph, pw = enc.patch_size_dhw
        else:
            patch = enc.patch_size if hasattr(enc, "patch_size") else self.patch_size
            gd = D // patch
            gh = H // patch
            gw = W // patch
            pd = ph = pw = patch
        num_patches = gd * gh * gw
        max_idx = target_indices.max().item() if target_indices.numel() > 0 else -1
        if max_idx >= num_patches:
            raise ValueError(
                f"target_indices max={max_idx} exceeds num_patches={num_patches} "
                f"(volume={D}x{H}x{W}, patch=({pd},{ph},{pw}), grid=({gd},{gh},{gw}))."
            )
        tic = target_indices.clamp(min=0)
        mask = torch.ones(B, num_patches, device=x.device)
        mask = mask.scatter(1, tic, torch.zeros_like(tic, dtype=mask.dtype))
        mask = mask.view(B, gd, gh, gw)
        mask = mask.repeat_interleave(pd, dim=1)
        mask = mask.repeat_interleave(ph, dim=2)
        mask = mask.repeat_interleave(pw, dim=3)
        return x * mask.unsqueeze(1)

    def mask_input_nd(self, x, target_indices):
        """Dispatch to 2D or 3D mask_input based on x.dim().

        Lets callers that don't know which spatial dimensionality the
        encoder is operating in pick the right path automatically.
        """
        if x.dim() == 4:
            return self.mask_input(x, target_indices)
        if x.dim() == 5:
            return self._mask_input_3d(x, target_indices)
        raise ValueError(f"mask_input_nd expects 4D or 5D; got {tuple(x.shape)}")

    def forward_operator_split(
        self, x, context_indices, target_indices, splitting: str = "lie_trotter"
    ):
        """True operator-splitting forward through K predictors.

        splitting:
          - "lie_trotter" (default, 1st order): each predictor runs in turn,
            ẑ^(k) = g_{φ_k}(ẑ^(k-1), z_context).
          - "strang" (2nd order, K=2 only):
            ẑ^(0.5) = g_{φ_1}^{1/2}(ẑ^(0), z_ctx)
            ẑ^(1.5) = g_{φ_2}(ẑ^(0.5), z_ctx)
            ẑ^(2)   = g_{φ_1}^{1/2}(ẑ^(1.5), z_ctx)
          - "monolithic": all predictors run in parallel on mask_tokens
            and averaged (ablation: "no operator splitting").

        Returns (z_pred, z_target_ema, stage_outputs_list, z_full).
        z_target_ema has stop_gradient applied. stage_outputs_list has
        length K (or K+1 for Strang) — one entry per predictor application.

        This method is additive: the legacy `forward(...)` above keeps the
        original semantics for 2D callers that use it.
        """
        B = x.shape[0]
        D = self.embed_dim

        # Encode targets with the EMA target encoder (no_grad)
        with torch.no_grad():
            z_target_full = self.target_encoder(x)

        # Mask input and encode with the context encoder
        x_masked = self.mask_input_nd(x, target_indices.clamp(min=0))
        z_full = self.encoder(x_masked)

        # Gather context once; it stays fixed across the chain.
        z_context = torch.gather(
            z_full, 1,
            context_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, D),
        )

        N_t = target_indices.shape[1]
        z_t = self.mask_token.expand(B, N_t, D).contiguous()

        stage_outputs = []
        K = len(self.predictors)
        if splitting == "strang" and K == 2:
            z_t = self.predictors[0].forward_chained(z_t, z_context)
            stage_outputs.append(z_t)
            z_t = self.predictors[1].forward_chained(z_t, z_context)
            stage_outputs.append(z_t)
            z_t = self.predictors[0].forward_chained(z_t, z_context)
            stage_outputs.append(z_t)
        elif splitting == "monolithic":
            outs = []
            for predictor in self.predictors:
                outs.append(predictor.forward_chained(z_t, z_context))
            z_t = torch.stack(outs, dim=0).mean(dim=0)
            stage_outputs.append(z_t)
        else:
            # Default: Lie-Trotter chain, 1st-order operator splitting.
            for predictor in self.predictors:
                z_t = predictor.forward_chained(z_t, z_context)
                stage_outputs.append(z_t)

        z_pred = z_t
        z_target = torch.gather(
            z_target_full, 1,
            target_indices.clamp(min=0).unsqueeze(-1).expand(-1, -1, D),
        )
        z_pred = F.normalize(F.layer_norm(z_pred, (D,)), dim=-1)
        z_target = F.normalize(F.layer_norm(z_target, (D,)), dim=-1)
        return z_pred, z_target.detach(), stage_outputs, z_full
