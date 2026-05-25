import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, q_input, kv_input):
        B, Nt, D = q_input.shape
        Nc = kv_input.shape[1]

        q = self.q(q_input).reshape(B, Nt, self.heads, self.head_dim).transpose(1, 2)
        kv = self.kv(kv_input).reshape(B, Nc, 2, self.heads, self.head_dim)

        k, v = kv.unbind(dim=2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nt, D)
        x = self.proj(x)
        x = self.drop(x)

        return x


class ChannelMixingAttention(nn.Module):
    """Two-stage attention: spatial self-attention + cross-species attention."""
    
    def __init__(
        self,
        dim: int,
        n_species: int,
        heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dim = dim
        self.n_species = n_species
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        
        # Stage 1: Spatial self-attention (within each species)
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_qkv = nn.Linear(dim, dim * 3)
        self.spatial_proj = nn.Linear(dim, dim)
        self.spatial_drop = nn.Dropout(dropout)
        
        # Stage 2: Cross-species attention
        self.species_norm = nn.LayerNorm(dim)
        self.species_qkv = nn.Linear(dim, dim * 3)
        self.species_proj = nn.Linear(dim, dim)
        self.species_drop = nn.Dropout(dropout)
    
    def _spatial_self_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial self-attention within each species."""
        B, S, N, D = x.shape
        
        # Normalize
        x_norm = self.spatial_norm(x)
        
        # Reshape to process all species in parallel: (B*S, N, D)
        x_flat = x_norm.reshape(B * S, N, D)
        
        # Compute Q, K, V
        qkv = self.spatial_qkv(x_flat).reshape(B * S, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*S, heads, N, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.spatial_drop(attn)
        
        # Apply attention and project
        out = (attn @ v).transpose(1, 2).reshape(B * S, N, D)
        out = self.spatial_proj(out)
        out = self.spatial_drop(out)
        
        # Reshape back and add residual
        out = out.reshape(B, S, N, D)
        return x + out
    
    def _cross_species_attention(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cross-species attention to mix information across species."""
        B, S, N, D = x.shape
        
        # Normalize
        x_norm = self.species_norm(x)
        
        # Transpose to (B, N, S, D) so attention is across species dimension
        x_t = x_norm.transpose(1, 2)  # (B, N, S, D)
        x_flat = x_t.reshape(B * N, S, D)
        
        # Compute Q, K, V
        qkv = self.species_qkv(x_flat).reshape(B * N, S, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B*N, heads, S, head_dim)
        q, k, v = qkv.unbind(0)
        
        # Attention across species
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.species_drop(attn)
        
        # Apply attention and project
        out = (attn @ v).transpose(1, 2).reshape(B * N, S, D)
        out = self.species_proj(out)
        out = self.species_drop(out)
        
        # Reshape back: (B*N, S, D) -> (B, N, S, D) -> (B, S, N, D)
        out = out.reshape(B, N, S, D).transpose(1, 2)
        return x + out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply two-stage channel-mixing attention."""
        # Stage 1: Spatial self-attention (within each species)
        x = self._spatial_self_attention(x)
        
        # Stage 2: Cross-species attention
        x = self._cross_species_attention(x)
        
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio, dropout):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


class PredictorBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio, dropout):
        super().__init__()

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

        self.cross_attn = CrossAttention(dim, heads, dropout)

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, z_target, z_context):
        z = z_target + self.cross_attn(
            self.norm_q(z_target),
            self.norm_kv(z_context)
        )

        z = z + self.mlp(self.norm_mlp(z))

        return z


class Stage(nn.Module):
    def __init__(self, dim, depth, heads, mlp_ratio, dropout):
        super().__init__()

        self.blocks = nn.ModuleList([
            PredictorBlock(dim, heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

    def forward(self, z_target, z_context):
        for blk in self.blocks:
            z_target = blk(z_target, z_context)
        return z_target


class Predictor(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]
        pred_cfg = model_cfg["predictor"]

        dim = model_cfg["latent_dim"]

        self.stages = nn.ModuleList([
            Stage(
                dim=dim,
                depth=stage_cfg["depth"],
                heads=stage_cfg["heads"],
                mlp_ratio=stage_cfg.get("mlp_ratio", 4.0),
                dropout=stage_cfg.get("dropout", 0.1)
            )
            for stage_cfg in pred_cfg["stages"]
        ])

        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, z_full, context_idx, target_idx):
        """Original forward used by 2D path. Gathers context+target from z_full,
        OVERWRITES target with mask_token, then refines via internal stages.

        Known limitation: when multiple Predictor instances are chained externally,
        each call to this forward discards the previous predictor's output. Use
        forward_chained() below for true Lie-Trotter operator-splitting chains.
        """
        B, N, D = z_full.shape

        z_context = torch.gather(
            z_full,
            1,
            context_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        z_target = torch.gather(
            z_full,
            1,
            target_idx.unsqueeze(-1).expand(-1, -1, D)
        )

        z_target = self.mask_token.expand_as(z_target)

        stage_outputs = {}

        for i, stage in enumerate(self.stages):
            z_delta = stage(z_target, z_context)

            z_target = z_target + z_delta

            stage_outputs[f"stage_{i}"] = z_target

        return z_target, stage_outputs

    def forward_chained(self, z_target_in, z_context):
        """True chained forward for operator-splitting.

        Refines an existing target representation `z_target_in` conditioned on
        the context `z_context`, without overwriting the input. Returns the
        refined (B, N_t, D) tensor.

        Use this for the Lie-Trotter / Strang chain across multiple predictors:
            z_t = mask_token.expand(...)
            for k, pred in enumerate(predictors):
                z_t = pred.forward_chained(z_t, z_context)
                stage_outputs.append(z_t)
        """
        z_target = z_target_in
        for stage in self.stages:
            z_delta = stage(z_target, z_context)
            z_target = z_target + z_delta
        return z_target

    def init_mask_tokens(self, B: int, N_t: int, device, dtype=None) -> torch.Tensor:
        """Initialize a (B, N_t, D) tensor from this predictor's mask_token.

        Used to seed the chain at predictor k=0.
        """
        z = self.mask_token.expand(B, N_t, -1).clone()
        if dtype is not None:
            z = z.to(dtype=dtype)
        return z.to(device)

    def rollout(self, z0, steps):
        traj = []
        z = z0

        B, N, D = z.shape
        idx = torch.arange(N, device=z.device).unsqueeze(0).repeat(B, 1)

        for _ in range(steps):
            z_pred, _ = self.forward(z, idx, idx)

            z = z.clone()
            z.scatter_(
                1,
                idx.unsqueeze(-1).expand(-1, -1, D),
                z_pred
            )

            traj.append(z)

        return torch.stack(traj, dim=1)


class MultiStepPredictor(nn.Module):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor

    def forward(self, z0, context_idx, target_idx, steps):
        z = z0
        outputs = []

        for _ in range(steps):
            z_pred, stage_out = self.predictor(z, context_idx, target_idx)

            z = z.clone()
            z.scatter_(
                1,
                target_idx.unsqueeze(-1).expand_as(z_pred),
                z_pred
            )

            outputs.append({
                "z": z,
                "stages": stage_out
            })

        return outputs


class MultiSpeciesPredictor(nn.Module):
    """Predictor for K=3 reactive transport with channel mixing."""
    
    def __init__(self, config: dict):
        super().__init__()
        
        model_cfg = config["model"]
        pred_cfg = model_cfg.get("predictor", {})
        
        self.dim = model_cfg["latent_dim"]
        self.num_predictors = model_cfg.get("num_predictors", 3)  # K=3 for reactive transport
        self.n_species = pred_cfg.get("n_species", 1)
        self.use_channel_mixing = pred_cfg.get("channel_mixing", True)
        
        heads = pred_cfg.get("heads", 8)
        dropout = pred_cfg.get("dropout", 0.1)
        
        # Create K predictors (one for each sub-operator)
        # Each predictor has its own set of stages
        stages_config = pred_cfg.get("stages", [{"depth": 2, "heads": 8}])
        
        self.predictors = nn.ModuleList()
        for k in range(self.num_predictors):
            # Each predictor gets its own Stage modules
            predictor_stages = nn.ModuleList([
                Stage(
                    dim=self.dim,
                    depth=stage_cfg.get("depth", 2),
                    heads=stage_cfg.get("heads", heads),
                    mlp_ratio=stage_cfg.get("mlp_ratio", 4.0),
                    dropout=stage_cfg.get("dropout", dropout)
                )
                for stage_cfg in stages_config
            ])
            self.predictors.append(predictor_stages)
        
        # Channel-mixing attention for multi-species processing
        if self.use_channel_mixing and self.n_species > 1:
            self.channel_mixing = nn.ModuleList([
                ChannelMixingAttention(
                    dim=self.dim,
                    n_species=self.n_species,
                    heads=heads,
                    dropout=dropout
                )
                for _ in range(self.num_predictors)
            ])
        else:
            self.channel_mixing = None
        
        # Mask token for target initialization (shared across predictors)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 1, self.dim))
        nn.init.normal_(self.mask_token, std=0.02)
        
        # Predictor names for interpretability
        self.predictor_names = ["pressure", "transport", "reaction"][:self.num_predictors]
    
    def _apply_predictor(
        self,
        predictor_idx: int,
        z_target: torch.Tensor,
        z_context: torch.Tensor
    ) -> torch.Tensor:
        """Apply a single sub-operator predictor."""
        predictor_stages = self.predictors[predictor_idx]
        
        # Check if input is multi-species
        is_multi_species = z_target.dim() == 4
        
        if is_multi_species:
            B, S, N_t, D = z_target.shape
            _, _, N_c, _ = z_context.shape
            
            # Process each species through the predictor stages
            # Reshape to (B*S, N, D) for stage processing
            z_target_flat = z_target.reshape(B * S, N_t, D)
            z_context_flat = z_context.reshape(B * S, N_c, D)
            
            for stage in predictor_stages:
                z_target_flat = stage(z_target_flat, z_context_flat)
            
            # Reshape back to (B, S, N, D)
            z_out = z_target_flat.reshape(B, S, N_t, D)
            
            # Apply channel-mixing attention if enabled
            if self.channel_mixing is not None:
                z_out = self.channel_mixing[predictor_idx](z_out)
        else:
            # Single-species case: (B, N, D)
            z_out = z_target
            for stage in predictor_stages:
                z_out = stage(z_out, z_context)
        
        return z_out
    
    def forward(
        self,
        z_full: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Forward pass through all K predictors in sequence (Lie-Trotter splitting)."""
        is_multi_species = z_full.dim() == 4
        
        if is_multi_species:
            B, S, N, D = z_full.shape
            
            # Gather context and target latents
            # context_idx: (B, N_c) -> expand to (B, S, N_c, D)
            context_idx_exp = context_idx.unsqueeze(1).unsqueeze(-1).expand(-1, S, -1, D)
            target_idx_exp = target_idx.unsqueeze(1).unsqueeze(-1).expand(-1, S, -1, D)
            
            z_context = torch.gather(z_full, 2, context_idx_exp)  # (B, S, N_c, D)
            z_target = torch.gather(z_full, 2, target_idx_exp)    # (B, S, N_t, D)
            
            # Initialize target with mask token
            N_t = target_idx.shape[1]
            z_target = self.mask_token.expand(B, S, N_t, D)
        else:
            B, N, D = z_full.shape
            
            # Gather context and target latents
            z_context = torch.gather(
                z_full, 1,
                context_idx.unsqueeze(-1).expand(-1, -1, D)
            )
            z_target = torch.gather(
                z_full, 1,
                target_idx.unsqueeze(-1).expand(-1, -1, D)
            )
            
            # Initialize target with mask token (squeeze species dim)
            N_t = target_idx.shape[1]
            z_target = self.mask_token.squeeze(1).expand(B, N_t, D)
        
        stage_outputs = {}
        
        # Apply predictors in sequence (Lie-Trotter operator splitting)
        for k in range(self.num_predictors):
            z_delta = self._apply_predictor(k, z_target, z_context)
            z_target = z_target + z_delta
            
            predictor_name = self.predictor_names[k] if k < len(self.predictor_names) else f"predictor_{k}"
            stage_outputs[predictor_name] = z_target.clone()
        
        return z_target, stage_outputs
    
    def forward_single_predictor(
        self,
        predictor_idx: int,
        z_full: torch.Tensor,
        context_idx: torch.Tensor,
        target_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply only a single sub-operator predictor."""
        is_multi_species = z_full.dim() == 4
        
        if is_multi_species:
            B, S, N, D = z_full.shape
            
            context_idx_exp = context_idx.unsqueeze(1).unsqueeze(-1).expand(-1, S, -1, D)
            target_idx_exp = target_idx.unsqueeze(1).unsqueeze(-1).expand(-1, S, -1, D)
            
            z_context = torch.gather(z_full, 2, context_idx_exp)
            
            N_t = target_idx.shape[1]
            z_target = self.mask_token.expand(B, S, N_t, D)
        else:
            B, N, D = z_full.shape
            
            z_context = torch.gather(
                z_full, 1,
                context_idx.unsqueeze(-1).expand(-1, -1, D)
            )
            
            N_t = target_idx.shape[1]
            z_target = self.mask_token.squeeze(1).expand(B, N_t, D)
        
        z_pred = self._apply_predictor(predictor_idx, z_target, z_context)
        
        return z_pred, z_context
    
    def rollout(
        self,
        z0: torch.Tensor,
        steps: int,
        noise_std: float = 0.0
    ) -> torch.Tensor:
        """Perform multi-step rollout with all K predictors."""
        traj = []
        z = z0
        
        is_multi_species = z.dim() == 4
        
        if is_multi_species:
            B, S, N, D = z.shape
            idx = torch.arange(N, device=z.device).unsqueeze(0).expand(B, -1)
        else:
            B, N, D = z.shape
            idx = torch.arange(N, device=z.device).unsqueeze(0).expand(B, -1)
        
        for step in range(steps):
            z_pred, _ = self.forward(z, idx, idx)
            
            # Inject noise for rollout stability (if enabled)
            if noise_std > 0:
                noise = torch.randn_like(z_pred) * noise_std
                z_pred = z_pred + noise
            
            # Update latent
            z = z.clone()
            if is_multi_species:
                idx_exp = idx.unsqueeze(1).unsqueeze(-1).expand(-1, S, -1, D)
                z.scatter_(2, idx_exp, z_pred)
            else:
                z.scatter_(1, idx.unsqueeze(-1).expand(-1, -1, D), z_pred)
            
            traj.append(z)
        
        return torch.stack(traj, dim=1)
