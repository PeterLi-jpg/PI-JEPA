import yaml
import os
from typing import Any, Dict, List, Optional


class Config:
    def __init__(self, cfg_dict):
        self._cfg = cfg_dict

    def __getitem__(self, key):
        return self._cfg[key]

    def __contains__(self, key):
        return key in self._cfg

    def __setitem__(self, key, value):
        self._cfg[key] = value

    def get(self, key, default=None):
        return self._cfg.get(key, default)

    def as_dict(self):
        return self._cfg


def load_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg = _apply_defaults(cfg)
    _validate(cfg)

    return Config(cfg)


def _apply_defaults(cfg):
    cfg.setdefault("experiment", {})
    cfg.setdefault("training", {})
    cfg.setdefault("model", {})
    cfg.setdefault("predictor", {})
    cfg.setdefault("decoder", {})
    cfg.setdefault("data", {})
    cfg.setdefault("ema", {})
    cfg.setdefault("masking", {})
    cfg.setdefault("pretraining", {})
    cfg.setdefault("finetuning", {})
    cfg.setdefault("evaluation", {})

    cfg["model"].setdefault("encoder", {})

    cfg["experiment"].setdefault("device", "cuda")
    cfg["experiment"].setdefault("precision", "fp32")

    cfg["training"].setdefault("epochs", 100)
    cfg["training"].setdefault("batch_size", 8)
    cfg["training"].setdefault("optim", {})
    cfg["training"]["optim"].setdefault("lr", 1e-4)
    cfg["training"]["optim"].setdefault("weight_decay", 1e-4)
    cfg["training"]["optim"].setdefault("betas", [0.9, 0.999])
    cfg["training"].setdefault("gradient", {})
    cfg["training"]["gradient"].setdefault("clip_norm", None)

    cfg["model"].setdefault("num_predictors", 1)

    cfg["ema"].setdefault("schedule", {})
    cfg["ema"]["schedule"].setdefault("tau_start", 0.996)
    cfg["ema"]["schedule"].setdefault("tau_end", 0.999)

    cfg["masking"].setdefault("context_ratio", 0.5)

    # Apply pretraining defaults
    cfg = _apply_pretraining_defaults(cfg)
    
    # Apply finetuning defaults
    cfg = _apply_finetuning_defaults(cfg)
    
    # Apply evaluation defaults
    cfg = _apply_evaluation_defaults(cfg)

    return cfg


def _apply_pretraining_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values for pretraining configuration."""
    pretraining = cfg.setdefault("pretraining", {})
    
    pretraining.setdefault("enabled", True)
    pretraining.setdefault("epochs", 500)
    pretraining.setdefault("batch_size", 64)
    pretraining.setdefault("n_unlabeled", 1000)
    
    # JEPA objective defaults
    jepa = pretraining.setdefault("jepa", {})
    jepa.setdefault("normalize_embeddings", True)
    jepa.setdefault("stop_gradient_target", True)
    
    # Masking defaults
    masking = pretraining.setdefault("masking", {})
    masking.setdefault("context_ratio", 0.65)
    masking.setdefault("min_block_size", 2)
    masking.setdefault("max_block_size", 4)
    
    # Physics defaults
    physics = pretraining.setdefault("physics", {})
    physics.setdefault("enabled", True)
    physics.setdefault("weight", 0.1)
    physics.setdefault("ramp_steps", 200)
    
    # New physics mode defaults (null = legacy mode for backward compatibility)
    physics.setdefault("mode", None)
    
    # Spectral residual defaults
    spectral = physics.setdefault("spectral", {})
    spectral.setdefault("resolution", 64)
    spectral.setdefault("cutoff_ratio", 0.667)
    spectral.setdefault("dx", 1.0)
    spectral.setdefault("dy", 1.0)
    
    # Latent flux defaults
    latent_flux = physics.setdefault("latent_flux", {})
    latent_flux.setdefault("grid_size", 8)
    latent_flux.setdefault("n_flux_heads", 4)
    
    # TPFA defaults
    tpfa = physics.setdefault("tpfa", {})
    tpfa.setdefault("enabled", False)
    tpfa.setdefault("dx", 1.0)
    tpfa.setdefault("dy", 1.0)
    
    # Curriculum defaults
    curriculum = physics.setdefault("curriculum", {})
    curriculum.setdefault("warmup_steps", 1000)
    curriculum.setdefault("pressure_ramp_steps", 500)
    curriculum.setdefault("saturation_ramp_steps", 500)
    curriculum.setdefault("ramp_type", "cosine")
    
    # Adaptive collocation defaults
    adaptive_collocation = physics.setdefault("adaptive_collocation", {})
    adaptive_collocation.setdefault("enabled", False)
    adaptive_collocation.setdefault("n_points", 1024)
    adaptive_collocation.setdefault("min_density", 0.1)
    adaptive_collocation.setdefault("update_interval", 50)
    
    # Learned weights defaults
    learned_weights = physics.setdefault("learned_weights", {})
    learned_weights.setdefault("enabled", False)
    learned_weights.setdefault("lr", 1e-3)
    
    # VICReg defaults
    vicreg = pretraining.setdefault("vicreg", {})
    vicreg.setdefault("variance_weight", 0.05)
    vicreg.setdefault("covariance_weight", 0.01)
    
    # Optimizer defaults
    optim = pretraining.setdefault("optim", {})
    optim.setdefault("lr", 1.5e-4)
    optim.setdefault("weight_decay", 5e-2)
    optim.setdefault("betas", [0.9, 0.95])
    
    # EMA defaults
    ema = pretraining.setdefault("ema", {})
    ema.setdefault("tau_start", 0.99)
    ema.setdefault("tau_end", 0.999)
    ema.setdefault("warmup_fraction", 0.1)
    
    # Checkpoint defaults
    checkpoint = pretraining.setdefault("checkpoint", {})
    checkpoint.setdefault("save_interval", 50)
    checkpoint.setdefault("save_best", True)
    
    return cfg


def _apply_finetuning_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values for finetuning configuration."""
    finetuning = cfg.setdefault("finetuning", {})
    
    finetuning.setdefault("enabled", True)
    finetuning.setdefault("epochs", 100)
    finetuning.setdefault("batch_size", 32)
    finetuning.setdefault("n_labeled", 100)
    finetuning.setdefault("n_labeled_sweep", [10, 25, 50, 100, 250, 500])
    finetuning.setdefault("freeze_encoder", True)
    
    # Prediction head defaults
    prediction_head = finetuning.setdefault("prediction_head", {})
    prediction_head.setdefault("hidden_dim", 512)
    prediction_head.setdefault("output_channels", 1)
    
    # Optimizer defaults
    optim = finetuning.setdefault("optim", {})
    optim.setdefault("lr", 1e-3)
    optim.setdefault("weight_decay", 1e-4)
    
    # Full finetune defaults
    full_finetune = finetuning.setdefault("full_finetune", {})
    full_finetune.setdefault("enabled", False)
    full_finetune.setdefault("encoder_lr_multiplier", 0.1)
    
    # Checkpoint defaults
    checkpoint = finetuning.setdefault("checkpoint", {})
    checkpoint.setdefault("save_interval", 20)
    checkpoint.setdefault("save_best", True)
    
    return cfg


def _apply_evaluation_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Apply default values for evaluation configuration."""
    evaluation = cfg.setdefault("evaluation", {})
    
    # Data efficiency defaults
    data_efficiency = evaluation.setdefault("data_efficiency", {})
    data_efficiency.setdefault("enabled", True)
    data_efficiency.setdefault("n_labeled_sweep", [10, 25, 50, 100, 250, 500])
    data_efficiency.setdefault("baselines", ["fno", "geo_fno", "deeponet"])
    data_efficiency.setdefault("test_set_size", 200)
    data_efficiency.setdefault("output_file", "benchmark_comparison.json")
    
    return cfg


def _validate(cfg):
    required = [
        ("model.encoder", "embed_dim"),
        ("model.encoder", "patch_size"),
        ("model.encoder", "image_size"),
        ("decoder", "image_size"),
        ("decoder", "patch_size"),
        ("data", "num_patches")
    ]

    for section, key in required:
        parts = section.split(".")
        ref = cfg

        for p in parts:
            if p not in ref:
                raise ValueError(f"Missing required config section: {section}")
            ref = ref[p]

        if key not in ref:
            raise ValueError(f"Missing required config: {section}.{key}")

    # Validate pretraining configuration
    _validate_pretraining(cfg)
    
    # Validate finetuning configuration
    _validate_finetuning(cfg)


def _validate_pretraining(cfg: Dict[str, Any]) -> None:
    """Validate pretraining configuration fields."""
    pretraining = cfg.get("pretraining", {})
    
    if not pretraining.get("enabled", True):
        return  # Skip validation if pretraining is disabled
    
    # Validate epochs
    epochs = pretraining.get("epochs", 500)
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"pretraining.epochs must be a positive integer, got {epochs}")
    
    # Validate batch_size
    batch_size = pretraining.get("batch_size", 64)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"pretraining.batch_size must be a positive integer, got {batch_size}")
    
    # Validate n_unlabeled
    n_unlabeled = pretraining.get("n_unlabeled", 1000)
    if not isinstance(n_unlabeled, int) or n_unlabeled <= 0:
        raise ValueError(f"pretraining.n_unlabeled must be a positive integer, got {n_unlabeled}")
    
    # Validate masking settings
    masking = pretraining.get("masking", {})
    context_ratio = masking.get("context_ratio", 0.65)
    if not isinstance(context_ratio, (int, float)) or not 0 < context_ratio < 1:
        raise ValueError(f"pretraining.masking.context_ratio must be between 0 and 1, got {context_ratio}")
    
    min_block = masking.get("min_block_size", 2)
    max_block = masking.get("max_block_size", 4)
    if min_block > max_block:
        raise ValueError(f"pretraining.masking.min_block_size ({min_block}) must be <= max_block_size ({max_block})")
    
    # Validate physics settings
    physics = pretraining.get("physics", {})
    weight = physics.get("weight", 0.1)
    if not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError(f"pretraining.physics.weight must be non-negative, got {weight}")
    
    ramp_steps = physics.get("ramp_steps", 200)
    if not isinstance(ramp_steps, int) or ramp_steps < 0:
        raise ValueError(f"pretraining.physics.ramp_steps must be a non-negative integer, got {ramp_steps}")
    
    # Validate physics mode
    _validate_physics_mode(physics)
    
    # Validate EMA settings
    ema = pretraining.get("ema", {})
    tau_start = ema.get("tau_start", 0.99)
    tau_end = ema.get("tau_end", 0.999)
    if not 0 <= tau_start <= 1 or not 0 <= tau_end <= 1:
        raise ValueError(f"pretraining.ema.tau_start and tau_end must be between 0 and 1")
    if tau_start > tau_end:
        raise ValueError(f"pretraining.ema.tau_start ({tau_start}) should be <= tau_end ({tau_end})")


def _validate_finetuning(cfg: Dict[str, Any]) -> None:
    """Validate finetuning configuration fields."""
    finetuning = cfg.get("finetuning", {})
    
    if not finetuning.get("enabled", True):
        return  # Skip validation if finetuning is disabled
    
    # Validate epochs
    epochs = finetuning.get("epochs", 100)
    if not isinstance(epochs, int) or epochs <= 0:
        raise ValueError(f"finetuning.epochs must be a positive integer, got {epochs}")
    
    # Validate batch_size
    batch_size = finetuning.get("batch_size", 32)
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"finetuning.batch_size must be a positive integer, got {batch_size}")
    
    # Validate n_labeled
    n_labeled = finetuning.get("n_labeled", 100)
    if not isinstance(n_labeled, int) or n_labeled <= 0:
        raise ValueError(f"finetuning.n_labeled must be a positive integer, got {n_labeled}")
    
    # Validate n_labeled_sweep
    n_labeled_sweep = finetuning.get("n_labeled_sweep", [10, 25, 50, 100, 250, 500])
    if not isinstance(n_labeled_sweep, list) or len(n_labeled_sweep) == 0:
        raise ValueError(f"finetuning.n_labeled_sweep must be a non-empty list")
    for n in n_labeled_sweep:
        if not isinstance(n, int) or n <= 0:
            raise ValueError(f"finetuning.n_labeled_sweep values must be positive integers, got {n}")
    
    # Validate prediction head settings
    prediction_head = finetuning.get("prediction_head", {})
    hidden_dim = prediction_head.get("hidden_dim", 512)
    if not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError(f"finetuning.prediction_head.hidden_dim must be a positive integer, got {hidden_dim}")
    
    output_channels = prediction_head.get("output_channels", 1)
    if not isinstance(output_channels, int) or output_channels <= 0:
        raise ValueError(f"finetuning.prediction_head.output_channels must be a positive integer, got {output_channels}")
    
    # Validate full_finetune settings
    full_finetune = finetuning.get("full_finetune", {})
    encoder_lr_multiplier = full_finetune.get("encoder_lr_multiplier", 0.1)
    if not isinstance(encoder_lr_multiplier, (int, float)) or encoder_lr_multiplier <= 0:
        raise ValueError(f"finetuning.full_finetune.encoder_lr_multiplier must be positive, got {encoder_lr_multiplier}")


def _validate_physics_mode(physics: Dict[str, Any]) -> None:
    """Validate the new physics mode configuration fields.
    
    Ensures backward compatibility: when mode is None (legacy), no new fields
    are required. When a mode is specified, validates the relevant sub-config.
    """
    VALID_MODES = [None, "spectral", "tpfa", "latent_flux", "combined"]
    VALID_RAMP_TYPES = ["linear", "cosine", "step"]
    
    mode = physics.get("mode", None)
    if mode not in VALID_MODES:
        raise ValueError(
            f"pretraining.physics.mode must be one of {VALID_MODES}, got '{mode}'"
        )
    
    # If mode is None (legacy), skip validation of new fields
    if mode is None:
        return
    
    # Validate spectral config (used by "spectral" and "combined" modes)
    if mode in ("spectral", "combined"):
        spectral = physics.get("spectral", {})
        resolution = spectral.get("resolution", 64)
        if not isinstance(resolution, int) or resolution <= 0:
            raise ValueError(
                f"pretraining.physics.spectral.resolution must be a positive integer, got {resolution}"
            )
        cutoff_ratio = spectral.get("cutoff_ratio", 0.667)
        if not isinstance(cutoff_ratio, (int, float)) or not 0 < cutoff_ratio <= 1:
            raise ValueError(
                f"pretraining.physics.spectral.cutoff_ratio must be in (0, 1], got {cutoff_ratio}"
            )
        dx = spectral.get("dx", 1.0)
        dy = spectral.get("dy", 1.0)
        if not isinstance(dx, (int, float)) or dx <= 0:
            raise ValueError(f"pretraining.physics.spectral.dx must be positive, got {dx}")
        if not isinstance(dy, (int, float)) or dy <= 0:
            raise ValueError(f"pretraining.physics.spectral.dy must be positive, got {dy}")
    
    # Validate latent_flux config (used by "latent_flux" and "combined" modes)
    if mode in ("latent_flux", "combined"):
        latent_flux = physics.get("latent_flux", {})
        grid_size = latent_flux.get("grid_size", 8)
        if not isinstance(grid_size, int) or grid_size <= 0:
            raise ValueError(
                f"pretraining.physics.latent_flux.grid_size must be a positive integer, got {grid_size}"
            )
        n_flux_heads = latent_flux.get("n_flux_heads", 4)
        if not isinstance(n_flux_heads, int) or n_flux_heads <= 0:
            raise ValueError(
                f"pretraining.physics.latent_flux.n_flux_heads must be a positive integer, got {n_flux_heads}"
            )
    
    # Validate TPFA config (used by "tpfa" mode)
    if mode == "tpfa":
        tpfa = physics.get("tpfa", {})
        dx = tpfa.get("dx", 1.0)
        dy = tpfa.get("dy", 1.0)
        if not isinstance(dx, (int, float)) or dx <= 0:
            raise ValueError(f"pretraining.physics.tpfa.dx must be positive, got {dx}")
        if not isinstance(dy, (int, float)) or dy <= 0:
            raise ValueError(f"pretraining.physics.tpfa.dy must be positive, got {dy}")
    
    # Validate curriculum config (always validated when mode is set)
    curriculum = physics.get("curriculum", {})
    warmup_steps = curriculum.get("warmup_steps", 1000)
    if not isinstance(warmup_steps, int) or warmup_steps < 0:
        raise ValueError(
            f"pretraining.physics.curriculum.warmup_steps must be a non-negative integer, got {warmup_steps}"
        )
    pressure_ramp_steps = curriculum.get("pressure_ramp_steps", 500)
    if not isinstance(pressure_ramp_steps, int) or pressure_ramp_steps < 0:
        raise ValueError(
            f"pretraining.physics.curriculum.pressure_ramp_steps must be a non-negative integer, got {pressure_ramp_steps}"
        )
    saturation_ramp_steps = curriculum.get("saturation_ramp_steps", 500)
    if not isinstance(saturation_ramp_steps, int) or saturation_ramp_steps < 0:
        raise ValueError(
            f"pretraining.physics.curriculum.saturation_ramp_steps must be a non-negative integer, got {saturation_ramp_steps}"
        )
    ramp_type = curriculum.get("ramp_type", "cosine")
    if ramp_type not in VALID_RAMP_TYPES:
        raise ValueError(
            f"pretraining.physics.curriculum.ramp_type must be one of {VALID_RAMP_TYPES}, got '{ramp_type}'"
        )
    
    # Validate adaptive collocation config
    adaptive = physics.get("adaptive_collocation", {})
    if adaptive.get("enabled", False):
        n_points = adaptive.get("n_points", 1024)
        if not isinstance(n_points, int) or n_points <= 0:
            raise ValueError(
                f"pretraining.physics.adaptive_collocation.n_points must be a positive integer, got {n_points}"
            )
        min_density = adaptive.get("min_density", 0.1)
        if not isinstance(min_density, (int, float)) or not 0 < min_density <= 1:
            raise ValueError(
                f"pretraining.physics.adaptive_collocation.min_density must be in (0, 1], got {min_density}"
            )
        update_interval = adaptive.get("update_interval", 50)
        if not isinstance(update_interval, int) or update_interval <= 0:
            raise ValueError(
                f"pretraining.physics.adaptive_collocation.update_interval must be a positive integer, got {update_interval}"
            )
    
    # Validate learned weights config
    learned_weights = physics.get("learned_weights", {})
    if learned_weights.get("enabled", False):
        lr = learned_weights.get("lr", 1e-3)
        if not isinstance(lr, (int, float)) or lr <= 0:
            raise ValueError(
                f"pretraining.physics.learned_weights.lr must be positive, got {lr}"
            )
