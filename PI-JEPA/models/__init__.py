from .pi_jepa import PIJEPA
from .encoder import ViTEncoder, TargetEncoder, update_ema
from .decoder import Decoder, Decoder3D
from .predictor import Predictor, MultiStepPredictor, MultiSpeciesPredictor, ChannelMixingAttention
from .prediction_head import PredictionHead
from .fourier_encoder import (
    FourierJEPAEncoder,
    MultiScaleFourierEncoder,
    FourierJEPAEncoder3D,
    SpectralConv3d,
    FourierBlock3D,
)

__all__ = [
    "PIJEPA",
    "ViTEncoder",
    "TargetEncoder",
    "Decoder",
    "Decoder3D",
    "Predictor",
    "MultiStepPredictor",
    "MultiSpeciesPredictor",
    "ChannelMixingAttention",
    "PredictionHead",
    "update_ema",
    "FourierJEPAEncoder",
    "MultiScaleFourierEncoder",
    "FourierJEPAEncoder3D",
    "SpectralConv3d",
    "FourierBlock3D",
]


def build_encoder(config: dict, in_channels: int = 1):
    """
    Factory function to build encoder based on config.

    Recognized `model.encoder.type` values:
      - "vit"                 -> ViTEncoder (2D)
      - "fourier"             -> FourierJEPAEncoder (2D)
      - "multiscale_fourier"  -> MultiScaleFourierEncoder (2D)
      - "fourier_3d"          -> FourierJEPAEncoder3D (3D)

    Args:
        config: Configuration dictionary
        in_channels: Number of input channels

    Returns:
        Encoder module
    """
    enc_cfg = config.get("model", {}).get("encoder", {})
    encoder_type = enc_cfg.get("type", "vit").lower()

    if encoder_type == "vit":
        return ViTEncoder(config, in_channels=in_channels)
    elif encoder_type == "fourier":
        return FourierJEPAEncoder(config, in_channels=in_channels)
    elif encoder_type == "multiscale_fourier":
        return MultiScaleFourierEncoder(config, in_channels=in_channels)
    elif encoder_type in ("fourier_3d", "fourier3d"):
        return FourierJEPAEncoder3D(config, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
