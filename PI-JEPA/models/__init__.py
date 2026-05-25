from .pi_jepa import PIJEPA
from .encoder import ViTEncoder, TargetEncoder, update_ema
from .decoder import Decoder
from .predictor import Predictor, MultiStepPredictor, MultiSpeciesPredictor, ChannelMixingAttention
from .prediction_head import PredictionHead
from .fourier_encoder import FourierJEPAEncoder, MultiScaleFourierEncoder
from .fourier_encoder_3d import FourierJEPAEncoder3D
from .pvt_conditioner import PVTConditioner
from .brooks_corey_conditioner import BrooksCoreyConditioner
from .well_conditioner import WellControlConditioner

__all__ = [
    "PIJEPA",
    "ViTEncoder",
    "TargetEncoder",
    "Decoder",
    "Predictor",
    "MultiStepPredictor",
    "MultiSpeciesPredictor",
    "ChannelMixingAttention",
    "PredictionHead",
    "update_ema",
    "FourierJEPAEncoder",
    "MultiScaleFourierEncoder",
    "FourierJEPAEncoder3D",
    "PVTConditioner",
    "BrooksCoreyConditioner",
    "WellControlConditioner",
]


def build_encoder(config: dict, in_channels: int = 1):
    """
    Factory function to build encoder based on config.
    
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
    elif encoder_type == "fourier_3d":
        return FourierJEPAEncoder3D(config, in_channels=in_channels)
    else:
        raise ValueError(f"Unknown encoder type: {encoder_type}")
