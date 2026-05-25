from .darcy import (
    physics_loss_pressure,
    physics_loss_saturation,
    grad_x,
    grad_y,
    divergence,
    mobility,
    BrooksCoreyModel,
    TwoPhaseDarcyPhysics,
)
from .reactive_transport import ReactiveTransportPhysics
from .spectral_residual import SpectralResidualModule
from .latent_flux import LatentFluxModule
from .tpfa import TPFALoss

__all__ = [
    "physics_loss_pressure",
    "physics_loss_saturation",
    "grad_x",
    "grad_y",
    "divergence",
    "mobility",
    "BrooksCoreyModel",
    "TwoPhaseDarcyPhysics",
    "ReactiveTransportPhysics",
    "SpectralResidualModule",
    "LatentFluxModule",
    "TPFALoss",
]
