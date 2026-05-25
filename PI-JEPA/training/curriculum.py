"""Physics curriculum for staged introduction of physics residual terms.

Implements a three-phase training curriculum:
  Phase 1: JEPA + VICReg only (warm-up) — all physics weights are 0
  Phase 2: + pressure residual (ramp in from 0 to 1)
  Phase 3: + saturation transport (ramp in from 0 to 1)
"""

import math
from typing import Dict


class PhysicsCurriculum:
    """Staged introduction of physics residual terms.

    Phase 1: JEPA + VICReg only (warm-up)
    Phase 2: + pressure residual (ramp in)
    Phase 3: + saturation transport (ramp in)
    """

    def __init__(
        self,
        warmup_steps: int = 1000,
        pressure_ramp_steps: int = 500,
        saturation_ramp_steps: int = 500,
        ramp_type: str = "cosine",  # 'linear', 'cosine', 'step'
    ):
        if warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {warmup_steps}")
        if pressure_ramp_steps < 0:
            raise ValueError(f"pressure_ramp_steps must be >= 0, got {pressure_ramp_steps}")
        if saturation_ramp_steps < 0:
            raise ValueError(f"saturation_ramp_steps must be >= 0, got {saturation_ramp_steps}")
        if ramp_type not in ("linear", "cosine", "step"):
            raise ValueError(
                f"ramp_type must be one of 'linear', 'cosine', 'step', got '{ramp_type}'"
            )

        self.warmup_steps = warmup_steps
        self.pressure_ramp_steps = pressure_ramp_steps
        self.saturation_ramp_steps = saturation_ramp_steps
        self.ramp_type = ramp_type

    def _compute_ramp(self, progress: float) -> float:
        """Compute ramp value from progress in [0, 1] using configured schedule.

        Args:
            progress: A value in [0, 1] representing how far through the ramp.

        Returns:
            Weight value in [0, 1].
        """
        # Clamp progress to [0, 1]
        progress = max(0.0, min(1.0, progress))

        if self.ramp_type == "linear":
            return progress
        elif self.ramp_type == "cosine":
            return 0.5 * (1.0 - math.cos(math.pi * progress))
        elif self.ramp_type == "step":
            return 1.0 if progress > 0.0 else 0.0
        else:
            raise ValueError(f"Unknown ramp_type: {self.ramp_type}")

    def get_weights(self, step: int) -> Dict[str, float]:
        """Return current weight for each physics term.

        Args:
            step: Current training step (must be >= 0).

        Returns:
            Dictionary with 'pressure' and 'saturation' weights, each in [0, 1].
        """
        if step < 0:
            raise ValueError(f"step must be >= 0, got {step}")

        # Phase 1: warmup — all physics weights are 0
        if step < self.warmup_steps:
            return {"pressure": 0.0, "saturation": 0.0}

        # Phase 2: pressure ramp
        pressure_start = self.warmup_steps
        if self.pressure_ramp_steps == 0:
            # Immediate full weight after warmup
            pressure_weight = 1.0
        else:
            pressure_progress = min(
                1.0, (step - pressure_start) / self.pressure_ramp_steps
            )
            pressure_weight = self._compute_ramp(pressure_progress)

        # Phase 3: saturation ramp starts after pressure ramp completes
        saturation_start = pressure_start + self.pressure_ramp_steps
        if step < saturation_start:
            saturation_weight = 0.0
        elif self.saturation_ramp_steps == 0:
            # Immediate full weight after pressure ramp
            saturation_weight = 1.0
        else:
            saturation_progress = min(
                1.0, (step - saturation_start) / self.saturation_ramp_steps
            )
            saturation_weight = self._compute_ramp(saturation_progress)

        return {"pressure": pressure_weight, "saturation": saturation_weight}
