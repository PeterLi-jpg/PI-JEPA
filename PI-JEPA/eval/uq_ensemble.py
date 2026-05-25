"""
Uncertainty Quantification via Deep Ensemble.

Implements UQ using an ensemble of independently pretrained PI-JEPA models.
Provides mean prediction and pointwise standard deviation maps.

Requirements: 18.1, 18.2, 18.3
"""

import warnings
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


class UQEnsemble:
    """Ensemble of independently pretrained PI-JEPA models for UQ.

    Provides mean prediction and pointwise uncertainty (std).
    Supports graceful degradation if fewer than 5 models are available.
    """

    def __init__(
        self,
        models: Optional[List[nn.Module]] = None,
        model_paths: Optional[List[str]] = None,
        config: Optional[dict] = None,
        device: str = 'cpu',
        n_ensemble: int = 5,
    ):
        """Initialize UQ ensemble.

        Args:
            models: List of pre-loaded model instances (preferred for testing).
            model_paths: List of paths to saved model checkpoints.
            config: Configuration dict for model construction (used with model_paths).
            device: Device to run inference on.
            n_ensemble: Expected number of ensemble members.
        """
        self.device = device
        self.n_ensemble = n_ensemble
        self.models: List[nn.Module] = []

        if models is not None:
            self.models = [m.to(device) for m in models]
        elif model_paths is not None:
            self._load_models(model_paths, config)

        if len(self.models) < n_ensemble:
            warnings.warn(
                f"UQEnsemble: Only {len(self.models)} models available "
                f"(expected {n_ensemble}). Proceeding with reduced ensemble."
            )

    def _load_models(self, model_paths: List[str], config: Optional[dict]) -> None:
        """Load models from checkpoint paths with graceful degradation."""
        for path in model_paths:
            try:
                checkpoint = torch.load(path, map_location=self.device)
                if isinstance(checkpoint, nn.Module):
                    self.models.append(checkpoint.to(self.device))
                elif isinstance(checkpoint, dict) and 'model' in checkpoint:
                    # Assume checkpoint dict with model state_dict
                    # Would need model factory here; skip for now
                    warnings.warn(f"Skipping {path}: state_dict loading requires model factory")
                else:
                    warnings.warn(f"Skipping {path}: unrecognized checkpoint format")
            except Exception as e:
                warnings.warn(f"Failed to load model from {path}: {e}")

    @property
    def num_models(self) -> int:
        """Number of successfully loaded models."""
        return len(self.models)

    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run ensemble prediction.

        Args:
            x: Input tensor (B, C, H, W) or (B, C, D, H, W) for 3D.

        Returns:
            Tuple of (mean_prediction, std_map), both same shape as single model output.

        Raises:
            RuntimeError: If no models are available.
        """
        if len(self.models) == 0:
            raise RuntimeError("No models available in ensemble.")

        x = x.to(self.device)
        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                pred = model(x)
                predictions.append(pred)

        # Stack predictions: (n_models, B, C, H, W)
        stacked = torch.stack(predictions, dim=0)

        # Compute mean and std across ensemble dimension
        mean_prediction = stacked.mean(dim=0)
        if stacked.shape[0] > 1:
            std_map = stacked.std(dim=0)
        else:
            # Single model: std is zero (no uncertainty estimate possible)
            std_map = torch.zeros_like(mean_prediction)

        return mean_prediction, std_map

    def calibration_metrics(
        self,
        predictions: torch.Tensor,
        std_maps: torch.Tensor,
        ground_truth: torch.Tensor,
        confidence: float = 0.9,
    ) -> dict:
        """Compute calibration metrics for the ensemble predictions.

        Args:
            predictions: Mean predictions (B, C, H, W).
            std_maps: Standard deviation maps (B, C, H, W).
            ground_truth: Ground truth values (B, C, H, W).
            confidence: Confidence level for interval (default 0.9 for 90% CI).

        Returns:
            Dict with calibration metrics including coverage and sharpness.
        """
        # For Gaussian assumption, 90% CI uses z = 1.645
        import math
        z_score = self._confidence_to_z(confidence)

        # Compute confidence interval bounds
        lower = predictions - z_score * std_maps
        upper = predictions + z_score * std_maps

        # Coverage: fraction of ground truth within CI
        within_ci = (ground_truth >= lower) & (ground_truth <= upper)
        coverage = within_ci.float().mean().item()

        # Sharpness: average width of CI (smaller is better)
        ci_width = (upper - lower).mean().item()

        # Mean absolute calibration error
        expected_coverage = confidence
        calibration_error = abs(coverage - expected_coverage)

        return {
            'coverage': coverage,
            'expected_coverage': expected_coverage,
            'calibration_error': calibration_error,
            'sharpness': ci_width,
            'confidence_level': confidence,
            'n_models': len(self.models),
        }

    @staticmethod
    def _confidence_to_z(confidence: float) -> float:
        """Convert confidence level to z-score (Gaussian assumption)."""
        # Common z-scores
        z_table = {
            0.90: 1.645,
            0.95: 1.960,
            0.99: 2.576,
            0.80: 1.282,
            0.85: 1.440,
        }
        if confidence in z_table:
            return z_table[confidence]
        # Approximate using inverse normal
        import math
        # Beasley-Springer-Moro approximation
        p = (1 + confidence) / 2
        t = math.sqrt(-2 * math.log(1 - p))
        z = t - (2.515517 + 0.802853 * t + 0.010328 * t**2) / (
            1 + 1.432788 * t + 0.189269 * t**2 + 0.001308 * t**3
        )
        return z
