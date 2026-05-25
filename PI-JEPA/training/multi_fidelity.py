"""
Multi-Fidelity Pretraining Module.

Implements three-tier data hierarchy with configurable weights and
progressive fidelity introduction based on epoch thresholds.

Requirements: 20.1, 20.2, 20.3
"""

from typing import List, Optional, Dict, Any

import torch
from torch.utils.data import Dataset, Sampler


class MultiFidelityPretrainer:
    """Three-tier pretraining: analytical → medium numerical → high-fidelity.

    Weights samples by fidelity tier and progressively introduces
    higher-fidelity data during training.
    """

    def __init__(
        self,
        tier_weights: Optional[List[float]] = None,
        progressive: bool = True,
        tier_introduction_epochs: Optional[List[int]] = None,
        n_tiers: int = 3,
    ):
        """Initialize multi-fidelity pretrainer.

        Args:
            tier_weights: Sampling weight for each tier (must sum to > 0).
                         Default: [0.5, 0.3, 0.2] for 3 tiers.
            progressive: Whether to progressively introduce tiers.
            tier_introduction_epochs: Epoch at which each tier becomes active.
                                     Default: [0, 100, 300].
            n_tiers: Number of fidelity tiers.
        """
        self.n_tiers = n_tiers

        if tier_weights is None:
            tier_weights = [0.5, 0.3, 0.2]
        if len(tier_weights) != n_tiers:
            raise ValueError(
                f"tier_weights length ({len(tier_weights)}) must match n_tiers ({n_tiers})"
            )
        if any(w < 0 for w in tier_weights):
            raise ValueError("All tier weights must be non-negative")

        self.tier_weights = tier_weights

        if tier_introduction_epochs is None:
            tier_introduction_epochs = [0, 100, 300]
        if len(tier_introduction_epochs) != n_tiers:
            raise ValueError(
                f"tier_introduction_epochs length ({len(tier_introduction_epochs)}) "
                f"must match n_tiers ({n_tiers})"
            )
        # Ensure introduction epochs are non-decreasing
        for i in range(1, len(tier_introduction_epochs)):
            if tier_introduction_epochs[i] < tier_introduction_epochs[i - 1]:
                raise ValueError("tier_introduction_epochs must be non-decreasing")

        self.tier_introduction_epochs = tier_introduction_epochs
        self.progressive = progressive

    def get_active_tiers(self, epoch: int) -> List[int]:
        """Get list of active tier indices at the given epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            List of active tier indices (0-based).
        """
        if not self.progressive:
            return list(range(self.n_tiers))

        active = []
        for i, intro_epoch in enumerate(self.tier_introduction_epochs):
            if epoch >= intro_epoch:
                active.append(i)
        return active

    def get_sampling_weights(self, epoch: int) -> List[float]:
        """Get normalized sampling weights for the current epoch.

        Only active tiers have non-zero weight. Weights are renormalized
        to sum to 1.0 among active tiers.

        Args:
            epoch: Current training epoch.

        Returns:
            List of sampling weights for each tier (length n_tiers).
        """
        active_tiers = self.get_active_tiers(epoch)

        weights = [0.0] * self.n_tiers
        total = 0.0
        for i in active_tiers:
            weights[i] = self.tier_weights[i]
            total += self.tier_weights[i]

        # Normalize to sum to 1
        if total > 0:
            weights = [w / total for w in weights]

        return weights

    def sample_tier(self, epoch: int) -> int:
        """Sample a tier index according to current weights.

        Args:
            epoch: Current training epoch.

        Returns:
            Sampled tier index.
        """
        weights = self.get_sampling_weights(epoch)
        # Use torch for sampling
        probs = torch.tensor(weights, dtype=torch.float32)
        if probs.sum() == 0:
            # Fallback to tier 0
            return 0
        return torch.multinomial(probs, 1).item()

    def build_epoch_sampler(
        self,
        tier_datasets: List[Dataset],
        epoch: int,
        batch_size: int,
        total_samples: int,
    ) -> List[Dict[str, Any]]:
        """Build a sampling plan for one epoch.

        Args:
            tier_datasets: List of datasets, one per tier.
            epoch: Current training epoch.
            batch_size: Batch size.
            total_samples: Total number of samples to draw this epoch.

        Returns:
            List of dicts with 'tier' and 'indices' keys specifying
            which samples to draw from which tier.
        """
        weights = self.get_sampling_weights(epoch)
        plan = []

        for tier_idx in range(self.n_tiers):
            tier_weight = weights[tier_idx]
            if tier_weight <= 0 or tier_idx >= len(tier_datasets):
                continue

            n_samples_tier = max(1, int(total_samples * tier_weight))
            dataset_size = len(tier_datasets[tier_idx])

            if dataset_size == 0:
                continue

            # Sample indices with replacement if needed
            indices = torch.randint(0, dataset_size, (n_samples_tier,)).tolist()
            plan.append({
                'tier': tier_idx,
                'indices': indices,
                'weight': tier_weight,
            })

        return plan

    def get_tier_info(self, epoch: int) -> Dict[str, Any]:
        """Get information about tier status at given epoch.

        Args:
            epoch: Current training epoch.

        Returns:
            Dict with tier status information.
        """
        active = self.get_active_tiers(epoch)
        weights = self.get_sampling_weights(epoch)

        return {
            'epoch': epoch,
            'active_tiers': active,
            'n_active': len(active),
            'sampling_weights': weights,
            'progressive': self.progressive,
        }
