"""Rollout evaluation with noise annealing."""

import math
from typing import Literal

import torch


class NoiseSchedule:
    """Noise annealing schedule for rollout stability."""
    
    def __init__(
        self,
        schedule_type: Literal['linear', 'cosine', 'none'] = 'linear',
        sigma_start: float = 1e-2,
        sigma_end: float = 1e-4,
        total_steps: int = 40
    ):
        if schedule_type not in ('linear', 'cosine', 'none'):
            raise ValueError(f"schedule_type must be 'linear', 'cosine', or 'none', got '{schedule_type}'")
        if sigma_start < 0:
            raise ValueError(f"sigma_start must be non-negative, got {sigma_start}")
        if sigma_end < 0:
            raise ValueError(f"sigma_end must be non-negative, got {sigma_end}")
        if total_steps < 1:
            raise ValueError(f"total_steps must be at least 1, got {total_steps}")
        
        self.schedule_type = schedule_type
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.total_steps = total_steps
    
    def get_sigma(self, step: int) -> float:
        """Get noise level σ at the given step."""
        if step < 0:
            step = 0
        if step >= self.total_steps:
            step = self.total_steps - 1
        
        if self.schedule_type == 'none':
            return self.sigma_start
        
        t_ratio = step / max(self.total_steps - 1, 1)
        
        if self.schedule_type == 'linear':
            sigma = self.sigma_start - (self.sigma_start - self.sigma_end) * t_ratio
        elif self.schedule_type == 'cosine':
            sigma = self.sigma_end + 0.5 * (self.sigma_start - self.sigma_end) * (1 + math.cos(math.pi * t_ratio))
        else:
            sigma = self.sigma_start
        
        return sigma
    
    def __call__(self, step: int) -> float:
        return self.get_sigma(step)


def rollout(
    model,
    x_init,
    steps,
    decoder,
    noise_std=0.0,
    device="cuda"
):
    model.eval()
    x = x_init.to(device)
    outputs = []

    with torch.no_grad():
        for _ in range(steps):
            z_full = model.encoder(x)

            B, N, D = z_full.shape

            context_indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
            target_indices = context_indices

            z_pred = model.predict_latent(
                z_full,
                context_indices,
                target_indices
            )

            if noise_std > 0.0:
                z_pred = z_pred + noise_std * torch.randn_like(z_pred)

            x = decoder(z_pred)

            outputs.append(x)

    return torch.stack(outputs, dim=1)


def rollout_with_metrics(
    model,
    x_init,
    ground_truth,
    steps,
    decoder,
    metric_fn,
    noise_std=0.0,
    device="cuda"
):
    model.eval()
    x = x_init.to(device)
    outputs = []
    metrics = []

    with torch.no_grad():
        for t in range(steps):
            z_full = model.encoder(x)

            B, N, D = z_full.shape

            context_indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
            target_indices = context_indices

            z_pred = model.predict_latent(
                z_full,
                context_indices,
                target_indices
            )

            if noise_std > 0.0:
                z_pred = z_pred + noise_std * torch.randn_like(z_pred)

            x = decoder(z_pred)

            outputs.append(x)

            if ground_truth is not None:
                gt = ground_truth[:, t].to(device)
                metrics.append(metric_fn(x, gt))

    outputs = torch.stack(outputs, dim=1)

    if len(metrics) > 0:
        metrics = torch.stack(metrics, dim=1)
        return outputs, metrics

    return outputs


def single_step(
    model,
    x,
    decoder,
    noise_std=0.0,
    device="cuda"
):
    model.eval()

    with torch.no_grad():
        x = x.to(device)

        z_full = model.encoder(x)

        B, N, D = z_full.shape

        context_indices = torch.arange(N, device=device).unsqueeze(0).repeat(B, 1)
        target_indices = context_indices

        z_pred = model.predict_latent(
            z_full,
            context_indices,
            target_indices
        )

        if noise_std > 0.0:
            z_pred = z_pred + noise_std * torch.randn_like(z_pred)

        x_next = decoder(z_pred)

    return x_next



class RolloutEvaluator:
    """Multi-horizon rollout evaluation with noise annealing."""

    def __init__(
        self,
        model,
        decoder,
        noise_schedule: NoiseSchedule = None,
        device: str = 'cpu'
    ):
        self.model = model
        self.decoder = decoder
        self.noise_schedule = noise_schedule
        self.device = device
        self.model.to(device)
        self.decoder.to(device)

    def rollout_single(self, x_init: torch.Tensor, steps: int) -> torch.Tensor:
        """Perform closed-loop autoregressive rollout."""
        self.model.eval()
        x = x_init.to(self.device)
        outputs = []

        # Create noise schedule for this rollout if enabled
        if self.noise_schedule is not None:
            # Create a new schedule with the correct total_steps for this rollout
            rollout_noise_schedule = NoiseSchedule(
                schedule_type=self.noise_schedule.schedule_type,
                sigma_start=self.noise_schedule.sigma_start,
                sigma_end=self.noise_schedule.sigma_end,
                total_steps=steps
            )
        else:
            rollout_noise_schedule = None

        with torch.no_grad():
            for t in range(steps):
                # Encode current state
                z_full = self.model.encoder(x)

                B, N, D = z_full.shape

                # Use all patches as both context and target
                context_indices = torch.arange(N, device=self.device).unsqueeze(0).repeat(B, 1)
                target_indices = context_indices

                # Predict next latent state
                z_pred = self.model.predict_latent(
                    z_full,
                    context_indices,
                    target_indices
                )

                # Inject Gaussian noise: ẑ ← ẑ + σ(t)·ε where ε ~ N(0, I)
                if rollout_noise_schedule is not None:
                    sigma = rollout_noise_schedule.get_sigma(t)
                    if sigma > 0:
                        epsilon = torch.randn_like(z_pred)
                        z_pred = z_pred + sigma * epsilon

                # Decode to output space
                x = self.decoder(z_pred)

                outputs.append(x)

        return torch.stack(outputs, dim=1)

    def evaluate(
        self,
        dataloader,
        horizons: list = None,
        metric_fn=None
    ) -> dict:
        """Evaluate model at multiple rollout horizons."""
        if horizons is None:
            horizons = [1, 5, 10, 20, 40]

        if metric_fn is None:
            metric_fn = self._relative_l2_error

        self.model.eval()

        # Initialize results storage
        results = {
            'horizons': horizons,
            'relative_l2': {},
            'per_sample_errors': {h: [] for h in horizons},
            'cumulative_errors': {}
        }

        max_horizon = max(horizons)

        with torch.no_grad():
            for batch in dataloader:
                # Handle different batch formats
                if isinstance(batch, dict):
                    x = batch.get('input', batch.get('x'))
                    y = batch.get('target', batch.get('y', None))
                    if x is not None:
                        if x.dim() == 3:
                            x = x.unsqueeze(1)
                    if y is not None:
                        if y.dim() == 3:
                            y = y.unsqueeze(1)
                        x_init = torch.cat([y, x], dim=1)
                    else:
                        x_init = x
                    ground_truth = None
                elif isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        x_init, ground_truth = batch[0], batch[1]
                    else:
                        x_init = batch[0]
                        ground_truth = None
                else:
                    x_init = batch
                    ground_truth = None

                x_init = x_init.to(self.device)
                if ground_truth is not None:
                    ground_truth = ground_truth.to(self.device)

                # Determine available timesteps
                if ground_truth is not None:
                    available_steps = ground_truth.shape[1]
                else:
                    available_steps = max_horizon

                # Perform rollout up to max horizon (capped by available data)
                rollout_steps = min(max_horizon, available_steps)
                predictions = self.rollout_single(x_init, rollout_steps)

                # Compute metrics at each horizon
                for h in horizons:
                    if h > rollout_steps:
                        continue

                    # Get prediction at horizon h (0-indexed, so h-1)
                    pred_h = predictions[:, h-1]

                    if ground_truth is not None and h <= ground_truth.shape[1]:
                        target_h = ground_truth[:, h-1]

                        # Compute per-sample error
                        error = metric_fn(pred_h, target_h)
                        results['per_sample_errors'][h].append(error.item())

        # Aggregate results
        for h in horizons:
            if results['per_sample_errors'][h]:
                errors = results['per_sample_errors'][h]
                results['relative_l2'][h] = sum(errors) / len(errors)

        # Compute cumulative errors
        cumulative = 0.0
        for h in sorted(horizons):
            if h in results['relative_l2']:
                cumulative += results['relative_l2'][h]
                results['cumulative_errors'][h] = cumulative

        return results

    def evaluate_ood(
        self,
        ood_dataloader,
        horizons: list = None,
        metric_fn=None
    ) -> dict:
        """Evaluate on out-of-distribution data."""
        if horizons is None:
            horizons = [1, 5, 10, 20, 40]

        # Use the same evaluation logic as evaluate()
        results = self.evaluate(ood_dataloader, horizons, metric_fn)

        # Rename keys to indicate OOD evaluation
        results['ood_relative_l2'] = results.pop('relative_l2')
        results['is_ood'] = True

        return results

    def evaluate_per_step(
        self,
        x_init: torch.Tensor,
        ground_truth: torch.Tensor,
        steps: int = 10,
        metric_fn=None,
    ) -> dict:
        """Evaluate model computing error at each autoregressive step.

        Produces error curves suitable for plotting rollout degradation.

        Args:
            x_init: Initial condition (B, C, H, W).
            ground_truth: Ground truth trajectory (B, T, C, H, W) or (B, T, ...).
            steps: Number of autoregressive steps (at least 10 recommended).
            metric_fn: Error metric function. Defaults to relative L2.

        Returns:
            Dict with:
                - 'predictions': Tensor of shape (B, steps, ...)
                - 'errors': List of T error values (one per step)
                - 'steps': Number of steps evaluated
        """
        if metric_fn is None:
            metric_fn = self._relative_l2_error

        # Cap steps to available ground truth
        available_steps = ground_truth.shape[1] if ground_truth is not None else steps
        actual_steps = min(steps, available_steps)

        # Perform rollout
        predictions = self.rollout_single(x_init, actual_steps)

        # Compute per-step errors
        errors = []
        for t in range(actual_steps):
            pred_t = predictions[:, t]
            if ground_truth is not None:
                gt_t = ground_truth[:, t].to(self.device)
                error = metric_fn(pred_t, gt_t)
                if isinstance(error, torch.Tensor):
                    error = error.item()
                errors.append(error)
            else:
                errors.append(0.0)

        return {
            'predictions': predictions,
            'errors': errors,
            'steps': actual_steps,
        }

    def produce_error_curves(
        self,
        dataloader,
        steps: int = 10,
        metric_fn=None,
    ) -> dict:
        """Produce error curves over a dataset for plotting.

        Args:
            dataloader: DataLoader yielding (x_init, ground_truth) batches.
            steps: Number of rollout steps.
            metric_fn: Error metric function.

        Returns:
            Dict with averaged error curve and per-sample details.
        """
        if metric_fn is None:
            metric_fn = self._relative_l2_error

        all_errors = []  # List of per-step error lists

        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                x_init, ground_truth = batch[0], batch[1]
            else:
                continue

            result = self.evaluate_per_step(x_init, ground_truth, steps, metric_fn)
            all_errors.append(result['errors'])

        if not all_errors:
            return {'mean_errors': [], 'steps': 0}

        # Average errors across batches
        n_steps = len(all_errors[0])
        mean_errors = []
        for t in range(n_steps):
            step_errors = [errs[t] for errs in all_errors if t < len(errs)]
            mean_errors.append(sum(step_errors) / len(step_errors))

        return {
            'mean_errors': mean_errors,
            'all_errors': all_errors,
            'steps': n_steps,
        }

    def _relative_l2_error(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Relative L2 error: ||û - u||_2 / ||u||_2."""
        # Flatten spatial dimensions
        pred_flat = pred.reshape(pred.shape[0], -1)
        target_flat = target.reshape(target.shape[0], -1)

        # Compute L2 norms
        diff_norm = torch.norm(pred_flat - target_flat, p=2, dim=1)
        target_norm = torch.norm(target_flat, p=2, dim=1)

        # Avoid division by zero
        target_norm = torch.clamp(target_norm, min=1e-8)

        # Relative L2 error per sample, then mean
        relative_error = diff_norm / target_norm

        return relative_error.mean()
