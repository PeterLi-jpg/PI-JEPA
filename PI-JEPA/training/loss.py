import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from ..physics.darcy import (
        physics_loss_pressure,
        physics_loss_saturation
    )
except ImportError:
    from physics.darcy import (
        physics_loss_pressure,
        physics_loss_saturation
    )


class JEPAAlignmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z_pred, z_target):
        return 2 - 2 * (z_pred * z_target).sum(dim=-1).mean()


class VarianceRegularization(nn.Module):
    def __init__(self, gamma=1.0, eps=1e-4):
        super().__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, z):
        z = z.reshape(-1, z.shape[-1])
        z = z - z.mean(dim=0, keepdim=True)

        std = torch.sqrt(z.var(dim=0, unbiased=False) + self.eps)

        return torch.mean(F.relu(self.gamma - std) ** 2)


class CovarianceRegularization(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z = z.reshape(-1, z.shape[-1])
        z = z - z.mean(dim=0, keepdim=True)

        N, D = z.shape

        z = z / (z.std(dim=0, keepdim=True) + 1e-5)

        cov = (z.T @ z) / (N - 1 + 1e-6)

        off_diag = cov.flatten()[~torch.eye(D, dtype=bool, device=z.device).flatten()]

        return (off_diag ** 2).mean()


def split_fields(x):
    p = x[:, 0:1]
    Sw = x[:, 1:2]
    return p, Sw


class PhysicsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        phys_cfg = config["loss"]["physics"]

        self.mu_w = phys_cfg.get("mu_w", 1.0)
        self.mu_n = phys_cfg.get("mu_n", 1.0)
        self.lam = phys_cfg.get("lambda", 1.0)
        self.dx = phys_cfg.get("dx", 1.0)
        self.dy = phys_cfg.get("dy", 1.0)
        self.dt = phys_cfg.get("dt", 1.0)

        self.max_clip = phys_cfg.get("max_clip", 10.0)

    def forward(self, x_pred, x_true, K, q, q_w, phi):
        p_pred, Sw_pred = split_fields(x_pred)
        p_true, Sw_true = split_fields(x_true)

        loss_p = physics_loss_pressure(
            p_pred, Sw_pred, K, q,
            self.mu_w, self.mu_n,
            self.lam,
            self.dx, self.dy
        )

        loss_s = physics_loss_saturation(
            Sw_pred, Sw_true, p_pred, K, q_w, phi,
            self.mu_w, self.mu_n,
            self.lam,
            self.dx, self.dy, self.dt
        )

        loss = loss_p + loss_s

        loss = torch.log1p(loss)

        return torch.clamp(loss, max=self.max_clip)


class LossBuilder(nn.Module):
    def __init__(self, config):
        super().__init__()

        loss_cfg = config["loss"]

        self.pred_loss = JEPAAlignmentLoss()

        self.var_reg = VarianceRegularization(
            gamma=loss_cfg["regularization"]["variance"]["gamma"]
        )

        self.cov_reg = CovarianceRegularization()

        self.physics_loss_fn = None
        if loss_cfg["physics"].get("enabled", False):
            self.physics_loss_fn = PhysicsLoss(config)

        # Loss weights pulled directly from config. The legacy LossBuilder
        # used to coerce cov→max(5.0, ...) and multiply physics by ×5; both
        # silently overrode the configured values. Removed to keep config
        # the single source of truth.
        self.weights = {
            "pred": loss_cfg["prediction"]["weight"],
            "var": loss_cfg["regularization"]["variance"]["weight"],
            "cov": loss_cfg["regularization"]["covariance"]["weight"],
            "physics": loss_cfg["physics"]["weight"],
        }
        # Cache these so forward() doesn't need the full config closure.
        self._recon_weight = float(loss_cfg.get("reconstruction", {}).get("weight", 0.0))
        self._physics_ramp_steps = float(loss_cfg.get("physics", {}).get("ramp_steps", 200))

        self.step = 0

    def forward(
        self,
        z_pred,
        z_target,
        x_pred=None,
        x_true=None,
        K=None,
        q=None,
        q_w=None,
        phi=None
    ):
        losses = {}

        assert z_pred.shape == z_target.shape

        self.step += 1

        losses["jepa"] = self.pred_loss(z_pred, z_target)

        z_all = torch.cat([z_pred, z_target], dim=0)

        losses["variance"] = self.var_reg(z_all)
        losses["covariance"] = self.cov_reg(z_all)

        if self.physics_loss_fn is not None:
            if all(v is not None for v in [x_pred, x_true, K, q, q_w, phi]):
                losses["physics"] = self.physics_loss_fn(
                    x_pred, x_true, K, q, q_w, phi
                )

        if x_pred is not None and x_true is not None:
            losses["recon"] = F.mse_loss(x_pred, x_true)

        total = (
            self.weights["pred"] * losses["jepa"] +
            self.weights["var"] * losses["variance"] +
            self.weights["cov"] * losses["covariance"]
        )

        # Optional reconstruction term, gated by config weight (not hardcoded).
        if "recon" in losses and self._recon_weight > 0:
            total += self._recon_weight * losses["recon"]

        if "physics" in losses:
            # Ramp weight fully governed by config.
            physics_weight = self.weights["physics"] * min(
                1.0, self.step / max(self._physics_ramp_steps, 1)
            )
            total += physics_weight * losses["physics"]

        losses["total"] = total

        return losses
