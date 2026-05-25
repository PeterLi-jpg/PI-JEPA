import torch
import torch.nn.functional as F


def grad_x(u, dx):
    u_pad = F.pad(u, (1, 1, 0, 0), mode="reflect")
    return (u_pad[:, :, :, 2:] - u_pad[:, :, :, :-2]) / (2 * dx + 1e-6)


def grad_y(u, dy):
    u_pad = F.pad(u, (0, 0, 1, 1), mode="reflect")
    return (u_pad[:, :, 2:, :] - u_pad[:, :, :-2, :]) / (2 * dy + 1e-6)


def divergence(fx, fy, dx, dy):
    fx_pad = F.pad(fx, (1, 1, 0, 0), mode="reflect")
    fy_pad = F.pad(fy, (0, 0, 1, 1), mode="reflect")

    dfx_dx = (fx_pad[:, :, :, 2:] - fx_pad[:, :, :, :-2]) / (2 * dx + 1e-6)
    dfy_dy = (fy_pad[:, :, 2:, :] - fy_pad[:, :, :-2, :]) / (2 * dy + 1e-6)

    return dfx_dx + dfy_dy


# =============================================================================
# 3D primitives — second-order central differences with reflective BCs.
# Tensors are (B, C, D, H, W). F.pad ordering for 3D is
# (W_left, W_right, H_left, H_right, D_left, D_right).
# =============================================================================


def grad_x_3d(u, dx):
    """∂/∂x along the W axis."""
    u_pad = F.pad(u, (1, 1, 0, 0, 0, 0), mode="reflect")
    return (u_pad[:, :, :, :, 2:] - u_pad[:, :, :, :, :-2]) / (2 * dx + 1e-6)


def grad_y_3d(u, dy):
    """∂/∂y along the H axis."""
    u_pad = F.pad(u, (0, 0, 1, 1, 0, 0), mode="reflect")
    return (u_pad[:, :, :, 2:, :] - u_pad[:, :, :, :-2, :]) / (2 * dy + 1e-6)


def grad_z_3d(u, dz):
    """∂/∂z along the D axis (depth)."""
    u_pad = F.pad(u, (0, 0, 0, 0, 1, 1), mode="reflect")
    return (u_pad[:, :, 2:, :, :] - u_pad[:, :, :-2, :, :]) / (2 * dz + 1e-6)


def divergence_3d(fx, fy, fz, dx, dy, dz):
    """∇·(fx, fy, fz) for 5D tensors (B, C, D, H, W)."""
    fx_pad = F.pad(fx, (1, 1, 0, 0, 0, 0), mode="reflect")
    fy_pad = F.pad(fy, (0, 0, 1, 1, 0, 0), mode="reflect")
    fz_pad = F.pad(fz, (0, 0, 0, 0, 1, 1), mode="reflect")

    dfx_dx = (fx_pad[:, :, :, :, 2:] - fx_pad[:, :, :, :, :-2]) / (2 * dx + 1e-6)
    dfy_dy = (fy_pad[:, :, :, 2:, :] - fy_pad[:, :, :, :-2, :]) / (2 * dy + 1e-6)
    dfz_dz = (fz_pad[:, :, 2:, :, :] - fz_pad[:, :, :-2, :, :]) / (2 * dz + 1e-6)
    return dfx_dx + dfy_dy + dfz_dz


def darcy_residual_3d(p, K, q=None, dx=1.0, dy=1.0, dz=1.0):
    """Steady-state single-phase Darcy residual in 3D: -∇·(K ∇p) - q.

    Args:
        p:  (B, 1, D, H, W) pressure field (decoded)
        K:  (B, 1, D, H, W) permeability field (conditioning)
        q:  (B, 1, D, H, W) source term, or None → assumed zero
    Returns:
        residual: (B, 1, D, H, W). Squaring and meaning gives the PDE loss.
    """
    if p.dim() != 5 or K.dim() != 5:
        raise ValueError(
            f"darcy_residual_3d expects 5D inputs (B,1,D,H,W); got p={tuple(p.shape)}, K={tuple(K.shape)}"
        )
    dp_dx = grad_x_3d(p, dx)
    dp_dy = grad_y_3d(p, dy)
    dp_dz = grad_z_3d(p, dz)

    flux_x = -K * dp_dx
    flux_y = -K * dp_dy
    flux_z = -K * dp_dz

    div_term = divergence_3d(flux_x, flux_y, flux_z, dx, dy, dz)
    if q is None:
        return div_term
    return div_term - q


# =============================================================================
# Spectral physics residuals.
# Paper contribution (iii): the original PI-JEPA paper found its FD physics
# residual neutral-to-harmful (-8.1% in the ablation). The hypothesis from
# the paper's own Discussion was that the finite-difference Laplacian
# stencil introduces dispersion artifacts that conflict with the JEPA loss.
# Spectral derivatives don't have dispersion error: ∂/∂x ↔ i·k_x in Fourier
# space. We assume periodic boundaries (reasonable approximation for our
# synthetic Darcy on the unit cube with a centered Gaussian source).
# =============================================================================


def _spectral_grad_3d(u: torch.Tensor, dx: float, dy: float, dz: float):
    """Compute (∂u/∂x, ∂u/∂y, ∂u/∂z) via 3D rFFT.

    u: (B, C, D, H, W). Returns three tensors of the same shape.
    Periodic BCs assumed.
    """
    B, C, D, H, W = u.shape
    u_ft = torch.fft.rfftn(u, dim=(-3, -2, -1))

    # Build wavenumber grids matching rFFT layout. rFFT halves the LAST dim.
    kz = torch.fft.fftfreq(D, d=dz, device=u.device) * 2 * torch.pi   # (D,)
    ky = torch.fft.fftfreq(H, d=dy, device=u.device) * 2 * torch.pi   # (H,)
    kx = torch.fft.rfftfreq(W, d=dx, device=u.device) * 2 * torch.pi  # (W//2+1,)

    KZ = kz.view(D, 1, 1)
    KY = ky.view(1, H, 1)
    KX = kx.view(1, 1, -1)

    # ∂u/∂x via i * k_x * û
    dxu_ft = 1j * KX * u_ft
    dyu_ft = 1j * KY * u_ft
    dzu_ft = 1j * KZ * u_ft

    dxu = torch.fft.irfftn(dxu_ft, s=(D, H, W), dim=(-3, -2, -1))
    dyu = torch.fft.irfftn(dyu_ft, s=(D, H, W), dim=(-3, -2, -1))
    dzu = torch.fft.irfftn(dzu_ft, s=(D, H, W), dim=(-3, -2, -1))
    return dxu, dyu, dzu


def _spectral_div_3d(fx: torch.Tensor, fy: torch.Tensor, fz: torch.Tensor,
                     dx: float, dy: float, dz: float) -> torch.Tensor:
    """Compute ∇·(fx, fy, fz) via spectral derivatives. Periodic BCs."""
    B, C, D, H, W = fx.shape
    fx_ft = torch.fft.rfftn(fx, dim=(-3, -2, -1))
    fy_ft = torch.fft.rfftn(fy, dim=(-3, -2, -1))
    fz_ft = torch.fft.rfftn(fz, dim=(-3, -2, -1))

    kz = torch.fft.fftfreq(D, d=dz, device=fx.device) * 2 * torch.pi
    ky = torch.fft.fftfreq(H, d=dy, device=fx.device) * 2 * torch.pi
    kx = torch.fft.rfftfreq(W, d=dx, device=fx.device) * 2 * torch.pi

    KZ = kz.view(D, 1, 1)
    KY = ky.view(1, H, 1)
    KX = kx.view(1, 1, -1)

    div_ft = 1j * (KX * fx_ft + KY * fy_ft + KZ * fz_ft)
    return torch.fft.irfftn(div_ft, s=(D, H, W), dim=(-3, -2, -1))


def spectral_darcy_residual_3d(p, K, q=None, dx=1.0, dy=1.0, dz=1.0):
    """Spectral (FFT-based) 3D Darcy residual: -∇·(K ∇p) - q.

    Mathematically identical to darcy_residual_3d but uses exact spectral
    derivatives instead of second-order central differences. Should produce
    a cleaner gradient signal that doesn't conflict with the JEPA loss the
    way the FD residual did in the original paper.
    """
    if p.dim() != 5 or K.dim() != 5:
        raise ValueError(
            f"spectral_darcy_residual_3d expects 5D (B,1,D,H,W); got p={p.shape}, K={K.shape}"
        )
    dpx, dpy, dpz = _spectral_grad_3d(p, dx, dy, dz)
    flux_x = -K * dpx
    flux_y = -K * dpy
    flux_z = -K * dpz
    div_term = _spectral_div_3d(flux_x, flux_y, flux_z, dx, dy, dz)
    if q is None:
        return div_term
    return div_term - q


def _spectral_grad_2d(u: torch.Tensor, dx: float, dy: float):
    """2D analog of _spectral_grad_3d. u: (B, C, H, W)."""
    B, C, H, W = u.shape
    u_ft = torch.fft.rfft2(u, dim=(-2, -1))
    ky = torch.fft.fftfreq(H, d=dy, device=u.device) * 2 * torch.pi
    kx = torch.fft.rfftfreq(W, d=dx, device=u.device) * 2 * torch.pi
    KY = ky.view(H, 1)
    KX = kx.view(1, -1)
    dxu_ft = 1j * KX * u_ft
    dyu_ft = 1j * KY * u_ft
    dxu = torch.fft.irfft2(dxu_ft, s=(H, W), dim=(-2, -1))
    dyu = torch.fft.irfft2(dyu_ft, s=(H, W), dim=(-2, -1))
    return dxu, dyu


def _spectral_div_2d(fx, fy, dx: float, dy: float):
    """2D divergence via FFT."""
    B, C, H, W = fx.shape
    fx_ft = torch.fft.rfft2(fx, dim=(-2, -1))
    fy_ft = torch.fft.rfft2(fy, dim=(-2, -1))
    ky = torch.fft.fftfreq(H, d=dy, device=fx.device) * 2 * torch.pi
    kx = torch.fft.rfftfreq(W, d=dx, device=fx.device) * 2 * torch.pi
    KY = ky.view(H, 1)
    KX = kx.view(1, -1)
    div_ft = 1j * (KX * fx_ft + KY * fy_ft)
    return torch.fft.irfft2(div_ft, s=(H, W), dim=(-2, -1))


def spectral_darcy_residual_2d(p, K, q=None, dx=1.0, dy=1.0):
    """Spectral (FFT-based) 2D Darcy residual: -∇·(K ∇p) - q."""
    if p.dim() != 4 or K.dim() != 4:
        raise ValueError(
            f"spectral_darcy_residual_2d expects 4D (B,1,H,W); got p={p.shape}, K={K.shape}"
        )
    dpx, dpy = _spectral_grad_2d(p, dx, dy)
    flux_x = -K * dpx
    flux_y = -K * dpy
    div_term = _spectral_div_2d(flux_x, flux_y, dx, dy)
    if q is None:
        return div_term
    return div_term - q


def effective_saturation(Sw, Swr=0.0, Snr=0.0):
    return (Sw - Swr) / (1.0 - Swr - Snr + 1e-6)


def rel_perm(Sw, krw_exp, kro_exp, Swr=0.0, Snr=0.0):
    """LEGACY simple-power-law relperm: k_rw = Se^krw_exp, k_rn = (1-Se)^kro_exp.

    Kept for back-compat. NOT the Brooks-Corey form the paper specifies.
    For the published Brooks-Corey form (Eq. 10 of the PI-JEPA paper), use
    `brooks_corey_rel_perm(Sw, lambda_bc, Swr, Snr)` below.
    """
    Se = effective_saturation(Sw, Swr, Snr)
    Se = Se.clamp(1e-4, 1.0 - 1e-4)

    krw = Se ** krw_exp
    kro = (1.0 - Se) ** kro_exp

    return krw, kro


def brooks_corey_rel_perm(Sw, lambda_bc, Swr=0.0, Snr=0.0):
    """Brooks-Corey relative permeabilities (the paper's Eq. 10 formulation).

        k_rw = S_e^((2 + 3λ) / λ)
        k_rn = (1 - S_e)^2 * (1 - S_e^((2 + λ) / λ))

    where S_e = (S_w - S_wr) / (1 - S_wr - S_nr) and λ is the pore-size
    distribution index.

    This was previously only available via the dead `BrooksCoreyModel` class;
    exposing it as a function so the LIVE physics-loss path can use it.
    """
    Se = effective_saturation(Sw, Swr, Snr)
    Se = Se.clamp(1e-4, 1.0 - 1e-4)

    exp_w = (2.0 + 3.0 * lambda_bc) / lambda_bc
    exp_n = (2.0 + lambda_bc) / lambda_bc

    krw = Se ** exp_w
    krn = (1.0 - Se) ** 2 * (1.0 - Se ** exp_n)
    return krw, krn


def mobility(Sw, mu_w, mu_o, krw_exp, kro_exp, lambda_bc=None):
    """Phase mobilities and water fractional flow.

    If `lambda_bc` is provided, uses the Brooks-Corey relperm form (matches
    the PI-JEPA paper Eq. 10). Otherwise falls back to the legacy simple
    power-law `rel_perm(.., krw_exp, kro_exp)` for back-compat.
    """
    if lambda_bc is not None:
        krw, kro = brooks_corey_rel_perm(Sw, lambda_bc)
    else:
        krw, kro = rel_perm(Sw, krw_exp, kro_exp)

    lambda_w = krw / (mu_w + 1e-6)
    lambda_o = kro / (mu_o + 1e-6)

    lambda_t = lambda_w + lambda_o + 1e-6
    fw = lambda_w / lambda_t

    return lambda_t, fw


def physics_loss_pressure(
    p, Sw, K, q,
    mu_w, mu_o,
    krw_exp, kro_exp,
    dx=1.0, dy=1.0
):
    p = torch.tanh(p) * 5.0
    Sw = torch.sigmoid(Sw)

    lambda_t, _ = mobility(Sw, mu_w, mu_o, krw_exp, kro_exp)

    dp_dx = grad_x(p, dx)
    dp_dy = grad_y(p, dy)

    vx = -K * lambda_t * dp_dx
    vy = -K * lambda_t * dp_dy

    div_term = divergence(vx, vy, dx, dy)

    residual = div_term - q

    loss = (residual ** 2).mean()

    grad_penalty = dp_dx.abs().mean() + dp_dy.abs().mean()

    return loss + 0.05 * grad_penalty


def physics_loss_saturation(
    Sw_pred, Sw_true, p, K, q_w, phi,
    mu_w, mu_o,
    krw_exp, kro_exp,
    dx=1.0, dy=1.0, dt=1.0
):
    p = torch.tanh(p) * 5.0
    Sw_pred = torch.sigmoid(Sw_pred)
    Sw_true = torch.sigmoid(Sw_true)

    lambda_t, fw = mobility(Sw_pred, mu_w, mu_o, krw_exp, kro_exp)

    dp_dx = grad_x(p, dx)
    dp_dy = grad_y(p, dy)

    vx = -K * lambda_t * dp_dx
    vy = -K * lambda_t * dp_dy

    fx = fw * vx
    fy = fw * vy

    div_term = divergence(fx, fy, dx, dy)

    dSw_dt = (Sw_pred - Sw_true) / (dt + 1e-6)
    q_term = q_w / (phi + 1e-6)

    residual = dSw_dt + div_term - q_term

    loss = (residual ** 2).mean()

    grad_penalty = dp_dx.abs().mean() + dp_dy.abs().mean()

    return loss + 0.05 * grad_penalty


class BrooksCoreyModel:
    """Brooks-Corey relative permeability model."""

    def __init__(self, S_wr: float = 0.0, S_nr: float = 0.0, lambda_bc: float = 2.0):
        self.S_wr = S_wr
        self.S_nr = S_nr
        self.lambda_bc = lambda_bc

    def effective_saturation(self, S_w: torch.Tensor) -> torch.Tensor:
        """S_e = (S_w - S_wr) / (1 - S_wr - S_nr)."""
        denominator = 1.0 - self.S_wr - self.S_nr
        S_e = (S_w - self.S_wr) / (denominator + 1e-8)
        return S_e.clamp(0.0, 1.0)

    def relative_permeability_water(self, S_e: torch.Tensor) -> torch.Tensor:
        """k_rw = S_e^((2 + 3λ) / λ)."""
        S_e_safe = S_e.clamp(1e-8, 1.0)
        exponent = (2.0 + 3.0 * self.lambda_bc) / self.lambda_bc
        return S_e_safe ** exponent

    def relative_permeability_nonwetting(self, S_e: torch.Tensor) -> torch.Tensor:
        """k_rn = (1 - S_e)² · (1 - S_e^((2 + λ) / λ))."""
        S_e_safe = S_e.clamp(0.0, 1.0 - 1e-8)
        term1 = (1.0 - S_e_safe) ** 2
        exponent = (2.0 + self.lambda_bc) / self.lambda_bc
        term2 = 1.0 - S_e_safe ** exponent
        return term1 * term2

    def capillary_pressure(self, S_e: torch.Tensor, P_entry: float = 1.0) -> torch.Tensor:
        """P_c = P_entry · S_e^(-1/λ)."""
        S_e_safe = S_e.clamp(1e-8, 1.0)
        exponent = -1.0 / self.lambda_bc
        return P_entry * (S_e_safe ** exponent)


class TwoPhaseDarcyPhysics:
    """Two-phase Darcy flow with capillary pressure."""

    def __init__(
        self,
        brooks_corey: BrooksCoreyModel,
        mu_w: float = 1.0,
        mu_n: float = 1.0,
        collocation_size: int = 32,
        dx: float = 1.0,
        dy: float = 1.0
    ):
        self.brooks_corey = brooks_corey
        self.mu_w = mu_w
        self.mu_n = mu_n
        self.collocation_size = collocation_size
        self.dx = dx
        self.dy = dy

    def fractional_flow(self, S_w: torch.Tensor) -> torch.Tensor:
        """f_w = (k_rw/μ_w) / (k_rw/μ_w + k_rn/μ_n)."""
        S_e = self.brooks_corey.effective_saturation(S_w)
        k_rw = self.brooks_corey.relative_permeability_water(S_e)
        k_rn = self.brooks_corey.relative_permeability_nonwetting(S_e)
        lambda_w = k_rw / (self.mu_w + 1e-8)
        lambda_n = k_rn / (self.mu_n + 1e-8)
        lambda_total = lambda_w + lambda_n + 1e-8
        return lambda_w / lambda_total

    def _compute_mobilities(self, S_w: torch.Tensor):
        S_e = self.brooks_corey.effective_saturation(S_w)
        k_rw = self.brooks_corey.relative_permeability_water(S_e)
        k_rn = self.brooks_corey.relative_permeability_nonwetting(S_e)
        lambda_w = k_rw / (self.mu_w + 1e-8)
        lambda_n = k_rn / (self.mu_n + 1e-8)
        lambda_T = lambda_w + lambda_n + 1e-8
        return lambda_w, lambda_n, lambda_T

    def pressure_residual(
        self,
        p_w: torch.Tensor,
        S_w: torch.Tensor,
        K: torch.Tensor,
        q_T: torch.Tensor
    ) -> torch.Tensor:
        """R_1: -∇·(λ_T K ∇p_w) + ∇·(λ_n K ∇P_c(S_w)) = q_T."""
        # Ensure 4D tensors for consistent processing
        if p_w.dim() == 3:
            p_w = p_w.unsqueeze(1)
        if S_w.dim() == 3:
            S_w = S_w.unsqueeze(1)
        if K.dim() == 3:
            K = K.unsqueeze(1)
        if q_T.dim() == 3:
            q_T = q_T.unsqueeze(1)

        # Compute mobilities
        lambda_w, lambda_n, lambda_T = self._compute_mobilities(S_w)

        # Compute effective saturation for capillary pressure
        S_e = self.brooks_corey.effective_saturation(S_w)

        # Compute capillary pressure
        P_c = self.brooks_corey.capillary_pressure(S_e)

        # Term 1: -∇·(λ_T K ∇p_w)
        # Compute pressure gradient
        dp_dx = grad_x(p_w, self.dx)
        dp_dy = grad_y(p_w, self.dy)

        # Compute flux: -λ_T K ∇p_w
        flux_x_1 = -lambda_T * K * dp_dx
        flux_y_1 = -lambda_T * K * dp_dy

        # Compute divergence of flux
        div_term_1 = divergence(flux_x_1, flux_y_1, self.dx, self.dy)

        # Term 2: ∇·(λ_n K ∇P_c(S_w))
        # Compute capillary pressure gradient
        dPc_dx = grad_x(P_c, self.dx)
        dPc_dy = grad_y(P_c, self.dy)

        # Compute capillary flux: λ_n K ∇P_c
        flux_x_2 = lambda_n * K * dPc_dx
        flux_y_2 = lambda_n * K * dPc_dy

        # Compute divergence of capillary flux
        div_term_2 = divergence(flux_x_2, flux_y_2, self.dx, self.dy)

        # Compute residual: -∇·(λ_T K ∇p_w) + ∇·(λ_n K ∇P_c) - q_T
        # Note: div_term_1 already has the negative sign from the flux definition
        residual = -div_term_1 + div_term_2 - q_T

        return residual

    def saturation_residual(
        self,
        S_w_pred: torch.Tensor,
        S_w_true: torch.Tensor,
        p_w: torch.Tensor,
        K: torch.Tensor,
        phi: torch.Tensor,
        q_w: torch.Tensor,
        dt: float
    ) -> torch.Tensor:
        """R_2: φ·∂S_w/∂t + ∇·(f_w · v_T) = q_w."""
        # Ensure 4D tensors for consistent processing
        if S_w_pred.dim() == 3:
            S_w_pred = S_w_pred.unsqueeze(1)
        if S_w_true.dim() == 3:
            S_w_true = S_w_true.unsqueeze(1)
        if p_w.dim() == 3:
            p_w = p_w.unsqueeze(1)
        if K.dim() == 3:
            K = K.unsqueeze(1)
        if phi.dim() == 3:
            phi = phi.unsqueeze(1)
        if q_w.dim() == 3:
            q_w = q_w.unsqueeze(1)

        # Compute time derivative: ∂S_w/∂t ≈ (S_w_pred - S_w_true) / dt
        dSw_dt = (S_w_pred - S_w_true) / (dt + 1e-8)

        # Compute mobilities using predicted saturation
        _, _, lambda_T = self._compute_mobilities(S_w_pred)

        # Compute fractional flow
        f_w = self.fractional_flow(S_w_pred)

        # Compute pressure gradient
        dp_dx = grad_x(p_w, self.dx)
        dp_dy = grad_y(p_w, self.dy)

        # Compute total Darcy velocity: v_T = -λ_T K ∇p_w
        v_T_x = -lambda_T * K * dp_dx
        v_T_y = -lambda_T * K * dp_dy

        # Compute water flux: f_w · v_T
        flux_w_x = f_w * v_T_x
        flux_w_y = f_w * v_T_y

        # Compute divergence of water flux
        div_flux_w = divergence(flux_w_x, flux_w_y, self.dx, self.dy)

        # Compute residual: φ·∂S_w/∂t + ∇·(f_w · v_T) - q_w
        residual = phi * dSw_dt + div_flux_w - q_w

        return residual