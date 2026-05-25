#!/usr/bin/env python
"""
Generate compositional flow data for CO2+brine system.

Implements a simplified 2-component compositional model:
- Flash calculation for CO2 dissolution in brine
- Pressure equation (elliptic)
- Saturation transport (hyperbolic)
- Composition transport (CO2 mole fraction)

Output: .pt files with shape (N, 3, 64, 64) per timestep
  Channel 0: pressure
  Channel 1: water saturation (Sw)
  Channel 2: CO2 mole fraction in liquid phase (x_CO2)
"""

import os
import argparse
import numpy as np
import torch
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter


def generate_permeability_field(N: int, correlation_length: float = 0.1,
                                 variance: float = 1.0, rng=None) -> np.ndarray:
    """Generate log-normal permeability with spatial correlation."""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.standard_normal((N, N))
    sigma = correlation_length * N
    smoothed = gaussian_filter(noise, sigma=sigma, mode="wrap")
    smoothed = smoothed / (smoothed.std() + 1e-8) * np.sqrt(variance)
    return np.exp(smoothed)


def flash_calculation(x_CO2: np.ndarray, pressure: np.ndarray,
                      temperature: float = 323.15) -> dict:
    """Simplified flash calculation for CO2-brine system.

    Uses Henry's law approximation for CO2 solubility:
        x_CO2_eq = H * p_CO2 / P_total

    where H is Henry's constant (temperature-dependent).

    Args:
        x_CO2: Overall CO2 mole fraction in liquid phase (N, N).
        pressure: Pressure field in Pa (N, N).
        temperature: Temperature in K (isothermal assumption).

    Returns:
        Dictionary with phase properties:
        - 'Sg': Gas saturation (CO2-rich phase)
        - 'Sw': Water saturation (brine phase)
        - 'x_CO2_liq': CO2 mole fraction in liquid
        - 'y_CO2_gas': CO2 mole fraction in gas (≈1 for simplified model)
        - 'rho_g': Gas phase density
        - 'rho_w': Water phase density
        - 'mu_g': Gas viscosity
        - 'mu_w': Water viscosity
    """
    # Henry's constant for CO2 in water (simplified, Pa)
    # At ~50°C, H ≈ 3000 MPa
    H_CO2 = 3.0e9  # Pa

    # Equilibrium CO2 solubility at given pressure
    # x_CO2_eq = P / H_CO2 (simplified Henry's law)
    p_ref = 1.0e7  # 10 MPa reference pressure
    x_CO2_eq = np.clip(pressure / H_CO2, 0.0, 0.05)  # max 5% solubility

    # If x_CO2 > x_CO2_eq, excess forms gas phase
    excess_CO2 = np.maximum(x_CO2 - x_CO2_eq, 0.0)

    # Gas saturation from excess CO2 (simplified volume balance)
    # Sg ≈ excess_CO2 * V_mol_liq / V_mol_gas
    V_ratio = 0.01  # molar volume ratio (liquid/gas) at reservoir conditions
    Sg = np.clip(excess_CO2 / (excess_CO2 + V_ratio + 1e-10), 0.0, 0.8)
    Sw = 1.0 - Sg

    # Phase properties (simplified)
    rho_w = 1000.0  # kg/m³
    rho_g = 500.0 + 50.0 * pressure / p_ref  # density increases with pressure
    mu_w = 0.5e-3  # Pa·s
    mu_g = 0.05e-3  # Pa·s

    return {
        'Sg': Sg,
        'Sw': Sw,
        'x_CO2_liq': np.minimum(x_CO2, x_CO2_eq),
        'y_CO2_gas': np.ones_like(x_CO2) * 0.99,
        'rho_g': rho_g,
        'rho_w': rho_w,
        'mu_g': mu_g,
        'mu_w': mu_w,
    }


def solve_pressure_compositional(K: np.ndarray, Sw: np.ndarray, Sg: np.ndarray,
                                  mu_w: float, mu_g: float,
                                  q_T: np.ndarray = None, N: int = 64) -> np.ndarray:
    """Solve pressure equation for compositional system.

    -∇·(λ_T K ∇p) = q_T
    where λ_T = kr_w/μ_w + kr_g/μ_g
    """
    h = 1.0 / (N - 1)

    # Relative permeabilities (quadratic)
    S_w_eff = np.clip(Sw, 0.01, 0.99)
    kr_w = S_w_eff ** 2
    kr_g = (1.0 - S_w_eff) ** 2

    lam_w = kr_w / mu_w
    lam_g = kr_g / mu_g
    lam_T = lam_w + lam_g

    T = lam_T * K

    # Build sparse system
    n_int = (N - 2) ** 2
    rows, cols, vals = [], [], []
    rhs = np.zeros(n_int)

    if q_T is None:
        q_T = np.zeros((N, N))
        q_T[N // 4, N // 4] = 1.0
        q_T[3 * N // 4, 3 * N // 4] = -1.0

    def idx(i, j):
        return (i - 1) * (N - 2) + (j - 1)

    for i in range(1, N - 1):
        for j in range(1, N - 1):
            k = idx(i, j)
            center = 0.0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                t_face = 2.0 * T[i, j] * T[ni, nj] / (T[i, j] + T[ni, nj] + 1e-12)
                coeff = t_face / h ** 2
                center -= coeff
                if 1 <= ni <= N - 2 and 1 <= nj <= N - 2:
                    rows.append(k)
                    cols.append(idx(ni, nj))
                    vals.append(coeff)
            rows.append(k)
            cols.append(k)
            vals.append(center)
            rhs[k] = q_T[i, j]

    A = sparse.csr_matrix((vals, (rows, cols)), shape=(n_int, n_int))
    p_int = spsolve(A, rhs)

    p = np.zeros((N, N))
    p[1:-1, 1:-1] = p_int.reshape(N - 2, N - 2)
    return p


def update_saturation_composition(
    Sw: np.ndarray, x_CO2: np.ndarray, p: np.ndarray,
    K: np.ndarray, phi: np.ndarray,
    mu_w: float, mu_g: float,
    dt: float = 0.002, N: int = 64,
    q_CO2: np.ndarray = None,
) -> tuple:
    """Update saturation and composition using explicit transport.

    Solves:
    - φ ∂Sw/∂t + ∇·(fw · v_T) = qw
    - φ ∂(Sw·x_CO2)/∂t + ∇·(x_CO2 · fw · v_T) = q_CO2

    Args:
        Sw: Water saturation (N, N).
        x_CO2: CO2 mole fraction in liquid (N, N).
        p: Pressure field (N, N).
        K: Permeability (N, N).
        phi: Porosity (N, N).
        mu_w: Water viscosity.
        mu_g: Gas viscosity.
        dt: Time step.
        N: Grid resolution.
        q_CO2: CO2 injection source term.

    Returns:
        (Sw_new, x_CO2_new): Updated fields.
    """
    h = 1.0 / (N - 1)

    # Mobilities
    S_w_eff = np.clip(Sw, 0.01, 0.99)
    kr_w = S_w_eff ** 2
    kr_g = (1.0 - S_w_eff) ** 2
    lam_w = kr_w / mu_w
    lam_g = kr_g / mu_g
    lam_T = lam_w + lam_g + 1e-12
    f_w = lam_w / lam_T

    # Darcy velocity
    dp_dx = np.zeros_like(p)
    dp_dy = np.zeros_like(p)
    dp_dx[:, 1:-1] = (p[:, 2:] - p[:, :-2]) / (2 * h)
    dp_dy[1:-1, :] = (p[2:, :] - p[:-2, :]) / (2 * h)

    v_x = -lam_T * K * dp_dx
    v_y = -lam_T * K * dp_dy

    # Saturation transport
    flux_x = f_w * v_x
    flux_y = f_w * v_y

    div_flux = np.zeros_like(Sw)
    div_flux[:, 1:-1] += (flux_x[:, 2:] - flux_x[:, :-2]) / (2 * h)
    div_flux[1:-1, :] += (flux_y[2:, :] - flux_y[:-2, :]) / (2 * h)

    Sw_new = Sw - dt / (phi + 1e-8) * div_flux

    # Composition transport (CO2 advected with water phase)
    co2_flux_x = x_CO2 * f_w * v_x
    co2_flux_y = x_CO2 * f_w * v_y

    div_co2_flux = np.zeros_like(x_CO2)
    div_co2_flux[:, 1:-1] += (co2_flux_x[:, 2:] - co2_flux_x[:, :-2]) / (2 * h)
    div_co2_flux[1:-1, :] += (co2_flux_y[2:, :] - co2_flux_y[:-2, :]) / (2 * h)

    # Source term for CO2 injection
    if q_CO2 is None:
        q_CO2 = np.zeros((N, N))
        q_CO2[N // 4, N // 4] = 0.01  # CO2 injection

    x_CO2_new = x_CO2 - dt / (phi * Sw_new.clip(0.01) + 1e-8) * div_co2_flux
    x_CO2_new += dt / (phi * Sw_new.clip(0.01) + 1e-8) * q_CO2

    # Clip to physical bounds
    Sw_new = np.clip(Sw_new, 0.0, 1.0)
    x_CO2_new = np.clip(x_CO2_new, 0.0, 0.1)  # max 10% CO2 in liquid

    return Sw_new, x_CO2_new


def generate_compositional_trajectory(
    K: np.ndarray, phi: np.ndarray,
    n_steps: int = 10, N: int = 64,
) -> np.ndarray:
    """Generate a compositional flow trajectory.

    Returns:
        Array of shape (n_steps, 3, N, N) with channels [pressure, Sw, x_CO2].
    """
    # Initial conditions
    Sw = np.ones((N, N)) * 0.95  # mostly brine initially
    x_CO2 = np.zeros((N, N))  # no CO2 initially
    p_ref = 1.0e7  # 10 MPa

    mu_w = 0.5e-3
    mu_g = 0.05e-3

    trajectory = []

    for t in range(n_steps):
        # Flash calculation to get phase split
        flash = flash_calculation(x_CO2, np.ones_like(Sw) * p_ref)
        Sg = flash['Sg']

        # Solve pressure
        p = solve_pressure_compositional(K, Sw, Sg, mu_w, mu_g, N=N)

        # Store state: [pressure, Sw, x_CO2]
        state = np.stack([p, Sw, x_CO2], axis=0)  # (3, N, N)
        trajectory.append(state)

        # Update saturation and composition
        Sw, x_CO2 = update_saturation_composition(
            Sw, x_CO2, p, K, phi, mu_w, mu_g, N=N
        )

    return np.stack(trajectory, axis=0)  # (n_steps, 3, N, N)


def generate_dataset(n_samples: int = 200, resolution: int = 64,
                     n_steps: int = 10, seed: int = 42) -> dict:
    """Generate full compositional dataset.

    Returns:
        Dictionary with:
        - 'trajectories': (n_samples, n_steps, 3, resolution, resolution)
        - 'permeability': (n_samples, resolution, resolution)
        - 'porosity': (n_samples, resolution, resolution)
    """
    rng = np.random.default_rng(seed)

    all_trajectories = []
    all_perm = []
    all_porosity = []

    for i in range(n_samples):
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Generating sample {i+1}/{n_samples}")

        K = generate_permeability_field(resolution, rng=rng)
        phi = 0.15 + 0.1 * rng.random((resolution, resolution))

        traj = generate_compositional_trajectory(K, phi, n_steps, resolution)

        all_trajectories.append(traj)
        all_perm.append(K)
        all_porosity.append(phi)

    return {
        'trajectories': np.stack(all_trajectories).astype(np.float32),
        'permeability': np.stack(all_perm).astype(np.float32),
        'porosity': np.stack(all_porosity).astype(np.float32),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate compositional CO2+brine flow data"
    )
    parser.add_argument("--n-train", type=int, default=200,
                        help="Number of training samples")
    parser.add_argument("--n-test", type=int, default=50,
                        help="Number of test samples")
    parser.add_argument("--resolution", type=int, default=64,
                        help="Grid resolution")
    parser.add_argument("--n-steps", type=int, default=10,
                        help="Number of timesteps per trajectory")
    parser.add_argument("--output-dir", type=str, default="data/compositional",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Generating compositional CO2+brine training data...")
    print(f"  Resolution: {args.resolution}x{args.resolution}")
    print(f"  Timesteps: {args.n_steps}")
    print(f"  Output shape per sample: ({args.n_steps}, 3, {args.resolution}, {args.resolution})")

    train_data = generate_dataset(args.n_train, args.resolution, args.n_steps, args.seed)
    train_path = os.path.join(args.output_dir, "compositional_train.pt")
    torch.save(train_data, train_path)
    print(f"  Saved training data: {train_path}")
    print(f"    trajectories: {train_data['trajectories'].shape}")

    print("Generating test data...")
    test_data = generate_dataset(args.n_test, args.resolution, args.n_steps, args.seed + 100)
    test_path = os.path.join(args.output_dir, "compositional_test.pt")
    torch.save(test_data, test_path)
    print(f"  Saved test data: {test_path}")
    print(f"    trajectories: {test_data['trajectories'].shape}")

    print("Done.")


if __name__ == "__main__":
    main()
