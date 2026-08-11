"""2D transient heat conduction with moving heat source (finite differences)."""

from __future__ import annotations

import numpy as np

from ..exceptions import ConfigurationError, SolverError
from ..types import WeldParams, MaterialParams

_VALID_DIRECTIONS = ("x", "y")


def _validate_inputs(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    t_end: float,
    dt: float,
    weld: WeldParams,
    material: MaterialParams,
) -> None:
    """Raise :class:`ConfigurationError` for inputs the solver cannot handle."""
    if nx < 3 or ny < 3:
        raise ConfigurationError(f"nx and ny must be >= 3 (got nx={nx}, ny={ny}).")
    if Lx <= 0 or Ly <= 0:
        raise ConfigurationError(f"Lx and Ly must be positive (got Lx={Lx}, Ly={Ly}).")
    if dt <= 0:
        raise ConfigurationError(f"dt must be positive (got {dt}).")
    if t_end <= 0:
        raise ConfigurationError(f"t_end must be positive (got {t_end}).")
    if dt > t_end:
        raise ConfigurationError(f"dt ({dt}) must not exceed t_end ({t_end}).")
    if material.k <= 0 or material.rho <= 0 or material.cp <= 0:
        raise ConfigurationError(
            "Material k, rho and cp must be positive "
            f"(got k={material.k}, rho={material.rho}, cp={material.cp})."
        )
    if weld.direction not in _VALID_DIRECTIONS:
        raise ConfigurationError(
            f"weld.direction must be one of {_VALID_DIRECTIONS} (got {weld.direction!r})."
        )
    if weld.sigma <= 0:
        raise ConfigurationError(f"weld.sigma must be positive (got {weld.sigma}).")
    if not np.isfinite(weld.speed):
        raise ConfigurationError(f"weld.speed must be finite (got {weld.speed}).")
    if weld.power < 0:
        raise ConfigurationError(f"weld.power must not be negative (got {weld.power}).")
    if not 0 < weld.efficiency <= 1:
        raise ConfigurationError(f"weld.efficiency must be in (0, 1] (got {weld.efficiency}).")
    if len(weld.start_pos) != 2:
        raise ConfigurationError(f"weld.start_pos must be an (x, y) pair (got {weld.start_pos!r}).")


def run_2d_fd_thermal(
    nx: int,
    ny: int,
    Lx: float,
    Ly: float,
    t_end: float,
    dt: float,
    weld: WeldParams,
    material: MaterialParams,
    T0: float = 300.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a 2D transient heat conduction simulation on a regular grid.

    Returns
    -------
    x : np.ndarray
        1D array of x coordinates (m), length nx.
    y : np.ndarray
        1D array of y coordinates (m), length ny.
    T : np.ndarray
        Temperature field at final time, shape (nx, ny).
    """
    _validate_inputs(nx, ny, Lx, Ly, t_end, dt, weld, material)

    # Grid
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)

    # Material
    k = material.k
    rho = material.rho
    cp = material.cp
    alpha = k / (rho * cp)

    # Stability check (explicit scheme)
    r_x = alpha * dt / (dx**2)
    r_y = alpha * dt / (dy**2)
    if r_x + r_y > 0.5:
        raise ConfigurationError(
            f"Unstable: r_x + r_y = {r_x + r_y:.3f} > 0.5. Reduce dt or refine mesh."
        )

    # Initialize temperature
    T = np.full((nx, ny), T0)
    T_new = T.copy()

    # Time stepping
    n_steps = int(np.ceil(t_end / dt))
    for step in range(n_steps):
        t = step * dt

        # Compute heat source term Q(x, y, t) on the grid
        Q = np.zeros_like(T)
        for i in range(nx):
            for j in range(ny):
                Q[i, j] = heat_source_at_point(x[i], y[j], t, weld)

        # Explicit update (interior points only)
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                lap = (
                    (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / (dx**2)
                    + (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / (dy**2)
                )
                T_new[i, j] = T[i, j] + alpha * dt * lap + (dt / (rho * cp)) * Q[i, j]

        # Boundary conditions: T = T0
        T_new[0, :] = T0
        T_new[-1, :] = T0
        T_new[:, 0] = T0
        T_new[:, -1] = T0

        T, T_new = T_new, T  # swap

        if not np.all(np.isfinite(T)):
            raise SolverError(
                f"Solution diverged at step {step + 1}/{n_steps} (t = {t + dt:.4g} s): "
                "temperature field contains NaN or infinite values. "
                "Reduce dt, refine the mesh, or check the material properties."
            )

    return x, y, T


def heat_source_at_point(
    x: float,
    y: float,
    t: float,
    weld: WeldParams,
) -> float:
    """
    Evaluate heat source (W/m^3) at a single (x, y, t) point.
    Uses a simple 2D Gaussian in the plane; z is assumed 0.
    """
    if weld.direction == "x":
        x_src = weld.start_pos[0] + weld.speed * t
        y_src = weld.start_pos[1]
    elif weld.direction == "y":
        x_src = weld.start_pos[0]
        y_src = weld.start_pos[1] + weld.speed * t
    else:
        raise ConfigurationError(
            f"weld.direction must be one of {_VALID_DIRECTIONS} (got {weld.direction!r})."
        )

    if weld.sigma <= 0:
        raise ConfigurationError(f"weld.sigma must be positive (got {weld.sigma}).")

    dx = x - x_src
    dy = y - y_src
    r2 = dx**2 + dy**2

    # Treat as surface heat flux spread over a small thickness h
    h = 0.001  # m, artificial thickness
    q_eff = weld.power * weld.efficiency
    sigma = weld.sigma

    q = (q_eff / (2 * np.pi * sigma**2 * h)) * np.exp(-r2 / (2 * sigma**2))
    return q
