"""2D transient heat conduction with moving heat source (finite differences)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..types import WeldParams, MaterialParams
from ..weld_path import WeldPath, WobbleParams


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
    h: float = 0.005,  # effective thickness (m)
    path: Optional[WeldPath] = None,
    wobble: Optional[WobbleParams] = None,
    probe: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Run a 2D transient heat conduction simulation on a regular grid.

    Parameters
    ----------
    h : float
        Effective thickness over which the surface heat flux is distributed (m).
    path : WeldPath | None
        Optional weld path. If given, overrides weld.direction-based motion.
    wobble : WobbleParams | None
        Optional laser wobble. Requires ``path``.

    Returns
    -------
    x : np.ndarray
        1D array of x coordinates (m), length nx.
    y : np.ndarray
        1D array of y coordinates (m), length ny.
    T : np.ndarray
        Temperature field at final time, shape (nx, ny).
    T_probe : np.ndarray | None
        Time-temperature history at ``probe`` (s, K) if requested.
    """
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
        raise ValueError(
            f"Unstable: r_x + r_y = {r_x + r_y:.3f} > 0.5. "
            "Reduce dt or refine mesh."
        )

    # Time stepping
    n_steps = int(np.ceil(t_end / dt))

    # Initialize temperature
    T = np.full((nx, ny), T0)
    T_new = T.copy()

    # Probe for time-temperature history
    T_probe = None
    if probe is not None:
        px, py = probe
        ix = min(max(int(round(px / dx)), 0), nx - 1)
        iy = min(max(int(round(py / dy)), 0), ny - 1)
        T_probe = np.zeros(n_steps)

    # Meshgrid for vectorised source term
    X, Y = np.meshgrid(x, y, indexing="ij")
    q_eff = weld.power * weld.efficiency
    q_denom = 2.0 * np.pi * weld.sigma**2
    h_eff = h

    def _beam_position(t: float) -> tuple[float, float]:
        if path is not None:
            from ..weld_path import beam_at_time

            w = wobble or WobbleParams(amplitude=0.0, frequency=0.0)
            return beam_at_time(path, w, t)
        return weld_position_at_time(weld, t)

    for step in range(n_steps):
        t = step * dt

        x_src, y_src = _beam_position(t)
        r2 = (X - x_src) ** 2 + (Y - y_src) ** 2
        Q = (q_eff / q_denom) * np.exp(-r2 / (2.0 * weld.sigma**2)) / h_eff

        # Vectorised explicit update (interior points only)
        lap = (
            (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (dx**2)
            + (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]) / (dy**2)
        )
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * dt * lap + (dt / (rho * cp)) * Q[1:-1, 1:-1]

        # Boundary conditions: T = T0
        T_new[0, :] = T0
        T_new[-1, :] = T0
        T_new[:, 0] = T0
        T_new[:, -1] = T0

        if T_probe is not None:
            T_probe[step] = T_new[ix, iy]

        T, T_new = T_new, T  # swap

    return x, y, T, T_probe


def weld_position_at_time(weld: WeldParams, t: float) -> tuple[float, float]:
    """Return the (x, y) position of the moving heat source at time t."""
    if weld.direction == "x":
        x_src = weld.start_pos[0] + weld.speed * t
        y_src = weld.start_pos[1]
    elif weld.direction == "y":
        x_src = weld.start_pos[0]
        y_src = weld.start_pos[1] + weld.speed * t
    else:
        raise ValueError("direction must be 'x' or 'y'")
    return x_src, y_src


def heat_source_at_point(
    x: float,
    y: float,
    t: float,
    weld: WeldParams,
    h: float,
    path: Optional[WeldPath] = None,
    wobble: Optional[WobbleParams] = None,
) -> float:
    """
    Evaluate heat source (W/m^3) at a single (x, y, t) point.

    Parameters
    ----------
    x, y : float
        Spatial coordinates (m).
    t : float
        Time (s).
    weld : WeldParams
        Welding process parameters (power, efficiency, speed, etc.).
    h : float
        Effective plate thickness (m).
    path : WeldPath | None
        Optional path for arbitrary weld trajectory.
    wobble : WobbleParams | None
        Optional wobble around the path.

    Returns
    -------
    q_vol : float
        Volumetric heat source (W/m^3).
    """
    if path is not None:
        from ..weld_path import heat_source_at_point as _path_heat

        w = wobble or WobbleParams(amplitude=0.0, frequency=0.0)
        return _path_heat(x, y, t, path, w, weld.power, weld.efficiency, weld.sigma, h)

    # Position of the moving heat source along the weld line
    x_src, y_src = weld_position_at_time(weld, t)

    dx = x - x_src
    dy = y - y_src
    r2 = dx**2 + dy**2

    q_eff = weld.power * weld.efficiency
    sigma = weld.sigma

    # 2D Gaussian heat flux [W/m^2]
    q_surf = (q_eff / (2.0 * np.pi * sigma**2)) * np.exp(-r2 / (2.0 * sigma**2))

    # Treat as surface heat flux spread over thickness h → volumetric [W/m^3]
    q_vol = q_surf / h

    return q_vol
