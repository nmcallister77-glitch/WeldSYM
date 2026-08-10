"""2D transient heat conduction with moving heat source (finite differences)."""

from __future__ import annotations

import numpy as np

from ..types import WeldParams, MaterialParams


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run a 2D transient heat conduction simulation on a regular grid.

    Parameters
    ----------
    h : float
        Effective thickness over which the surface heat flux is distributed (m).

    Returns
    -------
    x : np.ndarray
        1D array of x coordinates (m), length nx.
    y : np.ndarray
        1D array of y coordinates (m), length ny.
    T : np.ndarray
        Temperature field at final time, shape (nx, ny).
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
                Q[i, j] = heat_source_at_point(x[i], y[j], t, weld, h)

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

    return x, y, T


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

    Returns
    -------
    q_vol : float
        Volumetric heat source (W/m^3).
    """
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
