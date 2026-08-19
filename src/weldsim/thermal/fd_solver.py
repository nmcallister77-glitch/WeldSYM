"""2D transient heat conduction with moving heat source (finite differences)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

from ..errors import StabilityError, ValidationError
from ..types import WeldParams, MaterialParams
from ..weld_path import WeldPath, WobbleParams

#: Cooling-rate interval used for weldability assessment: 800 C down to 500 C.
T85_UPPER = 1073.15
T85_LOWER = 773.15

#: Stefan-Boltzmann constant (W/m^2 K^4).
SIGMA_SB = 5.670374419e-8


@dataclass
class PhaseModel:
    """Phase-change and surface-loss physics for the thin-plate solve.

    Pure conduction with constant properties has no way to stop: a concentrated
    beam drives the peak temperature far past anything metallurgically possible,
    which is why an unmodified run reports melt pools several times too wide.
    Three effects hold a real weld pool in check and are represented here:

    latent heat of fusion
        Absorbed as the material passes through the mushy range, modelled as an
        apparent heat capacity over ``solidus``..``liquidus``.
    evaporation
        Above the boiling point the surface vaporises and the plume carries the
        excess energy away, so temperature is capped there.
    surface losses
        Convection and radiation from both faces of the plate.

    Set ``latent_heat``, ``boiling`` or the loss coefficients to zero/None to
    disable the corresponding term.
    """

    solidus: float
    liquidus: float
    latent_heat: float = 0.0  # J/kg
    boiling: float | None = None  # K; temperature is capped here
    emissivity: float = 0.7
    convection_coefficient: float = 15.0  # W/m^2 K, per exposed face

    def apparent_capacity(self, T: np.ndarray, cp: float) -> np.ndarray:
        """Specific heat with the latent heat of fusion smeared over the mushy range."""
        if self.latent_heat <= 0 or self.liquidus <= self.solidus:
            return np.full_like(T, cp)
        mushy = (T >= self.solidus) & (T <= self.liquidus)
        cp_field = np.full_like(T, cp)
        cp_field[mushy] += self.latent_heat / (self.liquidus - self.solidus)
        return cp_field

    def surface_loss(self, T: np.ndarray, T0: float, thickness: float) -> np.ndarray:
        """Volumetric heat sink from both plate faces (W/m^3)."""
        if thickness <= 0:
            return np.zeros_like(T)
        flux = self.convection_coefficient * (T - T0)
        if self.emissivity > 0:
            flux = flux + self.emissivity * SIGMA_SB * (T**4 - T0**4)
        return 2.0 * flux / thickness


@dataclass
class ThermalHistory:
    """Per-cell thermal-cycle quantities accumulated during the transient run.

    Every field has the shape of the temperature grid, ``(nx, ny)``. These are
    what the weld-quality metrics are derived from: the final temperature field
    alone says nothing about the weld, because the fusion zone and the heat
    affected zone are defined by the *peak* temperature each point reached and
    by how fast it cooled afterwards.
    """

    T_peak: np.ndarray
    """Highest temperature reached at each cell (K)."""

    t_peak: np.ndarray
    """Time at which the peak occurred (s)."""

    t_8_5: np.ndarray
    """Time to cool from ``T85_UPPER`` to ``T85_LOWER`` (s); NaN where the cell
    never completed that interval."""

    cooling_rate: np.ndarray
    """Instantaneous cooling rate while passing ``T85_UPPER`` (K/s, positive);
    NaN where the cell never cooled through it."""

    dwell_above: np.ndarray
    """Time spent above the ``dwell_temp`` threshold (s)."""

    dwell_temp: float
    """Threshold used for :attr:`dwell_above` (K)."""

    extra_dwell: dict[float, np.ndarray] = field(default_factory=dict)
    """Time above further thresholds (s), keyed by threshold temperature (K).

    Dwell is threshold-specific and cannot be rescaled after the fact, so a
    consumer that needs the time above, say, the grain-coarsening temperature
    has to ask the solver to accumulate it during the run."""

    def time_above(self, threshold: float) -> np.ndarray | None:
        """Dwell field for ``threshold`` (K), or ``None`` if it was not tracked."""
        if math.isclose(threshold, self.dwell_temp):
            return self.dwell_above
        for tracked, dwell in self.extra_dwell.items():
            if math.isclose(threshold, tracked):
                return dwell
        return None


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
    dwell_temp: float | None = None,
    extra_dwell_temps: Sequence[float] = (),
    phase: Optional[PhaseModel] = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, ThermalHistory]:
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
    dwell_temp : float | None
        Temperature above which dwell time is accumulated (K), normally the
        solidus so that the melt-pool residence time is available. Defaults to
        ``T85_UPPER``.
    extra_dwell_temps : sequence of float
        Further thresholds (K) to accumulate dwell above, reported in
        ``ThermalHistory.extra_dwell``.
    phase : PhaseModel | None
        Latent heat, evaporation cap and surface losses. Omit for the plain
        constant-property conduction solve.

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
    history : ThermalHistory
        Peak temperature, cooling rate and dwell fields over the whole grid.
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
        dt_max = 0.5 / (alpha * (1.0 / dx**2 + 1.0 / dy**2))
        raise StabilityError(
            f"Unstable time step: r_x + r_y = {r_x + r_y:.3f} exceeds the explicit "
            f"limit of 0.5. Use dt <= {dt_max:.4g} s, or coarsen the mesh."
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

    # Thermal-cycle bookkeeping, used by the weld-quality metrics
    dwell_threshold = T85_UPPER if dwell_temp is None else dwell_temp
    T_peak = np.full((nx, ny), T0)
    t_peak = np.zeros((nx, ny))
    t_cross_hi = np.full((nx, ny), np.nan)
    t_cross_lo = np.full((nx, ny), np.nan)
    cooling_rate = np.full((nx, ny), np.nan)
    dwell_above = np.zeros((nx, ny))
    extra_dwell = {
        float(temp): np.zeros((nx, ny))
        for temp in extra_dwell_temps
        if not math.isclose(float(temp), dwell_threshold)
    }

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

        if path is None or t < path.duration:
            x_src, y_src = _beam_position(t)
            r2 = (X - x_src) ** 2 + (Y - y_src) ** 2
            Q = (q_eff / q_denom) * np.exp(-r2 / (2.0 * weld.sigma**2)) / h_eff
        else:
            Q = np.zeros_like(X)

        # Vectorised explicit update (interior points only)
        lap = (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / (dx**2) + (
            T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]
        ) / (dy**2)
        source = Q[1:-1, 1:-1]
        if phase is None:
            rho_cp = rho * cp
        else:
            rho_cp = rho * phase.apparent_capacity(T[1:-1, 1:-1], cp)
            source = source - phase.surface_loss(T[1:-1, 1:-1], T0, h_eff)
        T_new[1:-1, 1:-1] = T[1:-1, 1:-1] + (dt / rho_cp) * (k * lap + source)

        if phase is not None and phase.boiling is not None:
            # Energy driving the surface past boiling leaves with the vapour
            # plume instead of heating the plate further.
            np.minimum(T_new, phase.boiling, out=T_new)

        # Boundary conditions: T = T0
        T_new[0, :] = T0
        T_new[-1, :] = T0
        T_new[:, 0] = T0
        T_new[:, -1] = T0

        if T_probe is not None:
            T_probe[step] = T_new[ix, iy]

        t_next = t + dt
        hotter = T_new > T_peak
        T_peak[hotter] = T_new[hotter]
        t_peak[hotter] = t_next
        dwell_above[T_new > dwell_threshold] += dt
        for threshold, dwell in extra_dwell.items():
            dwell[T_new > threshold] += dt

        # Downward crossings of the 800 C / 500 C levels, linearly interpolated
        # within the step so the result does not depend on dt as strongly.
        for level, crossings in ((T85_UPPER, t_cross_hi), (T85_LOWER, t_cross_lo)):
            crossed = (T >= level) & (T_new < level) & np.isnan(crossings)
            if crossed.any():
                span = T[crossed] - T_new[crossed]
                frac = (T[crossed] - level) / np.where(span > 0, span, 1.0)
                crossings[crossed] = t + frac * dt
                if level == T85_UPPER:
                    cooling_rate[crossed] = span / dt

        T, T_new = T_new, T  # swap

    history = ThermalHistory(
        T_peak=T_peak,
        t_peak=t_peak,
        t_8_5=t_cross_lo - t_cross_hi,
        cooling_rate=cooling_rate,
        dwell_above=dwell_above,
        dwell_temp=dwell_threshold,
        extra_dwell=extra_dwell,
    )
    return x, y, T, T_probe, history


def weld_position_at_time(weld: WeldParams, t: float) -> tuple[float, float]:
    """Return the (x, y) position of the moving heat source at time t."""
    if weld.direction == "x":
        x_src = weld.start_pos[0] + weld.speed * t
        y_src = weld.start_pos[1]
    elif weld.direction == "y":
        x_src = weld.start_pos[0]
        y_src = weld.start_pos[1] + weld.speed * t
    else:
        raise ValidationError(f"Weld direction must be 'x' or 'y', got {weld.direction!r}.")
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
