"""3D transient conduction with a moving keyhole heat source.

This is the in-package alternative to the OpenFOAM case: it resolves depth, so
it answers the questions a 2D thin-plate solve cannot — how deep the weld goes,
what the transverse cross-section looks like, and whether the joint is fully
penetrated. It needs nothing but NumPy, so it runs anywhere the GUI runs, with
no external solver, no compilation and no network access.

What it models
--------------
* Transient heat conduction on a regular 3D grid (explicit, vectorised).
* A moving heat source split between a surface Gaussian and a conical
  volumetric term, the split set by how far the absorbed intensity exceeds the
  keyhole threshold. This is the standard engineering representation of a laser
  weld: the beam is delivered down the capillary rather than on the surface,
  which is what produces the deep narrow fusion zone a surface source cannot.
* Latent heat of fusion, an evaporation temperature cap, and convection plus
  radiation from the top and bottom faces.
* Beam wobble, sub-sampled within each time step so a 200 Hz oscillation is
  represented as the time-averaged track it really is.

What it does not model
----------------------
Free-surface motion, recoil pressure, vapour dynamics, Marangoni convection or
fluid flow in the pool. The keyhole here is an assumed shape that carries the
beam energy, not a computed one. For those, use the OpenFOAM case, which solves
the VOF free surface — the trade is that it needs a Linux/WSL2 install.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence

import numpy as np

from ..errors import AbortError, StabilityError, ValidationError
from ..keyhole import KEYHOLE_THRESHOLD_INTENSITY, TRANSITION_BAND, peak_intensity
from ..materials import Material
from ..types import WeldParams
from ..weld_metrics import level_extent
from ..weld_path import WeldPath, WobbleParams, beam_at_time
from .fd_solver import (
    SIGMA_SB,
    T85_LOWER,
    T85_UPPER,
    PhaseModel,
    ThermalHistory,
    weld_position_at_time,
)
from ._numba_step import HAS_NUMBA, numba_step

#: Fraction of the explicit stability limit used when the time step is chosen
#: automatically. Latent heat and the evaporation cap both add stiffness, so a
#: little margin below the theoretical limit is worth having.
DT_SAFETY = 0.4

#: Radius of the keyhole channel at the surface, as a multiple of the beam
#: sigma, and the taper to its tip. A real capillary is roughly the beam
#: diameter at the top and closes towards the root.
KEYHOLE_RADIUS_FACTOR = 1.5
KEYHOLE_TIP_TAPER = 0.4

#: Sub-samples of the beam position per time step, per wobble period, so a
#: high-frequency oscillation is not aliased into a stationary spot.
WOBBLE_SUBSAMPLES_PER_PERIOD = 8
MAX_WOBBLE_SUBSAMPLES = 16

#: Ceiling on cells x time steps. Raised to allow overnight/multi-day runs while
#: still catching accidental huge cases before they allocate or loop.
MAX_CELL_UPDATES = 1_000_000_000_000_000


@dataclass
class SectionMetrics:
    """Weld geometry measured on the transverse cross-section."""

    station: float  # m, x position of the section
    penetration: float  # m, depth of the fusion zone
    width_top: float  # m, fusion-zone width at the surface
    width_max: float  # m, widest point of the fusion zone
    depth_at_max: float  # m, depth at which it is widest
    fusion_area: float  # m², cross-sectional area of fused metal
    haz_area: float  # m², cross-sectional area of the HAZ
    haz_depth: float  # m, depth reached by the HAZ
    aspect_ratio: float  # penetration / width_top
    full_penetration: bool
    root_width: float  # m, fusion width at the bottom face (0 if not penetrated)
    melted: bool

    @property
    def penetration_mm(self) -> float:
        return self.penetration * 1e3

    @property
    def width_top_mm(self) -> float:
        return self.width_top * 1e3


@dataclass
class InterfaceMetrics:
    """Weld width measured at the joint interface (top/bottom sheet boundary)."""

    interface_width_min: float  # m, smallest fused width along the weld
    interface_width_max: float  # m, largest fused width along the weld
    interface_width_mean: float  # m, average fused width along the weld
    interface_depth: float  # m, depth of the interface below the top surface
    n_stations: int  # how many x stations contributed to the stats
    keyhole_depth: float  # m, assumed capillary depth from the 3D solver

    @property
    def interface_width_min_mm(self) -> float:
        return self.interface_width_min * 1e3

    @property
    def interface_width_max_mm(self) -> float:
        return self.interface_width_max * 1e3

    @property
    def interface_width_mean_mm(self) -> float:
        return self.interface_width_mean * 1e3

    @property
    def interface_depth_mm(self) -> float:
        return self.interface_depth * 1e3

    @property
    def keyhole_depth_mm(self) -> float:
        return self.keyhole_depth * 1e3


@dataclass
class Solution3D:
    """Fields and geometry from a 3D run.

    Arrays are indexed ``[ix, iy, iz]`` with ``z`` measured downwards from the
    top surface, so ``T[:, :, 0]`` is the face the beam sees.
    """

    x: np.ndarray  # m, along the weld
    y: np.ndarray  # m, transverse
    z: np.ndarray  # m, depth below the top surface
    T: np.ndarray  # K, final field
    T_peak: np.ndarray  # K, highest temperature each cell reached
    t_8_5: np.ndarray  # s, 800→500 °C cooling time (NaN where incomplete)
    cooling_rate: np.ndarray  # K/s at 800 °C (NaN where never crossed)
    dwell_above: np.ndarray  # s above dwell_temp
    dwell_temp: float  # K
    extra_dwell: dict[float, np.ndarray]  # threshold (K) -> s above it
    dt: float  # s, time step actually used
    steps: int
    keyhole_fraction: float  # share of absorbed power put down the capillary
    keyhole_depth: float  # m, assumed capillary depth
    solidus: float  # K
    haz_limit: float  # K
    warnings: list[str]
    T_history: list[np.ndarray] = field(default_factory=list)  # optional animation frames
    frame_times: list[float] = field(default_factory=list)  # times for T_history

    @property
    def thickness(self) -> float:
        return float(self.z[-1])

    def section_index(self) -> int:
        """Index of the transverse section a macro-section would be cut at.

        The mid-point of the fused length, not the widest section: the start is
        cold and the stop crater accumulates heat, so neither represents the
        steady weld the parameters produce.
        """
        fused = (self.T_peak >= self.solidus).any(axis=(1, 2))
        if not fused.any():
            return len(self.x) // 2
        melted = np.flatnonzero(fused)
        return int(melted[len(melted) // 2])

    def section(self, index: int | None = None) -> np.ndarray:
        """Peak-temperature field on a transverse section, shape ``(ny, nz)``."""
        if index is None:
            index = self.section_index()
        return self.T_peak[index, :, :]

    def section_metrics(self, index: int | None = None) -> SectionMetrics:
        """Measure the weld on a transverse section."""
        if index is None:
            index = self.section_index()
        return _section_metrics(
            station=float(self.x[index]),
            y=self.y,
            z=self.z,
            T_peak=self.section(index),
            solidus=self.solidus,
            haz_limit=self.haz_limit,
        )

    def interface_metrics(self, top_thickness: float | None = None) -> InterfaceMetrics:
        """Measure the fused width at the top/bottom sheet interface along the weld.

        For a 2t lap stack the joint is at ``top_thickness`` below the surface.
        If ``top_thickness`` is not supplied, the mid-thickness is used, which
        makes sense for a symmetric full-penetration butt joint.
        """
        if top_thickness is None:
            top_thickness = self.thickness / 2.0
        top_thickness = float(np.clip(top_thickness, 0.0, self.thickness))

        # Find the z-level at or just below the interface depth.
        iz = int(np.searchsorted(self.z, top_thickness, side="right"))
        iz = min(max(iz, 0), len(self.z) - 1)
        # Linear interpolation factor between iz-1 and iz.
        if iz > 0:
            z1, z0 = self.z[iz], self.z[iz - 1]
            if abs(z1 - z0) > 1e-15:
                frac = (top_thickness - z0) / (z1 - z0)
            else:
                frac = 0.0
        else:
            frac = 0.0

        # Slice at the interface, with simple linear interpolation in z.
        if iz > 0 and frac > 0.0:
            T_interface = (1.0 - frac) * self.T_peak[:, :, iz - 1] + frac * self.T_peak[:, :, iz]
        else:
            T_interface = self.T_peak[:, :, iz]

        widths = np.array(
            [level_extent(self.y, T_interface[ix, :], self.solidus) for ix in range(len(self.x))]
        )
        fused = widths > 0.0
        if not fused.any():
            return InterfaceMetrics(
                interface_width_min=0.0,
                interface_width_max=0.0,
                interface_width_mean=0.0,
                interface_depth=top_thickness,
                n_stations=0,
                keyhole_depth=self.keyhole_depth,
            )

        fused_widths = widths[fused]
        return InterfaceMetrics(
            interface_width_min=float(fused_widths.min()),
            interface_width_max=float(fused_widths.max()),
            interface_width_mean=float(fused_widths.mean()),
            interface_depth=top_thickness,
            n_stations=int(fused_widths.size),
            keyhole_depth=self.keyhole_depth,
        )

    def to_history(self) -> ThermalHistory:
        """Project the 3D fields onto the plan view the 2D post-processing uses.

        Peak temperature and melt dwell are taken through the thickness, because
        a cell counts as fused or heat affected if it was hot at any depth. The
        cooling quantities are taken at the top face, which is where CCT data and
        hardness measurements apply.
        """
        return ThermalHistory(
            T_peak=self.T_peak.max(axis=2),
            t_peak=np.zeros(self.T_peak.shape[:2]),
            t_8_5=self.t_8_5[:, :, 0],
            cooling_rate=self.cooling_rate[:, :, 0],
            dwell_above=self.dwell_above.max(axis=2),
            dwell_temp=self.dwell_temp,
            extra_dwell={t: d.max(axis=2) for t, d in self.extra_dwell.items()},
        )


def _section_metrics(
    station: float,
    y: np.ndarray,
    z: np.ndarray,
    T_peak: np.ndarray,
    solidus: float,
    haz_limit: float,
) -> SectionMetrics:
    """Measure fusion zone and HAZ on a ``(ny, nz)`` peak-temperature section.

    Boundaries are interpolated between nodes rather than counted in cells, so a
    2 mm weld on a 0.25 mm mesh does not read as 2.25 mm.
    """
    dz = float(z[1] - z[0]) if z.size > 1 else 0.0
    depth_profile = T_peak.max(axis=0)
    penetration = level_extent(z, depth_profile, solidus)
    haz_depth = level_extent(z, depth_profile, haz_limit)

    fusion_widths = np.array([level_extent(y, T_peak[:, iz], solidus) for iz in range(z.size)])
    haz_widths = np.array([level_extent(y, T_peak[:, iz], haz_limit) for iz in range(z.size)])
    fusion_area = float(fusion_widths.sum()) * dz
    haz_area = float(haz_widths.sum()) * dz - fusion_area

    if not (T_peak >= solidus).any():
        return SectionMetrics(
            station=station,
            penetration=0.0,
            width_top=0.0,
            width_max=0.0,
            depth_at_max=0.0,
            fusion_area=0.0,
            haz_area=max(haz_area, 0.0),
            haz_depth=haz_depth,
            aspect_ratio=0.0,
            full_penetration=False,
            root_width=0.0,
            melted=False,
        )

    width_top = float(fusion_widths[0])
    return SectionMetrics(
        station=station,
        penetration=penetration,
        width_top=width_top,
        width_max=float(fusion_widths.max()),
        depth_at_max=float(z[int(np.argmax(fusion_widths))]),
        fusion_area=fusion_area,
        haz_area=max(haz_area, 0.0),
        haz_depth=haz_depth,
        aspect_ratio=penetration / width_top if width_top > 0 else 0.0,
        full_penetration=bool((T_peak[:, -1] >= solidus).any()),
        root_width=float(fusion_widths[-1]),
        melted=True,
    )


def keyhole_power_fraction(intensity: float) -> float:
    """Share of the absorbed power delivered inside the capillary.

    Zero in conduction mode, ramping to one as the absorbed intensity climbs
    from the bottom of the transition band to the keyhole threshold.
    """
    onset = KEYHOLE_THRESHOLD_INTENSITY / TRANSITION_BAND
    if intensity <= onset:
        return 0.0
    if intensity >= KEYHOLE_THRESHOLD_INTENSITY:
        return 1.0
    return float((intensity - onset) / (KEYHOLE_THRESHOLD_INTENSITY - onset))


def stable_dt(alpha: float, dx: float, dy: float, dz: float) -> float:
    """Explicit stability limit (s). The mirrored faces double the z coupling."""
    return 0.5 / (alpha * (1.0 / dx**2 + 1.0 / dy**2 + 2.0 / dz**2))


def run_3d_thermal(
    nx: int,
    ny: int,
    nz: int,
    Lx: float,
    Ly: float,
    thickness: float,
    t_end: float,
    weld: WeldParams,
    material: Material,
    dt: float | None = None,
    T0: float = 300.0,
    path: Optional[WeldPath] = None,
    wobble: Optional[WobbleParams] = None,
    phase: Optional[PhaseModel] = None,
    keyhole_depth: float | None = None,
    keyhole_taper: float | None = None,
    dwell_temp: float | None = None,
    extra_dwell_temps: Sequence[float] = (),
    solidus: float | None = None,
    haz_limit: float | None = None,
    emissivity: float = 0.7,
    convection_coefficient: float = 15.0,
    on_progress: Optional[Callable[[float], None]] = None,
    abort: Optional[Callable[[], bool]] = None,
    store_frames: bool = False,
    frame_interval: float = 0.02,
    max_frames: int = 1_000,
) -> Solution3D:
    """Run the 3D transient solve and return the fields plus weld geometry.

    Parameters
    ----------
    nx, ny, nz : int
        Grid points along the weld, across it, and through the thickness.
    thickness : float
        Plate thickness (m); the grid spans the full thickness.
    dt : float | None
        Time step (s). Omit to use a stable step derived from the grid.
    extra_dwell_temps : sequence of float
        Further thresholds (K) to accumulate dwell above, reported in
        ``Solution3D.extra_dwell``: dwell is threshold-specific, so a consumer
        that needs the time above the grain-coarsening temperature has to have
        it counted during the run.
    keyhole_depth : float | None
        Depth of the assumed capillary (m). Omit to use the plate thickness in
        keyhole mode, which lets the solve itself decide how deep the fusion
        zone actually gets.
    keyhole_taper : float | None
        Radius of the capillary at its tip as a fraction of the radius at the
        surface, in ``(0, 1]``. This is the main shape knob when calibrating
        predicted penetration against measured macro-sections; omit for
        ``KEYHOLE_TIP_TAPER``.
    on_progress : callable | None
        Called with a 0..1 fraction roughly 100 times during the run, for GUI
        progress bars.
    abort : callable | None
        If it returns True, the solver stops and raises :class:`AbortError`.
    store_frames : bool
        If True, keep a sequence of 3D temperature snapshots for animation.
    frame_interval : float
        Target time between stored frames (s).
    max_frames : int
        Hard cap on stored frames to avoid runaway memory use.

    Returns
    -------
    Solution3D
    """
    if min(nx, ny, nz) < 3:
        raise ValidationError("The 3D grid needs at least 3 points in each direction.")
    if thickness <= 0:
        raise ValidationError("Plate thickness must be positive.")
    if t_end <= 0:
        raise ValidationError("Simulation time must be positive.")

    x = np.linspace(0.0, Lx, nx)
    y = np.linspace(0.0, Ly, ny)
    z = np.linspace(0.0, thickness, nz)
    dx, dy, dz = x[1] - x[0], y[1] - y[0], z[1] - z[0]

    k = material.thermal_conductivity
    rho = material.density
    cp = material.specific_heat
    alpha = k / (rho * cp)

    dt_max = stable_dt(alpha, dx, dy, dz)
    if dt is None:
        dt = DT_SAFETY * dt_max
    elif dt > dt_max:
        raise StabilityError(
            f"Unstable time step for the 3D grid: use dt <= {dt_max:.4g} s "
            f"(you asked for {dt:.4g} s), or coarsen the mesh."
        )

    warnings: list[str] = []
    n_steps = int(np.ceil(t_end / dt))
    updates = nx * ny * nz * n_steps
    if updates > MAX_CELL_UPDATES:
        raise ValidationError(
            f"A {nx}x{ny}x{nz} grid needs {n_steps} stable time steps for {t_end:.3g} s "
            f"({updates / 1e9:.1f} billion cell updates), which would take far too long. "
            "Coarsen the mesh, shorten the simulated time, or shrink the domain."
        )

    intensity = peak_intensity(weld, wobble.amplitude if wobble else 0.0)
    k_fraction = keyhole_power_fraction(intensity)
    if keyhole_depth is None:
        keyhole_depth = thickness if k_fraction > 0 else 0.0
    keyhole_depth = min(keyhole_depth, thickness)

    solidus_T = material.solidus if solidus is None else solidus
    haz_T = material.haz_outer_temperature if haz_limit is None else haz_limit
    dwell_threshold = solidus_T if dwell_temp is None else dwell_temp

    T = np.full((nx, ny, nz), T0)
    T_new = T.copy()
    T_peak = np.full((nx, ny, nz), T0)
    t_cross_hi = np.full((nx, ny, nz), np.nan)
    t_cross_lo = np.full((nx, ny, nz), np.nan)
    cooling_rate = np.full((nx, ny, nz), np.nan)
    dwell_above = np.zeros((nx, ny, nz))
    extra_thresholds_list = [
        float(temp) for temp in extra_dwell_temps if not math.isclose(float(temp), dwell_threshold)
    ]
    n_extra = len(extra_thresholds_list)
    extra_dwell_stack = np.zeros((n_extra, nx, ny, nz)) if n_extra else np.empty((0, nx, ny, nz))
    extra_dwell_arrs = [extra_dwell_stack[e] for e in range(n_extra)]
    extra_dwell = dict(zip(extra_thresholds_list, extra_dwell_arrs))
    extra_thresholds_arr = np.array(extra_thresholds_list, dtype=np.float64)

    # Only the neighbourhood of the beam gets a source term, which keeps the
    # per-step cost proportional to the pool rather than to the plate.
    reach = 4.0 * weld.sigma + (wobble.amplitude if wobble else 0.0)
    half_i = max(int(np.ceil(reach / dx)), 1)
    half_j = max(int(np.ceil(reach / dy)), 1)

    absorbed = weld.absorbed_power
    surface_share = absorbed * (1.0 - k_fraction)
    keyhole_share = absorbed * k_fraction

    # Depth profile of the capillary source: the channel radius tapers with
    # depth, and the energy per unit depth is uniform along it.
    tip_taper = KEYHOLE_TIP_TAPER if keyhole_taper is None else float(keyhole_taper)
    if not 0.0 < tip_taper <= 1.0:
        raise ValidationError(
            f"Keyhole tip taper must be in (0, 1], got {tip_taper}: 1 is a straight "
            "channel and smaller values pinch the tip."
        )
    if keyhole_share > 0 and keyhole_depth > 0:
        taper = 1.0 - (1.0 - tip_taper) * np.clip(z / keyhole_depth, 0.0, 1.0)
        radius_z = KEYHOLE_RADIUS_FACTOR * weld.sigma * taper
        in_channel = z <= keyhole_depth
    else:
        radius_z = np.full_like(z, KEYHOLE_RADIUS_FACTOR * weld.sigma)
        in_channel = np.zeros_like(z, dtype=bool)

    sub_samples = 1
    if wobble is not None and wobble.frequency > 0:
        per_step = wobble.frequency * dt * WOBBLE_SUBSAMPLES_PER_PERIOD
        sub_samples = int(min(max(np.ceil(per_step), 1), MAX_WOBBLE_SUBSAMPLES))
        if per_step > MAX_WOBBLE_SUBSAMPLES:
            warnings.append(
                f"Wobble at {wobble.frequency:.0f} Hz is faster than the time step can "
                f"resolve; the beam is applied as its {MAX_WOBBLE_SUBSAMPLES}-sample "
                "average per step. The swept track is right, the instantaneous spot is not."
            )

    def beam_position(t: float) -> tuple[float, float]:
        if path is not None:
            return beam_at_time(path, wobble or WobbleParams(amplitude=0.0, frequency=0.0), t)
        return weld_position_at_time(weld, t)

    # Nodes on the top and bottom faces own half a cell, so they carry half the
    # heat capacity and must be given half the energy share.
    layer_weight = np.ones(nz)
    layer_weight[0] = 0.5
    layer_weight[-1] = 0.5

    cell_volume = dx * dy * dz
    two_sigma2 = 2.0 * weld.sigma**2

    use_numba = HAS_NUMBA

    # The step is memory-bound, so every field it touches is allocated once and
    # reused; temporaries in the loop cost more than the arithmetic does.
    Q = np.zeros((nx, ny, nz))
    inv_dx2, inv_dy2, inv_dz2 = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2
    if use_numba:
        padded = None
        lap = None
        buf = None
        mask = None
        loss = None
        diagonal = 0.0
    else:
        padded = np.full((nx + 2, ny + 2, nz + 2), T0)
        lap = np.empty((nx, ny, nz))
        buf = np.empty((nx, ny, nz))
        mask = np.empty((nx, ny, nz), dtype=bool)
        loss = np.zeros((nx, ny, nz))
        diagonal = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2)

    T_history: list[np.ndarray] = []
    frame_times: list[float] = []
    if store_frames:
        frame_step = max(1, int(round(frame_interval / dt)))
        if (n_steps // frame_step) + 1 > max_frames:
            frame_step = max(1, n_steps // max_frames)
        T_history.append(T.copy())
        frame_times.append(0.0)

    for step in range(n_steps):
        t = step * dt

        # --- source term, built only on a window around the beam -------------
        Q[:] = 0.0
        if path is None or t < path.duration:
            for sub in range(sub_samples):
                x_src, y_src = beam_position(t + (sub + 0.5) * dt / sub_samples)
                i0 = max(int(round(x_src / dx)) - half_i, 0)
                i1 = min(int(round(x_src / dx)) + half_i + 1, nx)
                j0 = max(int(round(y_src / dy)) - half_j, 0)
                j1 = min(int(round(y_src / dy)) + half_j + 1, ny)
                if i0 >= i1 or j0 >= j1:
                    continue
                r2 = (x[i0:i1, None] - x_src) ** 2 + (y[None, j0:j1] - y_src) ** 2

                if surface_share > 0:
                    # Absorbed on the top face, so it heats the first cell layer.
                    spot = np.exp(-r2 / two_sigma2)
                    total = spot.sum() * cell_volume * layer_weight[0]
                    if total > 0:
                        Q[i0:i1, j0:j1, 0] += (surface_share / sub_samples) * spot / total
                if keyhole_share > 0 and in_channel.any():
                    radial = np.exp(
                        -r2[:, :, None] / (2.0 * np.maximum(radius_z[None, None, :], 1e-9) ** 2)
                    )
                    radial = radial * in_channel[None, None, :]
                    total = (radial * layer_weight[None, None, :]).sum() * cell_volume
                    if total > 0:
                        Q[i0:i1, j0:j1, :] += (keyhole_share / sub_samples) * radial / total

        # --- explicit conduction update ---------------------------------------
        if use_numba:
            numba_step(
                T,
                T_new,
                T_peak,
                Q,
                dwell_above,
                t_cross_hi,
                t_cross_lo,
                cooling_rate,
                extra_dwell_stack,
                extra_thresholds_arr,
                nx,
                ny,
                nz,
                dt,
                t,
                k,
                rho,
                cp,
                inv_dx2,
                inv_dy2,
                inv_dz2,
                T0,
                dz,
                convection_coefficient,
                emissivity,
                SIGMA_SB,
                phase is not None,
                solidus_T if phase is not None else 0.0,
                phase.liquidus if phase is not None else 0.0,
                phase.latent_heat if phase is not None else 0.0,
                phase.boiling if (phase is not None and phase.boiling is not None) else 0.0,
                dwell_threshold,
                T85_UPPER,
                T85_LOWER,
            )
            # Dirichlet sides at ambient (interior already written by numba_step).
            T_new[0, :, :] = T0
            T_new[-1, :, :] = T0
            T_new[:, 0, :] = T0
            T_new[:, -1, :] = T0
        else:
            # Pure NumPy path (kept for environments without numba).
            padded[1:-1, 1:-1, 1:-1] = T
            padded[1:-1, 1:-1, 0] = T[:, :, 1]
            padded[1:-1, 1:-1, -1] = T[:, :, -2]
            np.add(padded[2:, 1:-1, 1:-1], padded[:-2, 1:-1, 1:-1], out=lap)
            lap *= inv_dx2
            np.add(padded[1:-1, 2:, 1:-1], padded[1:-1, :-2, 1:-1], out=buf)
            buf *= inv_dy2
            lap += buf
            np.add(padded[1:-1, 1:-1, 2:], padded[1:-1, 1:-1, :-2], out=buf)
            buf *= inv_dz2
            lap += buf
            np.multiply(T, diagonal, out=buf)
            lap -= buf

            for iz in (0, nz - 1):
                face = T[:, :, iz]
                flux = convection_coefficient * (face - T0)
                if emissivity > 0:
                    flux = flux + emissivity * SIGMA_SB * (face**4 - T0**4)
                loss[:, :, iz] = 2.0 * flux / dz

            if phase is None:
                rho_cp = rho * cp
            else:
                rho_cp = rho * phase.apparent_capacity(T, cp)
            lap *= k
            lap += Q
            lap -= loss
            lap *= dt / rho_cp
            np.add(T, lap, out=T_new)

            T_new[0, :, :] = T0
            T_new[-1, :, :] = T0
            T_new[:, 0, :] = T0
            T_new[:, -1, :] = T0

            if phase is not None and phase.boiling is not None:
                np.minimum(T_new, phase.boiling, out=T_new)

            np.maximum(T_peak, T_new, out=T_peak)
            np.greater(T_new, dwell_threshold, out=mask)
            np.add(dwell_above, dt, out=dwell_above, where=mask)
            for threshold, dwell in extra_dwell.items():
                np.greater(T_new, threshold, out=mask)
                np.add(dwell, dt, out=dwell, where=mask)
        # Nothing can cross 800/500 C while the whole plate is below 500 C, which
        # is most of the run for a fast pass on a big plate.
        if not use_numba and T.max() >= T85_LOWER:
            for level, crossings in ((T85_UPPER, t_cross_hi), (T85_LOWER, t_cross_lo)):
                np.less(T_new, level, out=mask)
                mask &= T >= level
                mask &= np.isnan(crossings)
                if mask.any():
                    span = T[mask] - T_new[mask]
                    frac = (T[mask] - level) / np.where(span > 0, span, 1.0)
                    crossings[mask] = t + frac * dt
                    if level == T85_UPPER:
                        cooling_rate[mask] = span / dt

        T, T_new = T_new, T

        if store_frames and (step + 1) % frame_step == 0:
            T_history.append(T.copy())
            frame_times.append((step + 1) * dt)

        if abort is not None and step % 10 == 0 and abort():
            raise AbortError("Simulation aborted by user.")

        if on_progress is not None and step % max(n_steps // 100, 1) == 0:
            on_progress((step + 1) / n_steps)

    if on_progress is not None:
        on_progress(1.0)

    if phase is not None and phase.boiling is not None and T_peak.max() >= phase.boiling:
        warnings.append(
            f"The surface reached the boiling point ({phase.boiling:.0f} K), where this "
            "model caps it. Penetration is then governed by the assumed capillary "
            "rather than resolved vapour dynamics — the OpenFOAM case is the honest "
            "answer for that regime."
        )

    return Solution3D(
        x=x,
        y=y,
        z=z,
        T=T,
        T_peak=T_peak,
        t_8_5=t_cross_lo - t_cross_hi,
        cooling_rate=cooling_rate,
        dwell_above=dwell_above,
        dwell_temp=dwell_threshold,
        extra_dwell=extra_dwell,
        dt=dt,
        steps=n_steps,
        T_history=T_history,
        frame_times=frame_times,
        keyhole_fraction=k_fraction,
        keyhole_depth=keyhole_depth,
        solidus=solidus_T,
        haz_limit=haz_T,
        warnings=warnings,
    )
