"""How beam oscillation redistributes heat along and across the weld.

Wobble is used to bridge gaps, widen a narrow keyhole weld and stir the pool,
and it does all three by trading peak intensity for swept area. The questions an
engineer asks are: how far does the beam advance between loops (does the track
stay continuous?), how much does the peak energy density drop, and how evenly is
the energy spread over the track?

Those are answered by the beam kinematics plus the accumulated energy-density
map that :func:`weldsim.weld_path.heat_signature` already produces.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .types import WeldParams
from .weld_path import WeldPath, WobbleParams, beam_at_time, heat_signature


@dataclass
class WobbleAnalysis:
    """Heat-concentration metrics for a wobbled (or plain) weld pass."""

    pitch: float  # m advanced per oscillation cycle
    swept_width: float  # m, total width the beam covers
    overlap_ratio: float  # 0-1, overlap of consecutive beam footprints
    mean_beam_speed: float  # m/s, actual speed of the spot over the surface
    peak_beam_speed: float  # m/s
    dwell_time: float  # s, time the beam spends over one spot per pass
    peak_energy_density: float  # J/m², highest accumulated absorbed energy
    mean_energy_density: float  # J/m², averaged over the affected track
    concentration_ratio: float  # peak / mean over the track
    uniformity: float  # 0-1, 1 = perfectly even along the track
    peak_reduction: float  # 0-1, drop in peak energy density versus no wobble
    notes: list[str] = field(default_factory=list)

    @property
    def pitch_mm(self) -> float:
        return self.pitch * 1e3

    @property
    def swept_width_mm(self) -> float:
        return self.swept_width * 1e3


def beam_speed(
    path: WeldPath,
    wobble: WobbleParams,
    dt: float = 1e-5,
    samples: int = 2000,
) -> tuple[float, float]:
    """Mean and peak speed of the beam spot over the surface (m/s).

    Travel speed is what the fixture sees; the spot itself also runs around the
    wobble figure, and at a few hundred hertz that dominates. It is the spot
    speed that decides local dwell time.
    """
    duration = path.duration
    if duration <= 0:
        return 0.0, 0.0
    times = np.linspace(0.0, max(duration - dt, dt), samples)
    speeds = np.empty(times.size)
    for i, t in enumerate(times):
        x0, y0 = beam_at_time(path, wobble, t)
        x1, y1 = beam_at_time(path, wobble, t + dt)
        speeds[i] = math.hypot(x1 - x0, y1 - y0) / dt
    return float(speeds.mean()), float(speeds.max())


def analyse_wobble(
    path: WeldPath,
    wobble: WobbleParams,
    weld: WeldParams,
    thickness: float,
    x: np.ndarray,
    y: np.ndarray,
    t_end: float | None = None,
    dt: float = 0.002,
) -> WobbleAnalysis:
    """Quantify the heat concentration produced by a wobble setting.

    ``x`` and ``y`` are the grid the energy map is evaluated on (usually the
    thermal grid); ``thickness`` is the depth the surface flux is spread over,
    matching the thermal model, so the maps are directly comparable.
    """
    duration = t_end if t_end is not None else path.duration
    beam_diameter = 2.0 * weld.sigma

    if wobble.frequency > 0:
        pitch = path.speed / wobble.frequency
        dwell = 1.0 / wobble.frequency
    else:
        pitch = float("inf")
        dwell = beam_diameter / path.speed if path.speed > 0 else 0.0

    swept_width = beam_diameter + 2.0 * wobble.amplitude
    overlap = 1.0 - min(pitch / beam_diameter, 1.0) if beam_diameter > 0 else 0.0

    mean_speed, peak_speed = beam_speed(path, wobble)

    # Accumulated absorbed energy per unit area, with and without the wobble, so
    # the peak reduction is a like-for-like comparison.
    def energy_map(w: WobbleParams) -> np.ndarray:
        volumetric = heat_signature(
            path,
            w,
            weld.power,
            weld.efficiency,
            weld.sigma,
            thickness,
            x,
            y,
            duration,
            dt=dt,
        )
        return volumetric * thickness  # J/m³ → J/m²

    energy = energy_map(wobble)
    straight = energy_map(WobbleParams(amplitude=0.0, frequency=0.0, pattern=wobble.pattern))

    peak = float(energy.max())
    straight_peak = float(straight.max())
    # "Track" = where a meaningful amount of energy landed, so that the mean is
    # not diluted by the cold parts of the plate.
    track = energy > 0.1 * peak if peak > 0 else np.zeros_like(energy, dtype=bool)
    mean = float(energy[track].mean()) if track.any() else 0.0

    # Uniformity along the weld: how steady the deposited energy is from one
    # station to the next. A pitch longer than the beam shows up here as ripple.
    along = energy.max(axis=1)
    along = along[along > 0.1 * peak] if peak > 0 else along
    if along.size > 1 and along.mean() > 0:
        uniformity = float(max(0.0, 1.0 - along.std() / along.mean()))
    else:
        uniformity = 1.0

    notes: list[str] = []
    if wobble.frequency <= 0 or wobble.amplitude <= 0:
        notes.append("No wobble: the beam runs straight along the path.")
    elif overlap <= 0:
        notes.append(
            f"Beam advances {pitch * 1e3:.2f} mm per loop but is only "
            f"{beam_diameter * 1e3:.2f} mm across, so consecutive loops do not "
            "overlap: expect a scalloped, discontinuous track. Raise the "
            "frequency or lower the travel speed."
        )
    elif overlap < 0.5:
        notes.append(
            f"Only {overlap * 100:.0f} % footprint overlap between loops; "
            "raise the wobble frequency for a smoother bead."
        )
    if straight_peak > 0 and peak < 0.5 * straight_peak:
        notes.append(
            f"Wobble cuts peak energy density to {peak / straight_peak * 100:.0f} % "
            "of the straight-beam value, which widens the bead but reduces "
            "penetration for the same power."
        )
    if peak_speed > 20.0 * path.speed and path.speed > 0:
        notes.append(
            f"The spot moves at up to {peak_speed:.2f} m/s, {peak_speed / path.speed:.0f}× "
            "the travel speed: the pool sees stirring rather than a steady source."
        )

    return WobbleAnalysis(
        pitch=pitch,
        swept_width=swept_width,
        overlap_ratio=max(0.0, min(1.0, overlap)),
        mean_beam_speed=mean_speed,
        peak_beam_speed=peak_speed,
        dwell_time=dwell,
        peak_energy_density=peak,
        mean_energy_density=mean,
        concentration_ratio=peak / mean if mean > 0 else 0.0,
        uniformity=uniformity,
        peak_reduction=1.0 - peak / straight_peak if straight_peak > 0 else 0.0,
        notes=notes,
    )


def energy_density_map(
    path: WeldPath,
    wobble: WobbleParams,
    weld: WeldParams,
    thickness: float,
    x: np.ndarray,
    y: np.ndarray,
    t_end: float | None = None,
    dt: float = 0.002,
) -> np.ndarray:
    """Absorbed energy per unit area over the plate (J/m², shape ``(nx, ny)``)."""
    duration = t_end if t_end is not None else path.duration
    volumetric = heat_signature(
        path,
        wobble,
        weld.power,
        weld.efficiency,
        weld.sigma,
        thickness,
        x,
        y,
        duration,
        dt=dt,
    )
    return volumetric * thickness


__all__ = ["WobbleAnalysis", "analyse_wobble", "energy_density_map", "beam_speed"]
