"""Weld geometry and thermal-cycle metrics derived from a thermal solution.

The finite-difference run gives temperature fields; a welding engineer needs the
consequences of those fields: how wide the fusion zone is, how far the heat
affected zone reaches, how long the pool stayed molten and how fast the HAZ
cooled. Everything here is read off the peak-temperature and cooling-rate
fields in :class:`~weldsim.thermal.fd_solver.ThermalHistory`.

Coordinates follow the solver: ``x`` runs along the weld, ``y`` across it, and
fields are indexed ``[ix, iy]``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .materials import Material
from .thermal.fd_solver import T85_LOWER, T85_UPPER, ThermalHistory
from .types import WeldParams


@dataclass
class TransverseProfile:
    """Peak temperature across the weld at one station along it."""

    x: float  # m, station along the weld
    y: np.ndarray  # m
    T_peak: np.ndarray  # K


@dataclass
class WeldMetrics:
    """Fusion-zone and HAZ geometry plus the thermal cycle that produced it."""

    peak_temperature: float  # K
    melted: bool
    fusion_width: float  # m, widest transverse extent above the solidus
    fusion_length: float  # m, extent along the weld
    fusion_area: float  # m², plan-view melted area
    haz_width: float  # m, per side, between fusion boundary and HAZ limit
    haz_area: float  # m², plan view
    melt_dwell: float  # s, longest time any point stayed above the solidus
    t_8_5: float | None  # s, representative HAZ cooling time 800→500 °C
    cooling_rate: float | None  # K/s, representative HAZ rate at 800 °C
    heat_input: float  # J/mm of absorbed energy per unit length
    solidus: float  # K, threshold used for the fusion boundary
    haz_limit: float  # K, threshold used for the outer HAZ boundary
    profile: TransverseProfile
    warnings: list[str] = field(default_factory=list)

    @property
    def fusion_width_mm(self) -> float:
        return self.fusion_width * 1e3

    @property
    def haz_width_mm(self) -> float:
        return self.haz_width * 1e3

    @property
    def aspect_note(self) -> str:
        """Human-readable summary of whether anything melted at all."""
        if not self.melted:
            return "No melting: peak temperature stayed below the solidus."
        return (
            f"Fusion zone {self.fusion_width_mm:.2f} mm wide, HAZ {self.haz_width_mm:.2f} mm/side."
        )


def level_extent(coord: np.ndarray, values: np.ndarray, level: float) -> float:
    """Width of the region around the maximum where ``values >= level``.

    The boundary is linearly interpolated between the bracketing nodes, so the
    result is not quantised to the mesh spacing.

    Returns
    -------
    float
        Extent in the units of ``coord``; 0.0 if the level is never reached.
    """
    if values.size == 0 or float(values.max()) < level:
        return 0.0
    peak = int(np.argmax(values))

    def edge(step: int) -> float:
        i = peak
        while 0 <= i + step < values.size and values[i + step] >= level:
            i += step
        j = i + step
        if not (0 <= j < values.size):
            return float(coord[i])  # region runs off the domain
        span = values[i] - values[j]
        if span <= 0:
            return float(coord[i])
        frac = (values[i] - level) / span
        return float(coord[i] + frac * (coord[j] - coord[i]))

    return abs(edge(1) - edge(-1))


def _representative(values: np.ndarray, mask: np.ndarray) -> float | None:
    """Median of ``values`` over ``mask``, ignoring cells that never crossed."""
    selected = values[mask]
    selected = selected[np.isfinite(selected)]
    if selected.size == 0:
        return None
    return float(np.median(selected))


def compute_weld_metrics(
    x: np.ndarray,
    y: np.ndarray,
    history: ThermalHistory,
    material: Material,
    weld: WeldParams,
) -> WeldMetrics:
    """Derive fusion zone, HAZ and cooling metrics from a thermal history."""
    T_peak = history.T_peak
    solidus = material.solidus
    haz_limit = material.haz_outer_temperature

    dx = float(x[1] - x[0]) if x.size > 1 else 0.0
    dy = float(y[1] - y[0]) if y.size > 1 else 0.0
    cell_area = dx * dy

    fused = T_peak >= solidus
    heat_affected = (T_peak >= haz_limit) & ~fused

    # Report the widest transverse section, which is what a macro-section of the
    # weld would be cut at; fall back to the hottest station if nothing melted.
    if fused.any():
        station = int(np.argmax(fused.sum(axis=1)))
    else:
        station = int(np.unravel_index(np.argmax(T_peak), T_peak.shape)[0])

    section = T_peak[station, :]
    fusion_width = level_extent(y, section, solidus)
    outer_width = level_extent(y, section, haz_limit)
    haz_width = max(0.0, (outer_width - fusion_width) / 2.0)

    fusion_length = level_extent(x, T_peak.max(axis=1), solidus)

    # The cooling cycle that matters is the one just outside the fusion
    # boundary: the coarse-grained HAZ is where cracking and toughness loss show
    # up first.
    coarse_limit = material.grain_coarsening_temperature or haz_limit
    near_boundary = heat_affected & (T_peak >= coarse_limit)
    if not near_boundary.any():
        near_boundary = heat_affected

    t_8_5 = _representative(history.t_8_5, near_boundary)
    cooling_rate = _representative(history.cooling_rate, near_boundary)

    warnings: list[str] = []
    if not fused.any():
        warnings.append(
            "Peak temperature never reached the solidus: these parameters do not "
            "melt the plate, so there is no weld to assess."
        )
    if heat_affected.any() and t_8_5 is None:
        warnings.append(
            f"The HAZ had not cooled from {T85_UPPER - 273.15:.0f} to "
            f"{T85_LOWER - 273.15:.0f} °C when the run ended, so t8/5 and the "
            "microstructure prediction are unavailable. Extend the simulation "
            "time past the end of the weld."
        )
    if float(T_peak.max()) > material.boiling:
        warnings.append(
            f"Peak temperature {float(T_peak.max()):.0f} K is above the boiling "
            f"point ({material.boiling:.0f} K). The thin-plate model has no latent "
            "heat or evaporation, so it overshoots: reduce the heat input or use "
            "the 3D keyhole solver for this regime."
        )
    if fused.any() and fusion_width < 2.5 * dy:
        warnings.append(
            f"The fusion zone spans only ~{fusion_width / dy:.1f} cells across the "
            "weld; increase the transverse grid resolution before trusting the width."
        )

    return WeldMetrics(
        peak_temperature=float(T_peak.max()),
        melted=bool(fused.any()),
        fusion_width=fusion_width,
        fusion_length=fusion_length,
        fusion_area=float(fused.sum()) * cell_area,
        haz_width=haz_width,
        haz_area=float(heat_affected.sum()) * cell_area,
        melt_dwell=float(history.dwell_above.max()) if history.dwell_temp <= solidus else 0.0,
        t_8_5=t_8_5,
        cooling_rate=cooling_rate,
        heat_input=weld.line_energy_j_per_mm,
        solidus=solidus,
        haz_limit=haz_limit,
        profile=TransverseProfile(x=float(x[station]), y=y, T_peak=section),
        warnings=warnings,
    )


__all__ = ["WeldMetrics", "TransverseProfile", "compute_weld_metrics", "level_extent"]
