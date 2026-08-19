"""HAZ microstructure and hardness prediction from the weld thermal cycle.

Two things decide what the heat affected zone ends up as: the peak temperature
each point reached (which band of the HAZ it belongs to) and how fast it cooled
afterwards (which phases form). Both come out of the thermal solution, so the
metallurgical prediction is a post-processing step.

The transformation limits are alloy data, not code constants: they live in the
``haz`` block of the material YAML so an engineer can drop in the CCT data for
their own plate. Hardness for steels uses Yurioka's empirical relations, which
are functions of composition; both are approximations intended for parameter
screening rather than qualification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .materials import Material
from .thermal.fd_solver import ThermalHistory
from .weld_metrics import level_extent

#: Hardness above which hydrogen-assisted cold cracking becomes a real concern
#: in the HAZ of a ferritic steel weld (typical fabrication-code trigger).
CRACKING_HARDNESS_HV = 380.0

#: Time above the grain-coarsening temperature beyond which the coarse-grained
#: HAZ is pronounced enough to warn about (s).
COARSENING_DWELL_WARNING = 2.0

#: Multiple of the no-martensite cooling time at which transformation is fully
#: diffusional (ferrite + pearlite, no bainite left).
DIFFUSIONAL_COMPLETION = 4.0


@dataclass
class HazBand:
    """One HAZ band, measured on the peak-temperature field."""

    name: str
    t_min: float  # K
    t_max: float  # K
    width: float  # m, per side of the weld
    area: float  # m², plan view
    note: str = ""

    @property
    def width_mm(self) -> float:
        return self.width * 1e3


@dataclass
class MicrostructureResult:
    """Predicted HAZ constitution for one weld."""

    bands: list[HazBand]
    phases: dict[str, float]  # volume fractions, sums to 1 where predicted
    hardness_hv: float | None
    base_hardness_hv: float
    carbon_equivalent: float | None  # CE_IIW, steels only
    t_8_5: float | None  # s
    cooling_rate: float | None  # K/s
    coarse_grain_width: float  # m, per side, above the grain-coarsening limit
    coarse_grain_dwell: float | None  # s, time above it; None if the solver did not track it
    model: str  # which cooling-response model was used
    warnings: list[str] = field(default_factory=list)

    @property
    def dominant_phase(self) -> str | None:
        if not self.phases:
            return None
        return max(self.phases, key=lambda name: self.phases[name])


def carbon_equivalent_iiw(composition: dict[str, float]) -> float | None:
    """IIW carbon equivalent, the usual weldability index for C-Mn steels.

    ``CE = C + Mn/6 + (Cr + Mo + V)/5 + (Ni + Cu)/15``
    """
    if "C" not in composition:
        return None
    c = composition
    return (
        c.get("C", 0.0)
        + c.get("Mn", 0.0) / 6.0
        + (c.get("Cr", 0.0) + c.get("Mo", 0.0) + c.get("V", 0.0)) / 5.0
        + (c.get("Ni", 0.0) + c.get("Cu", 0.0)) / 15.0
    )


def _martensite_hardness(composition: dict[str, float]) -> float:
    """Yurioka's fully-martensitic HAZ hardness (HV): 884·C·(1 − 0.3·C²) + 294."""
    c = composition.get("C", 0.0)
    return 884.0 * c * (1.0 - 0.3 * c**2) + 294.0


def _bainite_hardness(composition: dict[str, float]) -> float:
    """Yurioka's fully-bainitic HAZ hardness (HV): 145 + 130·tanh(2.65·CE_II − 0.69)."""
    c = composition
    ce2 = (
        c.get("C", 0.0)
        + c.get("Si", 0.0) / 24.0
        + c.get("Mn", 0.0) / 6.0
        + c.get("Cu", 0.0) / 15.0
        + c.get("Ni", 0.0) / 12.0
        + c.get("Cr", 0.0) / 8.0
        + c.get("Mo", 0.0) / 4.0
    )
    return 145.0 + 130.0 * math.tanh(2.65 * ce2 - 0.69)


def _log_fraction(value: float, full: float, none: float) -> float:
    """Fraction of the fast-cooling product, interpolated on a log scale.

    ``full`` is the limit at which the product is complete, ``none`` the limit at
    which it has disappeared; CCT diagrams are read on log time, hence the log.
    """
    lo, hi = min(full, none), max(full, none)
    if value <= lo:
        frac = 1.0 if full <= none else 0.0
    elif value >= hi:
        frac = 0.0 if full <= none else 1.0
    else:
        s = (math.log10(value) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        frac = 1.0 - s if full <= none else s
    return float(min(1.0, max(0.0, frac)))


def _diffusional_phases(material: Material, t_8_5: float) -> dict[str, float]:
    """Phase fractions for a steel, keyed off the 800→500 °C cooling time."""
    response = material.cooling_response
    martensite = _log_fraction(t_8_5, response.t85_martensite, response.t85_no_martensite)
    # Past the no-martensite limit, bainite is progressively replaced by the
    # diffusional ferrite/pearlite products, complete by roughly four times that
    # cooling time.
    slow = _log_fraction(
        t_8_5,
        response.t85_no_martensite * DIFFUSIONAL_COMPLETION,
        response.t85_no_martensite,
    )
    slow = min(slow, 1.0 - martensite)
    return {
        response.fast_phase: martensite,
        response.intermediate_phase: max(0.0, 1.0 - martensite - slow),
        response.slow_phase: slow,
    }


def _rate_phases(material: Material, cooling_rate: float) -> dict[str, float]:
    """Phase fractions for an alloy governed by cooling rate (e.g. Ti-6Al-4V)."""
    response = material.cooling_response
    fast = _log_fraction(
        cooling_rate,
        response.rate_full_martensite,
        response.rate_no_martensite,
    )
    return {response.fast_phase: fast, response.slow_phase: 1.0 - fast}


def _hardness(material: Material, phases: dict[str, float]) -> float | None:
    """Weighted hardness of the predicted phase mixture (HV)."""
    response = material.cooling_response
    if response.kind == "diffusional_t85":
        if "C" not in material.composition:
            return None
        hv = {
            response.fast_phase: _martensite_hardness(material.composition),
            response.intermediate_phase: _bainite_hardness(material.composition),
            response.slow_phase: material.base_hardness_hv,
        }
    else:
        fast_hv = response.fast_hardness_hv
        slow_hv = response.slow_hardness_hv
        if fast_hv is None or slow_hv is None:
            return None
        hv = {response.fast_phase: fast_hv, response.slow_phase: slow_hv}
    return sum(fraction * hv[name] for name, fraction in phases.items() if name in hv)


def predict_microstructure(
    x: np.ndarray,
    y: np.ndarray,
    history: ThermalHistory,
    material: Material,
    t_8_5: float | None,
    cooling_rate: float | None,
) -> MicrostructureResult:
    """Predict HAZ bands, phase fractions and hardness for a thermal solution.

    ``t_8_5`` and ``cooling_rate`` are the representative HAZ values from
    :func:`~weldsim.weld_metrics.compute_weld_metrics`; either may be ``None``
    when the run stopped before the HAZ finished cooling.
    """
    T_peak = history.T_peak
    dx = float(x[1] - x[0]) if x.size > 1 else 0.0
    dy = float(y[1] - y[0]) if y.size > 1 else 0.0
    cell_area = dx * dy

    station = int(np.unravel_index(np.argmax(T_peak), T_peak.shape)[0])
    section = T_peak[station, :]

    bands: list[HazBand] = []
    for zone in material.haz_zones:
        inside = (T_peak >= zone.t_min) & (T_peak < zone.t_max)
        # Width per side: outer extent of the band minus the extent of everything
        # hotter than it, halved.
        outer = level_extent(y, section, zone.t_min)
        inner = level_extent(y, section, zone.t_max)
        bands.append(
            HazBand(
                name=zone.name,
                t_min=zone.t_min,
                t_max=zone.t_max,
                width=max(0.0, (outer - inner) / 2.0),
                area=float(inside.sum()) * cell_area,
                note=zone.note,
            )
        )

    coarse_limit = material.grain_coarsening_temperature
    coarse_dwell: float | None = None
    if coarse_limit is None:
        coarse_width = 0.0
    else:
        coarse_width = level_extent(y, section, coarse_limit) / 2.0
        # Dwell is counted per threshold during the run: the melt dwell in
        # ``history.dwell_above`` is time above the solidus, which is a much
        # shorter and unrelated interval, so it cannot stand in here.
        dwell = history.time_above(coarse_limit)
        if dwell is not None:
            above = T_peak >= coarse_limit
            coarse_dwell = float(dwell[above].max()) if above.any() else 0.0

    warnings: list[str] = []
    response = material.cooling_response
    phases: dict[str, float] = {}
    if response.kind == "diffusional_t85":
        if t_8_5 is not None and t_8_5 > 0:
            phases = _diffusional_phases(material, t_8_5)
        else:
            warnings.append(
                "No t8/5 available, so no phase prediction: extend the simulation "
                "time so the HAZ cools through 500 °C."
            )
    else:
        if cooling_rate is not None and cooling_rate > 0:
            phases = _rate_phases(material, cooling_rate)
        else:
            warnings.append(
                "No HAZ cooling rate available, so no phase prediction: extend "
                "the simulation time past the end of the weld."
            )

    hardness = _hardness(material, phases) if phases else None
    if response.kind == "diffusional_t85":
        if hardness is not None and hardness >= CRACKING_HARDNESS_HV:
            warnings.append(
                f"Predicted HAZ hardness {hardness:.0f} HV is above "
                f"{CRACKING_HARDNESS_HV:.0f} HV: hydrogen-assisted cracking risk. "
                "Consider preheat, slower cooling or a lower-carbon-equivalent plate."
            )
    elif phases.get(response.fast_phase, 0.0) > 0.9:
        warnings.append(
            f"The HAZ is predicted to be essentially all {response.fast_phase.lower()}: "
            "strong but with reduced ductility and toughness. A post-weld stress "
            "relief or anneal is usual."
        )
    coarsened = coarse_dwell is not None and coarse_dwell > COARSENING_DWELL_WARNING
    if coarse_limit is not None and coarse_width > 0 and coarsened:
        warnings.append(
            f"{coarse_dwell:.1f} s spent above {coarse_limit - 273.15:.0f} °C: "
            "expect pronounced grain coarsening and reduced HAZ toughness."
        )

    return MicrostructureResult(
        bands=bands,
        phases=phases,
        hardness_hv=hardness,
        base_hardness_hv=material.base_hardness_hv,
        carbon_equivalent=carbon_equivalent_iiw(material.composition),
        t_8_5=t_8_5,
        cooling_rate=cooling_rate,
        coarse_grain_width=coarse_width,
        coarse_grain_dwell=coarse_dwell,
        model=response.kind,
        warnings=warnings,
    )


__all__ = [
    "HazBand",
    "MicrostructureResult",
    "predict_microstructure",
    "carbon_equivalent_iiw",
    "CRACKING_HARDNESS_HV",
]
