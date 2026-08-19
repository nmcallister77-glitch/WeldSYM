"""Material library loader for welding simulations.

Loads temperature-dependent material properties from the keyhole-cfd/materials/
YAML files and exposes them to the simple 2D thermal solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import yaml

from .types import MaterialParams


@dataclass
class HazZone:
    """One metallurgical band of the heat affected zone.

    Bands are defined by the peak temperature the material reached, which is
    what a peak-temperature field can be sliced on directly.
    """

    name: str
    t_min: float  # K
    t_max: float  # K
    note: str = ""


@dataclass
class CoolingResponse:
    """How the alloy's microstructure responds to weld cooling rate.

    ``kind`` selects the model in :mod:`weldsim.microstructure`:

    ``diffusional_t85``
        Transformation is governed by the 800→500 °C cooling time; martensite
        forms below ``t85_martensite`` and is absent above ``t85_no_martensite``
        (steels, read off a CCT diagram).
    ``martensitic_rate``
        Transformation is governed by the instantaneous cooling rate; fully
        martensitic above ``rate_full_martensite`` K/s and fully diffusional
        below ``rate_no_martensite`` K/s (titanium alloys).
    """

    kind: str = "diffusional_t85"
    t85_martensite: float = 3.0  # s
    t85_no_martensite: float = 25.0  # s
    rate_full_martensite: float = 410.0  # K/s
    rate_no_martensite: float = 20.0  # K/s
    fast_phase: str = "Martensite"
    slow_phase: str = "Ferrite + pearlite"
    intermediate_phase: str = "Bainite"
    fast_hardness_hv: float | None = None
    slow_hardness_hv: float | None = None


@dataclass
class Mechanical:
    """Room-temperature mechanical properties used by the distortion model."""

    youngs_modulus: float = 210e9  # Pa
    yield_stress: float = 355e6  # Pa
    thermal_expansion: float = 12e-6  # 1/K
    poisson_ratio: float = 0.3

    @property
    def yield_strain(self) -> float:
        """Elastic strain at yield (-)."""
        return self.yield_stress / self.youngs_modulus


@dataclass
class Material:
    """Temperature-dependent material properties for a welding alloy."""

    name: str
    density: float  # kg/m³
    thermal_conductivity: float  # W/(m·K)
    specific_heat: float  # J/(kg·K)
    surface_tension: float = 1.0  # N/m
    d_sigma_dT: float = 0.0  # N/(m·K)
    solidus: float = 1700.0  # K
    liquidus: float = 1800.0  # K
    boiling: float = 3000.0  # K
    latent_heat_fusion: float = 2.7e5  # J/kg
    latent_heat_vaporization: float = 6.2e6  # J/kg
    T0: float = 293.15  # K
    composition: dict[str, float] = field(default_factory=dict)  # wt%
    mechanical: Mechanical = field(default_factory=Mechanical)
    haz_zones: list[HazZone] = field(default_factory=list)
    cooling_response: CoolingResponse = field(default_factory=CoolingResponse)
    base_hardness_hv: float = 170.0
    grain_coarsening_temperature: float | None = None  # K
    haz_lower_temperature: float | None = None  # K

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity [m²/s]."""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @property
    def room_temp_density(self) -> float:
        """Alias for density (used by solvers)."""
        return self.density

    @property
    def melting_enthalpy(self) -> float:
        """Energy needed to take 1 kg from T0 to a fully molten state [J/kg]."""
        return self.specific_heat * (self.liquidus - self.T0) + self.latent_heat_fusion

    @property
    def haz_outer_temperature(self) -> float:
        """Peak temperature below which the alloy is metallurgically unaffected (K)."""
        if self.haz_lower_temperature is not None:
            return self.haz_lower_temperature
        return self.T0 + 0.5 * (self.solidus - self.T0)


def material_from_params(params: MaterialParams) -> Material:
    """Wrap bare thermal properties as a Material so the metallurgy models run.

    Alloy data is unknown in this case, so the :class:`Material` defaults apply
    and any metallurgical output is generic rather than alloy-specific.
    """
    return Material(
        name="Custom (thermal properties only)",
        density=params.rho,
        thermal_conductivity=params.k,
        specific_heat=params.cp,
        T0=params.T0,
    )


def _interp1d(x: np.ndarray, y: np.ndarray) -> Callable[[float], float]:
    """Simple linear interpolation, clamping outside the table range."""
    x = np.asarray(x)
    y = np.asarray(y)

    def f(t: float) -> float:
        if t <= x[0]:
            return float(y[0])
        if t >= x[-1]:
            return float(y[-1])
        idx = np.searchsorted(x, t)
        if idx == 0:
            return float(y[0])
        x0, x1 = x[idx - 1], x[idx]
        y0, y1 = y[idx - 1], y[idx]
        return float(y0 + (y1 - y0) * (t - x0) / (x1 - x0))

    return f


def load_material(name: str, at_temperature: float = 1200.0) -> Material:
    """Load a material from keyhole-cfd/materials/<name>.yaml.

    Parameters
    ----------
    name : str
        Material filename without extension, e.g. "Ti6Al4V" or "S355_structural_steel".
    at_temperature : float
        Representative temperature (K) at which to sample thermal properties.
        The 2D FD solver is isothermal, so this sets k, rho, cp for the run.

    Returns
    -------
    Material
    """
    root = Path(__file__).resolve().parent.parent.parent / "keyhole-cfd" / "materials"
    path = root / f"{name}.yaml"
    if not path.exists():
        available = sorted([p.stem for p in root.glob("*.yaml")])
        raise FileNotFoundError(f"Material '{name}' not found. Available: {', '.join(available)}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    alloy = data["alloy"]
    phase = data["phase_change"]
    thermal = data["thermal"]

    T = np.array(thermal["temperature"], dtype=float)
    k = _interp1d(T, np.array(thermal["thermal_conductivity"], dtype=float))
    rho = _interp1d(T, np.array(thermal["density"], dtype=float))
    cp = _interp1d(T, np.array(thermal["specific_heat"], dtype=float))

    # Surface tension / Marangoni at the liquidus (best default for pool)
    surface = data.get("surface", {})
    if surface:
        Tsurf = np.array(surface["temperature"], dtype=float)
        sigma = _interp1d(Tsurf, np.array(surface["surface_tension"], dtype=float))
        dsig = _interp1d(Tsurf, np.array(surface["marangoni_coefficient"], dtype=float))
        t_ref = phase["liquidus_temperature"]
    else:

        def sigma(_: float) -> float:
            return 1.0

        def dsig(_: float) -> float:
            return 0.0

        t_ref = at_temperature

    haz = data.get("haz", {})
    return Material(
        name=alloy["name"],
        density=rho(at_temperature),
        thermal_conductivity=k(at_temperature),
        specific_heat=cp(at_temperature),
        surface_tension=sigma(t_ref),
        d_sigma_dT=dsig(t_ref),
        solidus=phase["solidus_temperature"],
        liquidus=phase["liquidus_temperature"],
        boiling=phase["boiling_temperature"],
        latent_heat_fusion=phase["latent_heat_fusion"],
        latent_heat_vaporization=phase["latent_heat_vaporization"],
        T0=phase.get("reference_temperature", 293.15),
        composition=_numeric_composition(alloy.get("composition", {})),
        mechanical=_mechanical(data.get("thermomechanical", {})),
        haz_zones=[
            HazZone(
                name=z["name"],
                t_min=float(z["t_min"]),
                t_max=float(z["t_max"]),
                note=z.get("note", ""),
            )
            for z in haz.get("zones", [])
        ],
        cooling_response=_cooling_response(haz.get("cooling_response", {})),
        base_hardness_hv=float(haz.get("base_hardness_hv", 170.0)),
        grain_coarsening_temperature=haz.get("grain_coarsening_temperature"),
        haz_lower_temperature=haz.get("lower_limit_temperature"),
    )


def _numeric_composition(raw: dict[str, Any]) -> dict[str, float]:
    """Keep only numeric alloy fractions, dropping entries like ``Fe: balance``."""
    return {el: float(v) for el, v in raw.items() if isinstance(v, (int, float))}


def _mechanical(raw: dict[str, Any]) -> Mechanical:
    """Room-temperature slice of the thermomechanical property tables."""
    if not raw:
        return Mechanical()
    defaults = Mechanical()
    return Mechanical(
        youngs_modulus=float(raw["youngs_modulus"][0]),
        yield_stress=float(raw["yield_stress"][0]),
        thermal_expansion=float(raw["thermal_expansion"][0]),
        poisson_ratio=float(raw.get("poisson_ratio", defaults.poisson_ratio)),
    )


def _cooling_response(raw: dict[str, Any]) -> CoolingResponse:
    defaults = CoolingResponse()
    return CoolingResponse(
        kind=raw.get("kind", defaults.kind),
        t85_martensite=float(raw.get("t85_martensite", defaults.t85_martensite)),
        t85_no_martensite=float(raw.get("t85_no_martensite", defaults.t85_no_martensite)),
        rate_full_martensite=float(raw.get("rate_full_martensite", defaults.rate_full_martensite)),
        rate_no_martensite=float(raw.get("rate_no_martensite", defaults.rate_no_martensite)),
        fast_phase=raw.get("fast_phase", defaults.fast_phase),
        slow_phase=raw.get("slow_phase", defaults.slow_phase),
        intermediate_phase=raw.get("intermediate_phase", defaults.intermediate_phase),
        fast_hardness_hv=raw.get("fast_hardness_hv"),
        slow_hardness_hv=raw.get("slow_hardness_hv"),
    )


def list_materials() -> list[str]:
    """Return available material YAML names."""
    root = Path(__file__).resolve().parent.parent.parent / "keyhole-cfd" / "materials"
    return sorted([p.stem for p in root.glob("*.yaml")])
