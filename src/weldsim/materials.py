"""Material library loader for welding simulations.

Loads temperature-dependent material properties from the keyhole-cfd/materials/
YAML files and exposes them to the simple 2D thermal solver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import yaml


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

    @property
    def thermal_diffusivity(self) -> float:
        """Thermal diffusivity [m²/s]."""
        return self.thermal_conductivity / (self.density * self.specific_heat)

    @property
    def room_temp_density(self) -> float:
        """Alias for density (used by solvers)."""
        return self.density


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
        raise FileNotFoundError(
            f"Material '{name}' not found. Available: {', '.join(available)}"
        )

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
        sigma = lambda _: 1.0
        dsig = lambda _: 0.0
        t_ref = at_temperature

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
    )


def list_materials() -> list[str]:
    """Return available material YAML names."""
    root = Path(__file__).resolve().parent.parent.parent / "keyhole-cfd" / "materials"
    return sorted([p.stem for p in root.glob("*.yaml")])
