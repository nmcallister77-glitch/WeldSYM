"""High-level simulation API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from .errors import ValidationError
from .materials import Material, list_materials, load_material
from .thermal.fd_solver import run_2d_fd_thermal
from .types import MaterialParams, WeldParams
from .weld_path import WeldPath, WobbleParams

#: Bounds on problem size, so a mistyped parameter cannot exhaust memory or run for hours.
MAX_CELLS = 1_000_000
MAX_STEPS = 1_000_000


@dataclass
class ThermalSimulationConfig:
    """Configuration for a transient thermal simulation."""

    nx: int = 51
    ny: int = 26
    Lx: float = 0.1  # m
    Ly: float = 0.05  # m
    t_end: float = 10.0  # s
    dt: float = 0.05  # s
    weld: WeldParams | None = None
    material: MaterialParams | Material = field(default_factory=MaterialParams)
    output_file: str | None = "results/temperature.csv"
    T1: float = 0.005  # effective thickness (m)
    path: WeldPath | None = None
    wobble: WobbleParams | None = None
    probe: tuple[float, float] | None = None


def _to_material_params(material: MaterialParams | Material) -> MaterialParams:
    if isinstance(material, Material):
        return MaterialParams(
            k=material.thermal_conductivity,
            rho=material.density,
            cp=material.specific_heat,
            T0=material.T0,
        )
    return material


def validate_config(config: ThermalSimulationConfig) -> None:
    """Check that a configuration is physically and numerically meaningful.

    Raises
    ------
    ValidationError
        Naming the offending parameter and its allowed range.
    """
    positive = {
        "Lx (plate length)": config.Lx,
        "Ly (plate width)": config.Ly,
        "T1 (effective thickness)": config.T1,
        "dt (time step)": config.dt,
        "t_end (simulation time)": config.t_end,
    }
    for name, value in positive.items():
        if not value > 0:
            raise ValidationError(f"{name} must be greater than 0, got {value}.")

    for name, count in (("nx", config.nx), ("ny", config.ny)):
        if count < 3:
            raise ValidationError(
                f"{name} must be at least 3 to have an interior node, got {count}."
            )

    cells = config.nx * config.ny
    if cells > MAX_CELLS:
        raise ValidationError(
            f"Grid of {config.nx}x{config.ny} = {cells} cells exceeds the limit of "
            f"{MAX_CELLS}. Coarsen the mesh or shrink the domain."
        )

    steps = int(np.ceil(config.t_end / config.dt))
    if steps > MAX_STEPS:
        raise ValidationError(
            f"t_end / dt = {steps} time steps exceeds the limit of {MAX_STEPS}. "
            "Increase dt or shorten t_end."
        )

    mat = _to_material_params(config.material)
    material_props = (
        ("k (conductivity)", mat.k),
        ("rho (density)", mat.rho),
        ("cp (specific heat)", mat.cp),
    )
    for name, value in material_props:
        if not value > 0:
            raise ValidationError(f"Material {name} must be greater than 0, got {value}.")

    weld = config.weld
    if weld is None:
        return
    if not weld.power > 0:
        raise ValidationError(f"Power must be greater than 0 W, got {weld.power}.")
    if not 0 < weld.efficiency <= 1:
        raise ValidationError(f"Efficiency must be in (0, 1], got {weld.efficiency}.")
    if not weld.speed > 0:
        raise ValidationError(
            f"Travel speed must be greater than 0 m/s, got {weld.speed}. "
            "A stationary or reversing torch is not supported."
        )
    if not weld.sigma > 0:
        raise ValidationError(f"Beam sigma must be greater than 0 m, got {weld.sigma}.")


def run_thermal_simulation(config: ThermalSimulationConfig) -> Dict[str, np.ndarray]:
    """
    Run a 2D transient thermal simulation with a moving heat source.

    Returns
    -------
    result : dict
        {
          "x": np.ndarray,
          "y": np.ndarray,
          "T": np.ndarray,
        }
    """
    if config.weld is None:
        config.weld = WeldParams(
            power=3000.0,
            efficiency=0.8,
            speed=0.005,
            start_pos=(0.01, config.Ly / 2),
            direction="x",
        )

    validate_config(config)

    mat = _to_material_params(config.material)

    x, y, T, T_probe = run_2d_fd_thermal(
        nx=config.nx,
        ny=config.ny,
        Lx=config.Lx,
        Ly=config.Ly,
        t_end=config.t_end,
        dt=config.dt,
        weld=config.weld,
        material=mat,
        T0=mat.T0,
        h=config.T1,
        path=config.path,
        wobble=config.wobble,
        probe=config.probe,
    )

    if config.output_file is not None:
        os.makedirs(os.path.dirname(config.output_file) or ".", exist_ok=True)
        save_temperature_csv(config.output_file, x, y, T)

    result = {"x": x, "y": y, "T": T}
    if T_probe is not None:
        result["t"] = np.arange(0, config.t_end, config.dt)
        result["T_probe"] = T_probe
    return result


def save_temperature_csv(path: str, x: np.ndarray, y: np.ndarray, T: np.ndarray):
    """Save temperature field as a simple CSV (flattened grid)."""
    nx, ny = T.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("x_m,y_m,T_K\n")
        for i in range(nx):
            for j in range(ny):
                f.write(f"{x[i]:.6e},{y[j]:.6e},{T[i, j]:.3f}\n")


__all__ = [
    "ThermalSimulationConfig",
    "run_thermal_simulation",
    "validate_config",
    "save_temperature_csv",
    "ValidationError",
    "WeldParams",
    "MaterialParams",
    "Material",
    "list_materials",
    "load_material",
    "WeldPath",
    "WobbleParams",
]
