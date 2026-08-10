"""High-level simulation API."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .materials import Material, list_materials, load_material
from .thermal.fd_solver import run_2d_fd_thermal
from .types import MaterialParams, WeldParams
from .weld_path import WeldPath, WobbleParams


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
    "save_temperature_csv",
    "WeldParams",
    "MaterialParams",
    "Material",
    "list_materials",
    "load_material",
    "WeldPath",
    "WobbleParams",
]
