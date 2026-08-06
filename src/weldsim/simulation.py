"""High-level simulation API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np

from .types import WeldParams, MaterialParams
from .thermal.fd_solver import run_2d_fd_thermal as _run_fd


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
    material: MaterialParams = field(default_factory=MaterialParams)
    output_file: str | None = "results/temperature.csv"
    T1: float = 0.005  # effective thickness (m)


def run_thermal_simulation(config: ThermalSimulationConfig):
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

    x, y, T = _run_fd(
        nx=config.nx,
        ny=config.ny,
        Lx=config.Lx,
        Ly=config.Ly,
        t_end=config.t_end,
        dt=config.dt,
        weld=config.weld,
        material=config.material,
        T0=config.material.T0,
        h=config.T1,   # pass T1 (m) into solver as h
    )

    if config.output_file is not None:
        import os
        os.makedirs(os.path.dirname(config.output_file) or ".", exist_ok=True)
        save_temperature_csv(config.output_file, x, y, T)

    return {"x": x, "y": y, "T": T}


def save_temperature_csv(path: str, x: np.ndarray, y: np.ndarray, T: np.ndarray):
    """Save temperature field as a simple CSV (flattened grid)."""
    nx, ny = T.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("x_m,y_m,T_K\n")
        for i in range(nx):
            for j in range(ny):
                f.write(f"{x[i]:.6e},{y[j]:.6e},{T[i, j]:.3f}\n")