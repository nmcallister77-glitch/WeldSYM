"""High-level simulation API."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

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
    max_cells: int = 4_000_000
    max_steps: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value in (("nx", self.nx), ("ny", self.ny)):
            if value < 3:
                raise ValueError(f"{name} must be at least 3 (got {value!r})")
        for name, value in (
            ("Lx", self.Lx),
            ("Ly", self.Ly),
            ("t_end", self.t_end),
            ("dt", self.dt),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite, positive value (got {value!r})")
        if self.dt > self.t_end:
            raise ValueError(f"dt ({self.dt}) must not exceed t_end ({self.t_end})")
        if self.nx * self.ny > self.max_cells:
            raise ValueError(
                f"grid of {self.nx}x{self.ny} cells exceeds max_cells ({self.max_cells})"
            )
        n_steps = math.ceil(self.t_end / self.dt)
        if n_steps > self.max_steps:
            raise ValueError(
                f"{n_steps} time steps exceed max_steps ({self.max_steps}); increase dt"
            )


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
    )

    if config.output_file is not None:
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
