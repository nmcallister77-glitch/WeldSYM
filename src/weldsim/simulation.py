"""High-level simulation API."""

from __future__ import annotations

from dataclasses import dataclass, field

from .io import DEFAULT_TEMPERATURE_CSV, save_temperature_csv
from .types import MaterialParams, WeldParams, default_weld_params
from .thermal.fd_solver import run_2d_fd_thermal as _run_fd

__all__ = [
    "MaterialParams",
    "WeldParams",
    "ThermalSimulationConfig",
    "run_thermal_simulation",
    "save_temperature_csv",
]


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
    output_file: str | None = DEFAULT_TEMPERATURE_CSV


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
        config.weld = default_weld_params(config.Ly)

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
        save_temperature_csv(config.output_file, x, y, T)

    return {"x": x, "y": y, "T": T}
