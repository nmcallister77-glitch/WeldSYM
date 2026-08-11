"""High-level simulation API."""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field

import numpy as np

from .exceptions import ConfigurationError, OutputError
from .types import WeldParams, MaterialParams
from .thermal.fd_solver import run_2d_fd_thermal as _run_fd

logger = logging.getLogger(__name__)


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
    if config.output_file is not None and not str(config.output_file).strip():
        raise ConfigurationError("output_file must be a non-empty path or None.")

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
        directory = os.path.dirname(config.output_file) or "."
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            raise OutputError(f"Could not create output directory {directory!r}: {exc}") from exc
        save_temperature_csv(config.output_file, x, y, T)

    return {"x": x, "y": y, "T": T}


def save_temperature_csv(path: str, x: np.ndarray, y: np.ndarray, T: np.ndarray):
    """
    Save temperature field as a simple CSV (flattened grid).

    The file is written to a temporary file in the same directory and moved into
    place afterwards, so a failure never leaves a truncated CSV behind.
    """
    if T.ndim != 2:
        raise ConfigurationError(f"T must be a 2D array (got shape {T.shape}).")
    nx, ny = T.shape
    if x.shape != (nx,) or y.shape != (ny,):
        raise ConfigurationError(
            f"Coordinate arrays do not match T of shape {T.shape} (x: {x.shape}, y: {y.shape})."
        )

    directory = os.path.dirname(path) or "."
    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            prefix=os.path.basename(path) + ".", suffix=".tmp", dir=directory
        )
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("x_m,y_m,T_K\n")
            for i in range(nx):
                for j in range(ny):
                    f.write(f"{x[i]:.6e},{y[j]:.6e},{T[i, j]:.3f}\n")
        os.replace(tmp_path, path)
        tmp_path = None
    except OSError as exc:
        raise OutputError(f"Could not write temperature CSV to {path!r}: {exc}") from exc
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError as cleanup_exc:
                logger.warning("Could not remove temporary file %s: %s", tmp_path, cleanup_exc)
