"""Core dataclasses for Weld Sim."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

DEFAULT_POWER = 3000.0
DEFAULT_EFFICIENCY = 0.8
DEFAULT_SPEED = 0.005  # m/s
DEFAULT_START_X = 0.01  # m
DEFAULT_SIGMA = 0.002  # m


@dataclass
class WeldParams:
    """Welding process parameters."""

    power: float  # W
    efficiency: float  # 0–1
    speed: float  # m/s
    start_pos: Tuple[float, float]  # (x, y) in meters
    direction: str = "x"  # 'x' or 'y'
    heat_source_type: str = "goldak"  # reserved for future
    goldak_params: Dict[str, float] | None = None
    sigma: float = DEFAULT_SIGMA  # for Gaussian


@dataclass
class MaterialParams:
    """Simplified material properties (constant with T for now)."""

    k: float = 50.0  # W/(m·K)
    rho: float = 7850.0  # kg/m^3
    cp: float = 500.0  # J/(kg·K)
    T0: float = 300.0  # K


def default_weld_params(
    Ly: float,
    power: float = DEFAULT_POWER,
    efficiency: float = DEFAULT_EFFICIENCY,
    speed: float = DEFAULT_SPEED,
) -> WeldParams:
    """Weld parameters for a bead deposited along x at mid-width of the plate."""
    return WeldParams(
        power=power,
        efficiency=efficiency,
        speed=speed,
        start_pos=(DEFAULT_START_X, Ly / 2),
        direction="x",
    )
