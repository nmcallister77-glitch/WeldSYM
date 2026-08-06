"""Core dataclasses for Weld Sim."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


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
    sigma: float = 0.002  # for Gaussian


@dataclass
class MaterialParams:
    """Simplified material properties (constant with T for now)."""

    k: float = 50.0  # W/(m·K)
    rho: float = 7850.0  # kg/m^3
    cp: float = 500.0  # J/(kg·K)
    T0: float = 300.0  # K
