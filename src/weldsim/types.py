"""Core dataclasses for Weld Sim."""

from __future__ import annotations

import math
from dataclasses import dataclass
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

    def __post_init__(self) -> None:
        if not math.isfinite(self.power) or self.power < 0.0:
            raise ValueError(f"power must be a finite, non-negative value (got {self.power!r})")
        if not math.isfinite(self.efficiency) or not 0.0 <= self.efficiency <= 1.0:
            raise ValueError(f"efficiency must be between 0 and 1 (got {self.efficiency!r})")
        if not math.isfinite(self.speed) or self.speed < 0.0:
            raise ValueError(f"speed must be a finite, non-negative value (got {self.speed!r})")
        if not math.isfinite(self.sigma) or self.sigma <= 0.0:
            raise ValueError(f"sigma must be a finite, positive value (got {self.sigma!r})")
        if len(self.start_pos) != 2 or not all(math.isfinite(c) for c in self.start_pos):
            raise ValueError(f"start_pos must be two finite coordinates (got {self.start_pos!r})")
        if self.direction not in ("x", "y"):
            raise ValueError(f"direction must be 'x' or 'y' (got {self.direction!r})")


@dataclass
class MaterialParams:
    """Simplified material properties (constant with T for now)."""

    k: float = 50.0  # W/(m·K)
    rho: float = 7850.0  # kg/m^3
    cp: float = 500.0  # J/(kg·K)
    T0: float = 300.0  # K

    def __post_init__(self) -> None:
        for name, value in (("k", self.k), ("rho", self.rho), ("cp", self.cp)):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a finite, positive value (got {value!r})")
        if not math.isfinite(self.T0) or self.T0 < 0.0:
            raise ValueError(f"T0 must be a finite, non-negative value (got {self.T0!r})")
