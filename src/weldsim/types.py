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
    heat_source_type: str = "gaussian"  # 'gaussian' or reserved 'goldak'
    goldak_params: Dict[str, float] | None = None
    sigma: float = 0.002  # 1/e² radius for Gaussian (m)

    @property
    def beam_diameter_m(self) -> float:
        """1/e² beam diameter (m)."""
        return 2.0 * self.sigma

    @property
    def absorbed_power(self) -> float:
        """Power actually absorbed by the workpiece (W)."""
        return self.power * self.efficiency

    @property
    def line_energy_j_per_mm(self) -> float:
        """Line energy = P / v  [J/mm]."""
        return self.absorbed_power / self.speed / 1000.0

    @property
    def spot_area_m2(self) -> float:
        """Beam spot area based on 1/e² radius (m²)."""
        import math

        return math.pi * self.sigma**2

    @property
    def power_density_w_per_m2(self) -> float:
        """Average power density over the 1/e² spot (W/m²)."""
        return self.absorbed_power / self.spot_area_m2


@dataclass
class MaterialParams:
    """Simplified material properties (constant with T for the 2D FD run)."""

    k: float = 50.0  # W/(m·K)
    rho: float = 7850.0  # kg/m^3
    cp: float = 500.0  # J/(kg·K)
    T0: float = 300.0  # K

    @classmethod
    def from_dict(cls, data: Dict[str, float]) -> "MaterialParams":
        return cls(
            k=data.get("k", 50.0),
            rho=data.get("rho", 7850.0),
            cp=data.get("cp", 500.0),
            T0=data.get("T0", 300.0),
        )
