"""Shared heat source geometry and Gaussian heat input model."""

from __future__ import annotations

import numpy as np

from .types import WeldParams

# Artificial thickness (m) used to turn a surface flux into a volumetric source.
SOURCE_THICKNESS = 0.001


def heat_source_position(weld: WeldParams, t: float) -> tuple[float, float]:
    """Position (m) of the moving heat source at time ``t`` (s)."""
    if weld.direction == "x":
        return weld.start_pos[0] + weld.speed * t, weld.start_pos[1]
    if weld.direction == "y":
        return weld.start_pos[0], weld.start_pos[1] + weld.speed * t
    raise ValueError("direction must be 'x' or 'y'")


def heat_source_at_point(x: float, y: float, t: float, weld: WeldParams) -> float:
    """
    Evaluate heat source (W/m^3) at a single (x, y, t) point.
    Uses a simple 2D Gaussian in the plane; z is assumed 0.
    """
    x_src, y_src = heat_source_position(weld, t)
    r2 = (x - x_src) ** 2 + (y - y_src) ** 2
    return float(_gaussian(r2, weld))


def heat_source_field(x: np.ndarray, y: np.ndarray, t: float, weld: WeldParams) -> np.ndarray:
    """Evaluate the heat source (W/m^3) on the grid spanned by ``x`` and ``y``."""
    x_src, y_src = heat_source_position(weld, t)
    r2 = (x[:, None] - x_src) ** 2 + (y[None, :] - y_src) ** 2
    return _gaussian(r2, weld)


def _gaussian(r2, weld: WeldParams):
    """Gaussian source intensity for squared radial distances ``r2``."""
    q_eff = weld.power * weld.efficiency
    sigma = weld.sigma
    amplitude = q_eff / (2 * np.pi * sigma**2 * SOURCE_THICKNESS)
    return amplitude * np.exp(-r2 / (2 * sigma**2))
