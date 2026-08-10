"""Weld path and wobble calculator.

Provides:
- WeldPath: a polyline weld path with speed per segment.
- WobbleParams: laser beam wobble (circular, figure-8, infinity, line).
- beam_at_time: the wobbled beam centre at any time t.
- heat_signature: a 2D map of absorbed energy density over the plate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class WeldPath:
    """Polyline weld path. For now a single straight segment."""

    start: Tuple[float, float]
    end: Tuple[float, float]
    speed: float  # m/s

    @property
    def length(self) -> float:
        """Path length (m)."""
        x0, y0 = self.start
        x1, y1 = self.end
        return math.hypot(x1 - x0, y1 - y0)

    @property
    def duration(self) -> float:
        """Travel time from start to end (s)."""
        return self.length / self.speed

    def nominal_position(self, t: float) -> Tuple[float, float]:
        """Beam centre without wobble at time t (m)."""
        if t <= 0:
            return self.start
        if t >= self.duration:
            return self.end
        x0, y0 = self.start
        x1, y1 = self.end
        s = t / self.duration
        return (x0 + s * (x1 - x0), y0 + s * (y1 - y0))

    def tangent(self, t: float) -> Tuple[float, float]:
        """Unit tangent vector at time t."""
        x0, y0 = self.start
        x1, y1 = self.end
        L = self.length
        if L == 0:
            return (1.0, 0.0)
        return ((x1 - x0) / L, (y1 - y0) / L)

    def normal(self, t: float) -> Tuple[float, float]:
        """Unit normal vector (perpendicular to tangent)."""
        tx, ty = self.tangent(t)
        return (-ty, tx)


@dataclass
class WobbleParams:
    """Laser beam wobble parameters."""

    amplitude: float  # m
    frequency: float  # Hz
    pattern: str = "circle"  # circle, line, figure8, infinity
    phase: float = 0.0  # rad

    @property
    def amplitude_mm(self) -> float:
        return self.amplitude * 1000.0


def _circle_offset(amp: float, theta: float) -> Tuple[float, float]:
    """Return (u, v) offset for a circular wobble in (tangent, normal) coords."""
    return amp * math.cos(theta), amp * math.sin(theta)


def _line_offset(amp: float, theta: float) -> Tuple[float, float]:
    """Pure transverse sinusoidal wobble."""
    return 0.0, amp * math.sin(theta)


def _figure8_offset(amp: float, theta: float) -> Tuple[float, float]:
    """Figure-8 (Lissajous with 2:1 frequency)."""
    return amp * math.sin(theta), amp * math.sin(2.0 * theta)


def _infinity_offset(amp: float, theta: float) -> Tuple[float, float]:
    """Lemniscate/infinity pattern."""
    denom = 1.0 + math.sin(theta) ** 2
    u = amp * math.cos(theta) / (denom + 1e-12)
    v = amp * math.sin(theta) * math.cos(theta) / (denom + 1e-12)
    return u, v


_PATTERN_FUNCS = {
    "circle": _circle_offset,
    "line": _line_offset,
    "sine": _line_offset,
    "figure8": _figure8_offset,
    "figure_8": _figure8_offset,
    "infinity": _infinity_offset,
    "lemniscate": _infinity_offset,
}


def beam_at_time(
    path: WeldPath,
    wobble: WobbleParams,
    t: float,
) -> Tuple[float, float]:
    """Beam centre (x, y) at time t, including wobble (m)."""
    x0, y0 = path.nominal_position(t)
    theta = 2.0 * math.pi * wobble.frequency * t + wobble.phase

    func = _PATTERN_FUNCS.get(wobble.pattern, _circle_offset)
    u, v = func(wobble.amplitude, theta)

    tx, ty = path.tangent(t)
    nx, ny = path.normal(t)

    x = x0 + u * tx + v * nx
    y = y0 + u * ty + v * ny
    return x, y


def heat_source_at_point(
    x: float,
    y: float,
    t: float,
    path: WeldPath,
    wobble: WobbleParams,
    power: float,
    efficiency: float,
    sigma: float,
    h: float,
) -> float:
    """Volumetric heat source [W/m³] at (x, y, t) for a wobbled moving Gaussian."""
    x_src, y_src = beam_at_time(path, wobble, t)
    r2 = (x - x_src) ** 2 + (y - y_src) ** 2
    q_eff = power * efficiency
    q_surf = (q_eff / (2.0 * math.pi * sigma**2)) * math.exp(-r2 / (2.0 * sigma**2))
    return q_surf / h


def heat_signature(
    path: WeldPath,
    wobble: WobbleParams,
    power: float,
    efficiency: float,
    sigma: float,
    h: float,
    x: np.ndarray,
    y: np.ndarray,
    t_end: float,
    dt: float = 0.002,
) -> np.ndarray:
    """Accumulated heat input per unit volume [J/m³] over the path.

    This is the time-integral of the surface Gaussian spread over thickness h.
    Useful for visualising the beam track before running the full thermal sim.
    """
    X, Y = np.meshgrid(x, y, indexing="ij")
    Q = np.zeros_like(X)
    q_eff = power * efficiency
    denom = 2.0 * math.pi * sigma**2
    two_sigma2 = 2.0 * sigma**2

    t = 0.0
    while t < t_end:
        x_src, y_src = beam_at_time(path, wobble, t)
        r2 = (X - x_src) ** 2 + (Y - y_src) ** 2
        q_surf = (q_eff / denom) * np.exp(-r2 / two_sigma2)
        Q += q_surf * dt / h
        t += dt

    return Q


def beam_trajectory(
    path: WeldPath,
    wobble: WobbleParams,
    t_end: float,
    dt: float = 0.002,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_traj, y_traj) arrays for the wobbled beam centre."""
    ts = np.arange(0.0, t_end, dt)
    x_traj = np.zeros_like(ts)
    y_traj = np.zeros_like(ts)
    for i, t in enumerate(ts):
        x_traj[i], y_traj[i] = beam_at_time(path, wobble, t)
    return x_traj, y_traj
