"""Weld path and wobble calculator.

Provides:
- WeldPath: a polyline weld path with speed per segment.
- WobbleParams: laser beam wobble (circle, figure-8, infinity, line).
- beam_at_time: the wobbled beam centre at any time t.
- heat_signature: a 2D map of absorbed energy density over the plate.
- wobble_animation_gif: an animated GIF of the wobbled beam and growing heat signature.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class WeldPath:
    """Polyline weld path; supports straight segments, arcs and spirals."""

    start: Tuple[float, float]
    end: Tuple[float, float]
    speed: float  # m/s
    points: List[Tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.points:
            self.points = [self.start, self.end]
        else:
            self.start = self.points[0]
            self.end = self.points[-1]

    @property
    def length(self) -> float:
        """Total path length (m)."""
        total = 0.0
        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    @property
    def duration(self) -> float:
        """Travel time from start to end (s)."""
        if self.speed <= 0:
            return 0.0
        return self.length / self.speed

    def _segment_at(self, t: float) -> Tuple[int, float]:
        """Return (segment index, interpolation fraction) for time t."""
        if t <= 0:
            return 0, 0.0
        if t >= self.duration:
            return max(0, len(self.points) - 2), 1.0
        dist = t * self.speed
        cum = 0.0
        for i in range(len(self.points) - 1):
            x0, y0 = self.points[i]
            x1, y1 = self.points[i + 1]
            seg_len = math.hypot(x1 - x0, y1 - y0)
            if seg_len == 0:
                continue
            if cum + seg_len >= dist:
                return i, (dist - cum) / seg_len
            cum += seg_len
        return max(0, len(self.points) - 2), 1.0

    def nominal_position(self, t: float) -> Tuple[float, float]:
        """Beam centre without wobble at time t (m)."""
        if t <= 0:
            return self.start
        if t >= self.duration:
            return self.end
        i, frac = self._segment_at(t)
        x0, y0 = self.points[i]
        x1, y1 = self.points[i + 1]
        return (x0 + frac * (x1 - x0), y0 + frac * (y1 - y0))

    def tangent(self, t: float) -> Tuple[float, float]:
        """Unit tangent vector at time t."""
        i, _ = self._segment_at(t)
        if i >= len(self.points) - 1:
            i = max(0, len(self.points) - 2)
        x0, y0 = self.points[i]
        x1, y1 = self.points[i + 1]
        L = math.hypot(x1 - x0, y1 - y0)
        if L == 0:
            return (1.0, 0.0)
        return ((x1 - x0) / L, (y1 - y0) / L)

    def normal(self, t: float) -> Tuple[float, float]:
        """Unit normal vector (perpendicular to tangent)."""
        tx, ty = self.tangent(t)
        return (-ty, tx)

    @classmethod
    def circle(
        cls,
        center: Tuple[float, float],
        radius: float,
        speed: float,
        turns: float = 1.0,
        n_points: int = 128,
        start_angle: float = 0.0,
    ) -> "WeldPath":
        """Circular weld path. ``turns=1`` is one full revolution."""
        points: List[Tuple[float, float]] = []
        total_angle = 2.0 * math.pi * turns
        for k in range(n_points + 1):
            angle = start_angle + total_angle * k / n_points
            x = float(center[0] + radius * math.cos(angle))
            y = float(center[1] + radius * math.sin(angle))
            points.append((x, y))
        return cls(start=points[0], end=points[-1], speed=speed, points=points)

    @classmethod
    def spiral(
        cls,
        center: Tuple[float, float],
        r_start: float,
        r_end: float,
        speed: float,
        turns: float = 2.0,
        n_points: int = 256,
    ) -> "WeldPath":
        """Archimedean spiral weld path."""
        points: List[Tuple[float, float]] = []
        for k in range(n_points + 1):
            angle = 2.0 * math.pi * turns * k / n_points
            r = r_start + (r_end - r_start) * k / n_points
            x = float(center[0] + r * math.cos(angle))
            y = float(center[1] + r * math.sin(angle))
            points.append((x, y))
        return cls(start=points[0], end=points[-1], speed=speed, points=points)


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
    t_start: float = 0.0,
) -> np.ndarray:
    """Accumulated heat input per unit volume [J/m³] over the path.

    This is the time-integral of the surface Gaussian spread over thickness h,
    from ``t_start`` to ``t_end``. Useful for visualising the beam track before
    running the full thermal sim.
    """
    if t_end > path.duration:
        t_end = path.duration
    X, Y = np.meshgrid(x, y, indexing="ij")
    Q = np.zeros_like(X)
    q_eff = power * efficiency
    denom = 2.0 * math.pi * sigma**2
    two_sigma2 = 2.0 * sigma**2

    t = t_start
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
    t_start: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (x_traj, y_traj) arrays for the wobbled beam centre."""
    ts = np.arange(t_start, t_end, dt)
    x_traj = np.zeros_like(ts)
    y_traj = np.zeros_like(ts)
    for i, t in enumerate(ts):
        x_traj[i], y_traj[i] = beam_at_time(path, wobble, t)
    return x_traj, y_traj


def wobble_animation_gif(
    path: WeldPath,
    wobble: WobbleParams,
    power: float,
    efficiency: float,
    sigma: float,
    h: float,
    x: np.ndarray,
    y: np.ndarray,
    t_end: float,
    frame_dt: float = 0.05,
    trail_time: float = 0.05,
    heat_dt: float = 0.002,
    fps: int = 15,
    gif_width: int = 400,
) -> bytes:
    """Generate a GIF animation of the wobbled weld being made.

    The GIF shows the beam moving along the path (with the recent wobble trail
    visible) and the heat signature growing in the background.  Brighter areas
    in the heat signature show where the beam has spent the most time / energy.

    Returns
    -------
    gif_bytes : bytes
        GIF data suitable for ``st.image`` or writing to a file.
    """
    import io

    import matplotlib
    from PIL import Image, ImageDraw

    hot = matplotlib.colormaps["hot"]

    # Downsample the simulation grid to a fixed image size
    gif_height = int(round(gif_width * (y[-1] - y[0]) / (x[-1] - x[0])))

    X, Y = np.meshgrid(x, y, indexing="ij")
    Q = np.zeros_like(X)
    q_eff = power * efficiency
    denom = 2.0 * math.pi * sigma**2
    two_sigma2 = 2.0 * sigma**2

    # Full trajectory at fine resolution for smooth wobble trail
    ts_full = np.arange(0.0, t_end, heat_dt)
    x_full = np.zeros_like(ts_full)
    y_full = np.zeros_like(ts_full)
    for i, t in enumerate(ts_full):
        x_full[i], y_full[i] = beam_at_time(path, wobble, t)

    # Animation frame times
    frame_times = np.arange(0.0, t_end + frame_dt, frame_dt)
    frame_times[-1] = min(frame_times[-1], t_end)

    # Precompute cumulative heat up to each frame time
    Q_frames = []
    t = 0.0
    Q = np.zeros_like(X)
    frame_idx = 0
    while t < t_end and frame_idx < len(frame_times):
        while frame_idx < len(frame_times) and t >= frame_times[frame_idx]:
            Q_frames.append(Q.copy())
            frame_idx += 1
        x_src, y_src = beam_at_time(path, wobble, t)
        r2 = (X - x_src) ** 2 + (Y - y_src) ** 2
        q_surf = (q_eff / denom) * np.exp(-r2 / two_sigma2)
        Q += q_surf * heat_dt / h
        t += heat_dt
    while frame_idx < len(frame_times):
        Q_frames.append(Q.copy())
        frame_idx += 1

    # Global log-scaled colormap range based on final accumulated heat
    Qmax = max(Q.max(), 1e-9)
    Qmin = Q[Q > 0].min() if Q.max() > 0 else 1e-9
    log_min = math.log10(Qmin)
    log_max = math.log10(Qmax)
    log_range = max(log_max - log_min, 1e-6)

    def _to_rgba(q: np.ndarray) -> np.ndarray:
        q = q.T  # (ny, nx)
        # y=0 at bottom
        q = np.flipud(q)
        q[q <= 0] = 1e-12
        log_q = np.log10(q)
        norm = (log_q - log_min) / log_range
        norm = np.clip(norm, 0.0, 1.0)
        return (hot(norm) * 255).astype(np.uint8)

    # Coordinate helpers for drawing on the downsampled image
    def _px(x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(round((x_m - x[0]) / (x[-1] - x[0]) * (gif_width - 1)))
        py = int(round((y_m - y[0]) / (y[-1] - y[0]) * (gif_height - 1)))
        # PIL y is from top, but our image is already flipud, so y=0 is bottom
        py = (gif_height - 1) - py
        return max(0, min(px, gif_width - 1)), max(0, min(py, gif_height - 1))

    frames: List[Image.Image] = []
    for k, t_now in enumerate(frame_times):
        rgba = _to_rgba(Q_frames[k])
        img = Image.fromarray(rgba, mode="RGBA").convert("RGB")
        if img.width != gif_width or img.height != gif_height:
            img = img.resize((gif_width, gif_height), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(img)

        # Recent wobble trail
        mask = (ts_full >= t_now - trail_time) & (ts_full <= t_now)
        if mask.any():
            pts = [_px(xf, yf) for xf, yf in zip(x_full[mask], y_full[mask])]
            if len(pts) > 1:
                draw.line(pts, fill=(0, 255, 255), width=1)
            x_now, y_now = x_full[mask][-1], y_full[mask][-1]
        else:
            x_now, y_now = beam_at_time(path, wobble, t_now)

        px, py = _px(x_now, y_now)
        r = 4
        draw.ellipse([px - r, py - r, px + r, py + r], fill=(0, 255, 255))

        # Start / end markers
        s1, s2 = _px(*path.start), _px(*path.end)
        draw.ellipse([s1[0] - 2, s1[1] - 2, s1[0] + 2, s1[1] + 2], fill=(0, 255, 0))
        draw.ellipse([s2[0] - 2, s2[1] - 2, s2[0] + 2, s2[1] + 2], fill=(0, 0, 255))

        frames.append(img)

    buf = io.BytesIO()
    frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
        optimize=True,
    )
    return buf.getvalue()
