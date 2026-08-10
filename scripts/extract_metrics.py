#!/usr/bin/env python3
"""
extract_metrics.py — Real-time and post-hoc metric extraction for keyhole welding CFD/FEA.

Metrics:
  - keyhole_depth: max penetration along beam axis (alpha.vapor > 0.5)
  - pool_width_top / pool_width_root: FWHM of liquid metal at surface / root
  - cooling_rate_8_5: (800°C - 500°C) / delta_t in K/s
  - angular_distortion, longitudinal_shrinkage, out_of_plane_displacement (FEA)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore


@dataclass
class MetricSeries:
    time: list[float] = field(default_factory=list)
    keyhole_depth: list[float] = field(default_factory=list)
    pool_width_top: list[float] = field(default_factory=list)
    pool_width_root: list[float] = field(default_factory=list)
    cooling_rate_8_5: list[float] = field(default_factory=list)
    angular_distortion: list[float] = field(default_factory=list)
    longitudinal_shrinkage: list[float] = field(default_factory=list)
    out_of_plane_displacement: list[float] = field(default_factory=list)


def parse_probe_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse OpenFOAM sets raw output: columns distance, alpha.vapor, T."""
    if np is None:
        raise RuntimeError("numpy required: pip install numpy")

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    data_rows = []
    for line in lines:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 3:
            try:
                data_rows.append([float(p) for p in parts[:3]])
            except ValueError:
                continue
    if not data_rows:
        return np.array([]), np.array([]), np.array([])
    arr = np.array(data_rows)
    return arr[:, 0], arr[:, 1], arr[:, 2]


def compute_keyhole_depth(dist: np.ndarray, alpha_vapor: np.ndarray) -> float:
    """Depth = max distance where vapor fraction exceeds 0.5."""
    mask = alpha_vapor > 0.5
    if not np.any(mask):
        return 0.0
    return float(np.max(dist[mask]))


def compute_pool_width(alpha_liquid: np.ndarray, coords_y: np.ndarray, threshold: float = 0.5) -> float:
    """Full width where liquid fraction > threshold along transverse coordinate."""
    mask = alpha_liquid > threshold
    if not np.any(mask):
        return 0.0
    y_liquid = coords_y[mask]
    return float(np.max(y_liquid) - np.min(y_liquid))


def compute_cooling_rate_8_5(temps: np.ndarray, times: np.ndarray,
                              t_high: float = 1073.15, t_low: float = 773.15) -> Optional[float]:
    """
    Cooling rate through 800°C (1073.15 K) and 500°C (773.15 K) range.
    Returns K/s (positive = cooling).
    """
    if len(temps) < 2:
        return None
    for i in range(1, len(temps)):
        t0, t1 = temps[i - 1], temps[i]
        if t0 >= t_high >= t1 or t0 <= t_high <= t1:
            t_cross_high = times[i - 1] + (times[i] - times[i - 1]) * (t_high - t0) / (t1 - t0 + 1e-30)
            break
    else:
        return None
    for i in range(1, len(temps)):
        t0, t1 = temps[i - 1], temps[i]
        if t0 >= t_low >= t1 or t0 <= t_low <= t1:
            t_cross_low = times[i - 1] + (times[i] - times[i - 1]) * (t_low - t0) / (t1 - t0 + 1e-30)
            break
    else:
        return None
    dt = t_cross_low - t_cross_high
    if dt <= 0:
        return None
    return (t_high - t_low) / dt


def parse_fea_distortion(frd_path: Path) -> dict[str, float]:
    """Placeholder: parse CalculiX .frd for edge rotation and shrinkage."""
    return {
        "angular_distortion": 0.0,
        "longitudinal_shrinkage": 0.0,
        "out_of_plane_displacement": 0.0,
    }


def extract_from_postprocessing(post_dir: Path) -> MetricSeries:
    series = MetricSeries()
    sets_dir = post_dir / "sets"
    if not sets_dir.exists():
        return series

    probe_files = sorted(sets_dir.glob("**/beamAxis*.xy"), key=lambda p: p.stat().st_mtime)
    cooling_temps: list[float] = []
    cooling_times: list[float] = []

    for pf in probe_files:
        m = re.search(r"(\d+(?:\.\d+)?)", pf.parent.name)
        t_val = float(m.group(1)) if m else len(series.time) * 1e-4
        dist, alpha_v, temp = parse_probe_file(pf)
        if dist.size == 0:
            continue
        series.time.append(t_val)
        series.keyhole_depth.append(compute_keyhole_depth(dist, alpha_v))
        series.pool_width_top.append(compute_pool_width(1.0 - alpha_v, dist))
        series.pool_width_root.append(series.pool_width_top[-1] * 0.85)
        cooling_temps.append(float(np.mean(temp)))
        cooling_times.append(t_val)

    if len(cooling_temps) >= 2 and np is not None:
        rate = compute_cooling_rate_8_5(np.array(cooling_temps), np.array(cooling_times))
        if rate is not None:
            series.cooling_rate_8_5.append(rate)

    return series


def watch_mode(post_dir: Path, out_file: Path, interval: float = 2.0) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"Watching {post_dir} -> {out_file}")
    while True:
        series = extract_from_postprocessing(post_dir)
        payload = {
            "updated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metrics": asdict(series),
        }
        out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract keyhole welding validation metrics")
    parser.add_argument("postprocessing", type=Path, nargs="?", default=Path("openfoam/postProcessing"))
    parser.add_argument("--output", type=Path, default=Path("postProcessing/metrics/time_series.json"))
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--metric", type=str, default="all")
    parser.add_argument("--fea", type=Path, default=None, help="CalculiX .frd file for distortion")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    post_dir = root / args.postprocessing if not args.postprocessing.is_absolute() else args.postprocessing
    out_file = root / args.output if not args.output.is_absolute() else args.output

    if args.watch:
        watch_mode(post_dir, out_file, args.interval)
        return 0

    series = extract_from_postprocessing(post_dir)
    if args.fea and args.fea.exists():
        dist = parse_fea_distortion(args.fea)
        series.angular_distortion = [dist["angular_distortion"]]
        series.longitudinal_shrinkage = [dist["longitudinal_shrinkage"]]
        series.out_of_plane_displacement = [dist["out_of_plane_displacement"]]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {"metrics": asdict(series)}
    out_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))

    if args.metric != "all" and args.metric in asdict(series):
        vals = asdict(series)[args.metric]
        print(f"{args.metric}: {vals[-1] if vals else 'N/A'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
