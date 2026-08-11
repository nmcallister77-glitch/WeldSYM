"""Shared file I/O helpers for temperature fields."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

X_COLUMN = "x_m"
Y_COLUMN = "y_m"
T_COLUMN = "T_K"
TEMPERATURE_COLUMNS = (X_COLUMN, Y_COLUMN, T_COLUMN)

DEFAULT_TEMPERATURE_CSV = "results/temperature.csv"
DEFAULT_TEMPERATURE_PNG = "results/temperature.png"


def ensure_parent_dir(path: str) -> None:
    """Create the parent directory of ``path`` if it does not exist."""
    Path(path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def save_temperature_csv(path: str, x: np.ndarray, y: np.ndarray, T: np.ndarray) -> None:
    """Save temperature field as a simple CSV (flattened grid)."""
    ensure_parent_dir(path)
    nx, ny = T.shape
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(TEMPERATURE_COLUMNS)
        for i in range(nx):
            for j in range(ny):
                writer.writerow([f"{x[i]:.6e}", f"{y[j]:.6e}", f"{T[i, j]:.3f}"])


def load_temperature_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a temperature CSV written by :func:`save_temperature_csv`."""
    xs: list[float] = []
    ys: list[float] = []
    temps: list[float] = []

    with open(path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            xs.append(float(row[X_COLUMN]))
            ys.append(float(row[Y_COLUMN]))
            temps.append(float(row[T_COLUMN]))

    x_unique = np.unique(np.array(xs))
    y_unique = np.unique(np.array(ys))
    T = np.array(temps).reshape((len(x_unique), len(y_unique)))

    return x_unique, y_unique, T
