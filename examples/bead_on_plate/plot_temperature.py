"""Plot 2D temperature field from CSV output."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_temperature_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load temperature CSV and reconstruct 2D grid."""
    xs = []
    ys = []
    temps = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            xs.append(float(row["x_m"]))
            ys.append(float(row["y_m"]))
            temps.append(float(row["T_K"]))

    x = np.array(xs)
    y = np.array(ys)
    T = np.array(temps)

    # Infer grid shape
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    T = T.reshape((nx, ny))

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    return x_unique, y_unique, T


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="results/temperature.csv",
        help="Input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/temperature.png",
        help="Output PNG file",
    )
    args = parser.parse_args()

    x, y, T = load_temperature_csv(args.input)

    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots()
    cmap = ax.pcolormesh(X * 1e3, Y * 1e3, T, shading="auto", cmap="inferno")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperature field (K)")
    fig.colorbar(cmap, ax=ax, label="Temperature (K)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
