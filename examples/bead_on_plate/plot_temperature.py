"""Plot 2D temperature field from CSV output."""

from __future__ import annotations

import argparse

import numpy as np
import matplotlib.pyplot as plt

from weldsim.io import (
    DEFAULT_TEMPERATURE_CSV,
    DEFAULT_TEMPERATURE_PNG,
    ensure_parent_dir,
    load_temperature_csv,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_TEMPERATURE_CSV,
        help="Input CSV file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TEMPERATURE_PNG,
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

    ensure_parent_dir(args.output)
    fig.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")


if __name__ == "__main__":
    main()
