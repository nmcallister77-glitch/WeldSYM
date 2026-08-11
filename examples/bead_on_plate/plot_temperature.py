"""Plot 2D temperature field from CSV output."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from weldsim.exceptions import InputDataError, WeldSimError

REQUIRED_COLUMNS = ("x_m", "y_m", "T_K")


def load_temperature_csv(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load temperature CSV and reconstruct 2D grid."""
    xs = []
    ys = []
    temps = []

    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise InputDataError(f"Could not open temperature CSV {path!r}: {exc}") from exc

    with handle as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise InputDataError(f"Temperature CSV {path!r} is empty.")
        missing = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise InputDataError(
                f"Temperature CSV {path!r} is missing column(s) {', '.join(missing)}; "
                f"found {', '.join(reader.fieldnames)}."
            )
        for line_no, row in enumerate(reader, start=2):
            try:
                xs.append(float(row["x_m"]))
                ys.append(float(row["y_m"]))
                temps.append(float(row["T_K"]))
            except (TypeError, ValueError) as exc:
                raise InputDataError(
                    f"{path}:{line_no}: could not parse row {row!r}: {exc}"
                ) from exc

    if not temps:
        raise InputDataError(f"Temperature CSV {path!r} contains no data rows.")

    x = np.array(xs)
    y = np.array(ys)
    T = np.array(temps)

    # Infer grid shape
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    if nx * ny != T.size:
        raise InputDataError(
            f"Temperature CSV {path!r} is not a complete rectangular grid: "
            f"{T.size} rows cannot be reshaped to {nx} x {ny}."
        )
    T = T.reshape((nx, ny))

    x_unique = np.sort(np.unique(x))
    y_unique = np.sort(np.unique(y))

    return x_unique, y_unique, T


def build_parser() -> argparse.ArgumentParser:
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        x, y, T = load_temperature_csv(args.input)
    except WeldSimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    X, Y = np.meshgrid(x, y, indexing="ij")

    fig, ax = plt.subplots()
    cmap = ax.pcolormesh(X * 1e3, Y * 1e3, T, shading="auto", cmap="inferno")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperature field (K)")
    fig.colorbar(cmap, ax=ax, label="Temperature (K)")

    try:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=150)
    except OSError as exc:
        print(f"error: could not write plot to {args.output!r}: {exc}", file=sys.stderr)
        return 1
    finally:
        plt.close(fig)

    print(f"Plot saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
