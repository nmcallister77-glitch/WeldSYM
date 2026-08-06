"""Command-line interface for Weld Sim."""

from __future__ import annotations

import argparse

from .simulation import (
    WeldParams,
    MaterialParams,
    ThermalSimulationConfig,
    run_thermal_simulation,
)


def main():
    parser = argparse.ArgumentParser(
        prog="weldsim",
        description="Run a simple 2D welding thermal simulation.",
    )
    parser.add_argument("--power", type=float, default=3000.0, help="Power (W)")
    parser.add_argument(
        "--efficiency", type=float, default=0.8, help="Process efficiency"
    )
    parser.add_argument(
        "--speed", type=float, default=0.005, help="Travel speed (m/s)"
    )
    parser.add_argument(
        "--t-end", type=float, default=10.0, help="Simulation time (s)"
    )
    parser.add_argument("--dt", type=float, default=0.05, help="Time step (s)")
    parser.add_argument(
        "--nx", type=int, default=51, help="Grid points in X"
    )
    parser.add_argument(
        "--ny", type=int, default=26, help="Grid points in Y"
    )
    parser.add_argument(
        "--Lx", type=float, default=0.1, help="Plate length (m)"
    )
    parser.add_argument(
        "--Ly", type=float, default=0.05, help="Plate width (m)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/temperature.csv",
        help="Output CSV file",
    )

    args = parser.parse_args()

    weld = WeldParams(
        power=args.power,
        efficiency=args.efficiency,
        speed=args.speed,
        start_pos=(0.01, args.Ly / 2),
        direction="x",
        sigma=0.002,
    )

    mat = MaterialParams()

    config = ThermalSimulationConfig(
        nx=args.nx,
        ny=args.ny,
        Lx=args.Lx,
        Ly=args.Ly,
        t_end=args.t_end,
        dt=args.dt,
        weld=weld,
        material=mat,
        output_file=args.output,
    )

    print("Running 2D thermal simulation...")
    result = run_thermal_simulation(config)
    print(f"Simulation complete. Output: {args.output}")
    print(f"Temperature range: {result['T'].min():.1f} K – {result['T'].max():.1f} K")


if __name__ == "__main__":
    main()
