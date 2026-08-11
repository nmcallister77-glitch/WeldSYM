"""Command-line interface for Weld Sim."""

from __future__ import annotations

import argparse

from .simulation import (
    MaterialParams,
    ThermalSimulationConfig,
    run_thermal_simulation,
)
from .types import (
    DEFAULT_EFFICIENCY,
    DEFAULT_POWER,
    DEFAULT_SPEED,
    default_weld_params,
)

_DEFAULTS = ThermalSimulationConfig()

_ARGUMENTS = (
    ("--power", float, DEFAULT_POWER, "Power (W)"),
    ("--efficiency", float, DEFAULT_EFFICIENCY, "Process efficiency"),
    ("--speed", float, DEFAULT_SPEED, "Travel speed (m/s)"),
    ("--t-end", float, _DEFAULTS.t_end, "Simulation time (s)"),
    ("--dt", float, _DEFAULTS.dt, "Time step (s)"),
    ("--nx", int, _DEFAULTS.nx, "Grid points in X"),
    ("--ny", int, _DEFAULTS.ny, "Grid points in Y"),
    ("--Lx", float, _DEFAULTS.Lx, "Plate length (m)"),
    ("--Ly", float, _DEFAULTS.Ly, "Plate width (m)"),
    ("--output", str, _DEFAULTS.output_file, "Output CSV file"),
)


def main():
    parser = argparse.ArgumentParser(
        prog="weldsim",
        description="Run a simple 2D welding thermal simulation.",
    )
    for flag, arg_type, default, help_text in _ARGUMENTS:
        parser.add_argument(flag, type=arg_type, default=default, help=help_text)

    args = parser.parse_args()

    config = ThermalSimulationConfig(
        nx=args.nx,
        ny=args.ny,
        Lx=args.Lx,
        Ly=args.Ly,
        t_end=args.t_end,
        dt=args.dt,
        weld=default_weld_params(
            args.Ly,
            power=args.power,
            efficiency=args.efficiency,
            speed=args.speed,
        ),
        material=MaterialParams(),
        output_file=args.output,
    )

    print("Running 2D thermal simulation...")
    result = run_thermal_simulation(config)
    print(f"Simulation complete. Output: {args.output}")
    print(f"Temperature range: {result['T'].min():.1f} K – {result['T'].max():.1f} K")


if __name__ == "__main__":
    main()
