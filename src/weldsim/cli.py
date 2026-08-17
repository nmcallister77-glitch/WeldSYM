"""Command-line interface for Weld Sim."""

from __future__ import annotations

import argparse
import json
import sys

from .errors import WeldSimError
from .materials import list_materials, load_material
from .report import build_report
from .simulation import (
    WeldParams,
    MaterialParams,
    ThermalSimulationConfig,
    run_thermal_simulation,
)
from .weld_path import WeldPath, WobbleParams


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="weldsim",
        description=(
            "Run a 2D welding thermal simulation and report the weld it would "
            "produce: fusion zone and HAZ size, penetration, HAZ microstructure "
            "and hardness, distortion and residual stress."
        ),
    )
    parser.add_argument("--power", type=float, default=3000.0, help="Power (W)")
    parser.add_argument("--efficiency", type=float, default=0.8, help="Process efficiency")
    parser.add_argument("--speed", type=float, default=0.005, help="Travel speed (m/s)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Beam Gaussian sigma (mm)")
    parser.add_argument("--t-end", type=float, default=10.0, help="Simulation time (s)")
    parser.add_argument("--dt", type=float, default=0.05, help="Time step (s)")
    parser.add_argument("--nx", type=int, default=51, help="Grid points in X")
    parser.add_argument("--ny", type=int, default=26, help="Grid points in Y")
    parser.add_argument("--Lx", type=float, default=0.1, help="Plate length (m)")
    parser.add_argument("--Ly", type=float, default=0.05, help="Plate width (m)")
    parser.add_argument(
        "--output",
        type=str,
        default="results/temperature.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--thickness",
        type=float,
        default=5.0,
        help="Plate thickness (mm); also the depth the surface flux is spread over",
    )
    parser.add_argument(
        "--material",
        type=str,
        default=None,
        help=(
            "Material from the library (omit for generic steel properties). "
            f"Available: {', '.join(list_materials())}"
        ),
    )
    parser.add_argument(
        "--wobble-amplitude",
        type=float,
        default=0.0,
        help="Beam wobble amplitude (mm); 0 disables wobble",
    )
    parser.add_argument(
        "--wobble-frequency",
        type=float,
        default=0.0,
        help="Beam wobble frequency (Hz)",
    )
    parser.add_argument(
        "--wobble-pattern",
        type=str,
        default="circle",
        choices=["circle", "line", "figure8", "infinity"],
        help="Beam wobble pattern",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Also write the weld assessment to this file as JSON",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    weld = WeldParams(
        power=args.power,
        efficiency=args.efficiency,
        speed=args.speed,
        start_pos=(0.01, args.Ly / 2),
        direction="x",
        sigma=args.sigma / 1000.0,
    )

    try:
        material = load_material(args.material) if args.material else MaterialParams()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    # Weld the length of the plate, less a margin at each end, so the thermal
    # field is not clipped by the fixed-temperature boundaries.
    path = WeldPath(
        start=(0.01, args.Ly / 2),
        end=(max(args.Lx - 0.01, 0.01), args.Ly / 2),
        speed=args.speed,
    )
    wobble = None
    if args.wobble_amplitude > 0 and args.wobble_frequency > 0:
        wobble = WobbleParams(
            amplitude=args.wobble_amplitude / 1000.0,
            frequency=args.wobble_frequency,
            pattern=args.wobble_pattern,
        )

    thickness = args.thickness / 1000.0
    config = ThermalSimulationConfig(
        nx=args.nx,
        ny=args.ny,
        Lx=args.Lx,
        Ly=args.Ly,
        t_end=args.t_end,
        dt=args.dt,
        weld=weld,
        material=material,
        output_file=args.output,
        T1=thickness,
        plate_thickness=thickness,
        path=path,
        wobble=wobble,
    )

    print("Running 2D thermal simulation...")
    try:
        result = run_thermal_simulation(config)
        report = build_report(config, result)
    except WeldSimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"Simulation complete. Output: {args.output}")
    print(f"Temperature range: {result['T'].min():.1f} K - {result['T'].max():.1f} K")
    print()
    for line in report.summary_lines():
        print(line)

    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report.as_dict(), f, indent=2)
        print(f"\nWeld assessment written to {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
