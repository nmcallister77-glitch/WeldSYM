#!/usr/bin/env python3
"""
configure_case.py — Generate OpenFOAM case from simulation_master.yaml
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("ERROR: PyYAML required. pip install pyyaml", file=sys.stderr)
    sys.exit(1)


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_openfoam_constants(cfg: dict, case_dir: Path, material: dict) -> None:
    """Patch thermophysicalProperties and laserHeatSource from master config."""
    const_dir = case_dir / "constant"
    const_dir.mkdir(parents=True, exist_ok=True)

    pc = material["phase_change"]
    thermo_path = const_dir / "thermophysicalProperties"
    if thermo_path.exists():
        text = thermo_path.read_text(encoding="utf-8")
        replacements = {
            "molWeight": f"molWeight       {material.get('vapor', {}).get('molar_mass', 0.0479) * 1000:.1f};",
            "Ts": f"Ts              {pc['solidus_temperature']};",
            "Tl": f"Tl              {pc['liquidus_temperature']};",
            "Tb": f"Tb              {pc['boiling_temperature']};",
            "Hf": f"Hf              {pc['latent_heat_fusion']};",
            "Hv": f"Hv              {pc['latent_heat_vaporization']};",
        }
        for key, val in replacements.items():
            import re
            text = re.sub(rf"({key}\s+)\S+", val, text, count=1)
        thermo_path.write_text(text, encoding="utf-8")

    laser = cfg["laser"]
    laser_path = const_dir / "laserHeatSource"
    if laser_path.exists():
        text = laser_path.read_text(encoding="utf-8")
        import re
        text = re.sub(r"(power\s+)[\d.eE+-]+", rf"\g<1>{laser['power']}", text, count=1)
        text = re.sub(r"(travelSpeed\s+)[\d.e+-]+", rf"\g<1>{laser['travel_speed']}", text, count=1)
        text = re.sub(r"(w0\s+)[\d.e+-]+", rf"\g<1>{laser['focus_radius_w0']}", text, count=1)
        laser_path.write_text(text, encoding="utf-8")


def patch_control_dict(cfg: dict, case_dir: Path) -> None:
    ctrl = case_dir / "system" / "controlDict"
    if not ctrl.exists():
        return
    time_cfg = cfg["time"]
    text = ctrl.read_text(encoding="utf-8")
    import re
    text = re.sub(r"(startFrom\s+)\w+", rf"\g<1>startTime", text, count=1)
    text = re.sub(r"(startTime\s+)[\d.e+-]+", rf"\g<1>{time_cfg['start']}", text, count=1)
    text = re.sub(r"(stopAt\s+)\w+", rf"\g<1>endTime", text, count=1)
    text = re.sub(r"(endTime\s+)[\d.e+-]+", rf"\g<1>{time_cfg['end']}", text, count=1)
    text = re.sub(r"(deltaT\s+)[\d.e+-]+", rf"\g<1>{time_cfg['initial_delta_t']}", text, count=1)
    text = re.sub(r"(maxDeltaT\s+)[\d.e+-]+", rf"\g<1>{time_cfg['max_delta_t']}", text, count=1)
    text = re.sub(r"(writeInterval\s+)[\d.e+-]+", rf"\g<1>{time_cfg['write_interval']}", text, count=1)
    text = re.sub(r"(maxCo\s+)[\d.]+", rf"\g<1>{time_cfg['max_courant']}", text, count=1)
    text = re.sub(r"(maxAlphaCo\s+)[\d.]+", rf"\g<1>{time_cfg.get('max_alpha_courant', time_cfg['max_courant'])}", text, count=1)
    text = re.sub(r"(writeFormat\s+)\w+", rf"\g<1>{time_cfg.get('write_format', 'ascii')}", text, count=1)
    text = re.sub(r"(writePrecision\s+)\d+", rf"\g<1>6", text, count=1)
    ctrl.write_text(text, encoding="utf-8")


def factor_cores(n: int) -> tuple[int, int, int]:
    """Return a simple (x y z) decomposition for n MPI ranks."""
    import math
    n = int(n)
    presets = {1: (1, 1, 1), 2: (2, 1, 1), 4: (2, 2, 1),
               8: (2, 2, 2), 16: (4, 2, 2), 32: (4, 4, 2)}
    if n in presets:
        return presets[n]
    for z in [1, 2, 3, 4, 5, 6, 7, 8]:
        rem = n // z
        if rem * z != n:
            continue
        for y in range(1, int(math.isqrt(rem)) + 1):
            if rem % y == 0:
                x = rem // y
                return (x, y, z)
    return (1, 1, n)


def patch_decompose_par_dict(cfg: dict, case_dir: Path) -> None:
    dp = case_dir / "system" / "decomposeParDict"
    if not dp.exists():
        return
    nprocs = int(cfg["parallel"]["num_processors"])
    x, y, z = factor_cores(nprocs)
    text = dp.read_text(encoding="utf-8")
    import re
    text = re.sub(r"(numberOfSubdomains\s+)\d+", rf"\g<1>{nprocs}", text, count=1)
    text = re.sub(r"(n\s+\()[\d\s]+(\);)", rf"\g<1>{x} {y} {z}\2", text, count=1)
    dp.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Configure laser keyhole CFD case")
    parser.add_argument("--config", type=Path, default=Path("config/simulation_master.yaml"))
    parser.add_argument("--case-dir", type=Path, default=Path("openfoam"))
    parser.add_argument("--output-case", type=Path, default=None,
                        help="Copy configured case to run directory")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg_path = root / args.config if not args.config.is_absolute() else args.config
    cfg = load_yaml(cfg_path)

    mat_path = root / cfg["material"]["properties_file"]
    material = load_yaml(mat_path)

    case_dir = root / args.case_dir
    write_openfoam_constants(cfg, case_dir, material)
    patch_control_dict(cfg, case_dir)
    patch_decompose_par_dict(cfg, case_dir)

    if args.output_case:
        out = root / args.output_case
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(case_dir, out)
        print(f"Configured case written to {out}")
    else:
        print(f"Configured case in {case_dir}")

    print(f"Material: {cfg['material']['alloy']}")
    print(f"Laser: {cfg['laser']['power']} W @ {cfg['laser']['travel_speed']} m/s")
    print(f"Coupling: {cfg['boundaries']['mechanical']['coupling_mode']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
