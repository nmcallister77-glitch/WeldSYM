#!/usr/bin/env python3
"""
generate_workpiece_stl.py — Optional triSurface for curved/fillet workpiece geometry.

Default output: openfoam/triSurface/workpiece.stl (box 80x40x6 mm).
Replace with CAD export for production runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def write_ascii_stl_box(path: Path, lx: float, ly: float, lz: float) -> None:
    """Write a simple rectangular prism STL centered at origin, top face at z=0."""
    hx, hy = lx / 2, ly / 2
    # Box from z=-lz to z=0, x in [0,lx], y in [-hy, hy]
    vertices = [
        (0, -hy, -lz), (lx, -hy, -lz), (lx, hy, -lz), (0, hy, -lz),
        (0, -hy, 0), (lx, -hy, 0), (lx, hy, 0), (0, hy, 0),
    ]
    faces = [
        (0, 1, 2), (0, 2, 3),       # bottom
        (4, 6, 5), (4, 7, 6),       # top
        (0, 4, 5), (0, 5, 1),       # front
        (1, 5, 6), (1, 6, 2),       # right
        (2, 6, 7), (2, 7, 3),       # back
        (3, 7, 4), (3, 4, 0),       # left
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("solid workpiece\n")
        for tri in faces:
            f.write("  facet normal 0 0 0\n    outer loop\n")
            for idx in tri:
                x, y, z = vertices[idx]
                f.write(f"      vertex {x} {y} {z}\n")
            f.write("    endloop\n  endfacet\n")
        f.write("endsolid workpiece\n")
    print(f"Wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("openfoam/triSurface/workpiece.stl"))
    parser.add_argument("--length", type=float, default=0.080)
    parser.add_argument("--width", type=float, default=0.040)
    parser.add_argument("--thickness", type=float, default=0.006)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = root / args.output if not args.output.is_absolute() else args.output
    write_ascii_stl_box(out, args.length, args.width, args.thickness)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
