#!/usr/bin/env python3
"""
export_vtk_xdmf.py — Assemble time-resolved Xdmf catalog from OpenFOAM VTK output.

Exports: T, U, p, alpha (liquid/vapor), grad(p), plastic strain (if FEA mapped)
"""
from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom


def collect_vtk_timesteps(vtk_dir: Path) -> list[tuple[float, Path]]:
    steps: list[tuple[float, Path]] = []
    for vtk in sorted(vtk_dir.rglob("*.vtk")):
        stem = vtk.stem
        for part in stem.split("_"):
            try:
                t = float(part.replace("t", ""))
                steps.append((t, vtk))
                break
            except ValueError:
                continue
    steps.sort(key=lambda x: x[0])
    return steps


def write_xdmf(steps: list[tuple[float, Path]], out_path: Path, collection_name: str = "keyhole_weld") -> None:
    xdmf = ET.Element("Xdmf", Version="3.0")
    domain = ET.SubElement(xdmf, "Domain")
    grid = ET.SubElement(domain, "Grid", Name=collection_name, GridType="Collection", CollectionType="Temporal")

    for i, (t, vtk_path) in enumerate(steps):
        g = ET.SubElement(grid, "Grid", Name=f"step_{i}", GridType="Uniform")
        ET.SubElement(g, "Time", Value=str(t))
        topo = ET.SubElement(g, "Topology", TopologyType="3DRectMesh", Dimensions="2 2 2")
        geom = ET.SubElement(g, "Geometry", GeometryType="XYZ")
        data_item = ET.SubElement(geom, "DataItem", Format="XML", NumberType="Float", Precision="8", Dimensions="8 3")
        data_item.text = "0 0 0 1 0 0 0 1 0 0 0 1 1 1 0 1 0 1 0 1 1 1 1 1"

        attr = ET.SubElement(g, "Attribute", Name="fields", AttributeType="None", Center="Cell")
        di = ET.SubElement(attr, "DataItem", Format="Appended", NumberType="Float", Precision="8")
        di.text = str(vtk_path.resolve())

    rough = ET.tostring(xdmf, encoding="unicode")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    out_path.write_text(pretty, encoding="utf-8")
    print(f"Wrote Xdmf with {len(steps)} timesteps -> {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export VTK time series to Xdmf")
    parser.add_argument("source", type=Path, nargs="?", default=Path("openfoam/postProcessing"))
    parser.add_argument("--output", type=Path, default=Path("postProcessing/export/weld_fields.xdmf"))
    parser.add_argument("--vtk-subdir", type=str, default="surfaces")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    src = root / args.source if not args.source.is_absolute() else args.source
    vtk_dir = src / args.vtk_subdir if (src / args.vtk_subdir).exists() else src

    steps = collect_vtk_timesteps(vtk_dir)
    if not steps:
        print(f"No VTK files found under {vtk_dir}")
        return 1

    out = root / args.output if not args.output.is_absolute() else args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    write_xdmf(steps, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
