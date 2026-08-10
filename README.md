# Weld Sim

Streamlit dashboard + Python tooling for laser-welding simulation: fast 2D thermal estimates with wobble, and a full 3D OpenFOAM VOF keyhole solver.

## Quick start

```bash
# 1. Install
cd welding-sim
pip install -e ".[gui]"

# 2. Launch the dashboard
python run_gui.py
# open http://localhost:8501
```

## What it does

- **2D Thermal + Wobble**
  - Material library (Ti-6Al-4V, S355 steel) with temperature-dependent properties
  - Arbitrary weld path with length, start/end points
  - Wobble calculator: circle, line/sine, figure-8, infinity patterns
  - "Go" button draws the wobbled beam path and accumulated heat signature
  - "Run" button runs a finite-difference thermal simulation
  - Outputs: 2D/3D temperature plots, melt-pool metrics, longitudinal/transverse profiles, time-temperature at a probe, CSV export

- **3D Keyhole CFD**
  - OpenFOAM VOF solver (`laserKeyholeVoF`) with enthalpy-porosity, recoil pressure, and laser ray tracing
  - Generate workpiece STL and configure the OpenFOAM case from the GUI
  - 3D STL preview with PyVista
  - Build/run on WSL2 / Ubuntu with `blockMesh` and `laserKeyholeVoF`

## Repository layout

- `app/gui.py` — Streamlit dashboard
- `src/weldsim/` — Python package (thermal FD solver, wobble path, material loader, simulation API)
- `keyhole-cfd/` — OpenFOAM case + custom `laserKeyholeVoF` solver
- `keyhole-cfd/materials/` — YAML material property tables

## CLI

```bash
python -m weldsim.cli --power 1500 --speed 0.01 --t-end 2 --dt 0.05
```

## 3D solver (WSL2 / Linux)

```bash
cd keyhole-cfd
python3 scripts/configure_case.py
python3 scripts/generate_workpiece_stl.py
blockMesh
laserKeyholeVoF
```
