# WeldSYM — Agent notes

## What this app is

A welding engineer enters process, material, geometry and wobble parameters and gets the
weld back: fusion zone and HAZ geometry, penetration and welding mode, t8/5 and cooling
rate, HAZ phase fractions and hardness, distortion and residual stress, and the wobble
heat-concentration map.

Both solvers live in the Python package and run offline with no external solver:

- `src/weldsim/thermal/fd_solver.py` — 2D thin-plate solve, for fast screening.
- `src/weldsim/thermal/solver3d.py` — 3D through-thickness solve. Penetration, root width,
  fusion area and HAZ depth are measured on the computed fusion boundary, and the weld
  cross-section is real output rather than an inferred shape.

`src/weldsim/simulation.py` dispatches on `ThermalSimulationConfig.solver` (`"2d"`/`"3d"`)
and returns the same keys either way, plus `solution3d` for a 3D run. Everything
downstream (`weld_metrics`, `keyhole`, `microstructure`, `distortion`, `wobble_analysis`,
`report`) consumes the projected `ThermalHistory`, so both solvers share the post-processing.

OpenFOAM is **optional** and not part of the normal workflow: it is an export for anyone
who wants resolved free-surface CFD (vapour recoil, melt flow, a true keyhole cavity).
It has never been compiled or run in this environment, so do not claim it as verified.

## Quick start

```bash
pip install -e ".[gui]"
python run_gui.py          # http://localhost:8501
python -m weldsim.cli --solver 3d --nz 13 --report weld.json
```

Offline install: `pip wheel -w wheelhouse ".[gui]"` on a connected machine, then
`pip install --no-index --find-links wheelhouse -e ".[gui]"`. Streamlit telemetry is off in
`.streamlit/config.toml`. The runtime makes no network calls; core dependencies are NumPy
and PyYAML, with Matplotlib/Streamlit in the `gui` extra and PyVista/Gmsh in `cfd`.

## Checks before pushing

```bash
.venv/bin/pytest -q
.venv/bin/black --check src tests app
.venv/bin/flake8 --max-line-length=100 src tests app
```

## Performance notes for the 3D solver

The explicit step is memory-bound, so the loop works on preallocated buffers. Cost is
`nx*ny*nz*steps`, and the stable step scales with the *smallest* spacing — usually `dz` —
so raising `nz` is expensive twice over. `MAX_CELL_UPDATES` in `solver3d.py` rejects runs
that would take minutes, and the GUI shows an estimated run time before you press run.

## Key files

- `README.md` — install, solver choice, offline notes, optional OpenFOAM route.
- `pyproject.toml` — package metadata, `gui`/`cfd`/`cad`/`dev` extras.
- `run_gui.py` — launches `streamlit run app/gui.py`.
- `app/gui.py` — Streamlit dashboard: Weld simulation page (Setup / Wobble signature /
  Weld result / Thermal field), Measured vs predicted page, optional OpenFOAM export page,
  Docs page.
- `src/weldsim/calibration.py` — measured macro-sections in, residuals out, plus a grid
  search that fits absorption efficiency and keyhole taper to them. The fit is empirical
  and only valid over the process window the coupons covered.
- `src/weldsim/cli.py` — CLI, `--solver 2d|3d`, `--nz`, `--dt-3d`, `--report`.
- `keyhole-cfd/` — optional OpenFOAM case, prep scripts and `laserKeyholeVoF` solver source.

## OpenFOAM route (optional, Linux/WSL2 only)

```bash
cd <repo>/keyhole-cfd
source /opt/openfoam11/etc/bashrc
python3 scripts/configure_case.py
python3 scripts/generate_workpiece_stl.py
cd openfoam && blockMesh && laserKeyholeVoF
```

The custom solver must be built with `wmake` from `keyhole-cfd/solver/laserKeyholeVoF`.
The GUI's WSL runner panel invokes `blockMesh`/`laserKeyholeVoF` through `wsl -e bash -c`
and degrades gracefully when WSL or OpenFOAM is absent.
