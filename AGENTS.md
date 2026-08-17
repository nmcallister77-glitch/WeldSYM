# WeldSYM — Agent notes

## Repo location

- Git remote: `https://github.com/nmcallister77-glitch/WeldSYM.git`
- `<REPO>` below means the repo root on the current machine. Nick's Windows checkout lives under
  `C:\Users\nmcal\welding-sim\welding-sim` (the outer `welding-sim` is a wrapper directory; the
  project is the nested one), but nothing in the code depends on that path.

## Quick start (2D, any OS)

```powershell
cd <REPO>
pip install -e ".[gui]"
python run_gui.py
```

Then open http://localhost:8501.

## WSL2 / 3D keyhole CFD

The 3D OpenFOAM solver is Linux-only. Required pieces:

- WSL2 + Ubuntu distro (the GUI's WSL runner panel lists the distros it detects).
- OpenFOAM (e.g. `openfoam11`) sourced via `/opt/openfoam11/etc/bashrc`.
- The custom solver built from `keyhole-cfd/solver/laserKeyholeVoF` (`wmake` from inside WSL).

A Windows checkout appears inside WSL as `/mnt/<drive>/...`; the GUI's WSL runner panel derives and
displays this path itself.

Run sequence once the solver is built:

```bash
cd <REPO-IN-WSL>/keyhole-cfd
source /opt/openfoam11/etc/bashrc
python3 scripts/configure_case.py
python3 scripts/generate_workpiece_stl.py

cd openfoam
blockMesh
laserKeyholeVoF
```

The Streamlit app has a WSL runner panel in `app/gui.py` that runs `blockMesh` and `laserKeyholeVoF`
from the GUI via `wsl -e bash -c ...`. On machines without `wsl` it reports a clear in-page error
rather than raising.

## Validation and errors

All simulation inputs are checked by `validate_config()` in `src/weldsim/simulation.py`, which raises
`ValidationError`; the CFL check in `thermal/fd_solver.py` raises `StabilityError`. Both derive from
`WeldSimError` (and `ValueError`) in `src/weldsim/errors.py`. The CLI turns them into a single-line
stderr message with exit code 2; the GUI shows them via `st.error`. Put new input checks in
`validate_config` so both front ends pick them up.

## Key files

- `README.md` — Windows vs. WSL2 setup and run instructions.
- `pyproject.toml` — package metadata, dependencies, optional `gui` extras.
- `run_gui.py` — `streamlit run app/gui.py` wrapper.
- `app/gui.py` — Streamlit dashboard with 2D thermal + wobble + 3D keyhole CFD tabs.
- `src/weldsim/cli.py` — command-line 2D thermal entry point.
- `src/weldsim/errors.py` — exception hierarchy shared by the CLI and GUI.
- `tests/` — `pytest -q` covers input validation, stability guard and CLI exit codes.
- `keyhole-cfd/config/simulation_master.yaml` — master config for the OpenFOAM case.
- `keyhole-cfd/scripts/{configure_case.py,generate_workpiece_stl.py}` — case prep scripts.
- `keyhole-cfd/solver/laserKeyholeVoF/` — custom OpenFOAM VOF solver source.
- `keyhole-cfd/openfoam/` — OpenFOAM case files (`blockMeshDict`, `controlDict`, etc.).
