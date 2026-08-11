# WeldSYM — Agent notes

## Repo location

- Root: `C:\Users\nmcal\welding-sim\welding-sim`
- Git remote: `https://github.com/nmcallister77-glitch/WeldSYM.git`
- Outer `C:\Users\nmcal\welding-sim` is a wrapper directory; the real project is the nested `welding-sim` folder.

## Quick start (Windows 2D)

```powershell
cd "C:\Users\nmcal\welding-sim\welding-sim"
pip install -e ".[gui]"
python run_gui.py
```

Then open http://localhost:8501.

## WSL2 / 3D keyhole CFD

The 3D OpenFOAM solver is Linux-only. Required pieces:

- WSL2 + Ubuntu distro (currently none installed on this machine).
- OpenFOAM (e.g. `openfoam11`) sourced via `/opt/openfoam11/etc/bashrc`.
- The custom solver built from `keyhole-cfd/solver/laserKeyholeVoF` (`wmake` from inside WSL).

WSL path for the case:

```text
/mnt/c/Users/nmcal/welding-sim/welding-sim/keyhole-cfd
```

Run sequence once the solver is built:

```bash
cd /mnt/c/Users/nmcal/welding-sim/welding-sim/keyhole-cfd
source /opt/openfoam11/etc/bashrc
python3 scripts/configure_case.py
python3 scripts/generate_workpiece_stl.py

cd openfoam
blockMesh
laserKeyholeVoF
```

The Streamlit app has a WSL runner panel in `app/gui.py` (around line 553) that tries to run `blockMesh` and `laserKeyholeVoF` directly from the GUI using `wsl -e bash -c ...`.

## Key files

- `README.md` — Windows vs. WSL2 setup and run instructions.
- `pyproject.toml` — package metadata, dependencies, optional `gui` extras.
- `run_gui.py` — `streamlit run app/gui.py` wrapper.
- `app/gui.py` — Streamlit dashboard with 2D thermal + wobble + 3D keyhole CFD tabs.
- `src/weldsim/cli.py` — command-line 2D thermal entry point.
- `keyhole-cfd/config/simulation_master.yaml` — master config for the OpenFOAM case.
- `keyhole-cfd/scripts/{configure_case.py,generate_workpiece_stl.py}` — case prep scripts.
- `keyhole-cfd/solver/laserKeyholeVoF/` — custom OpenFOAM VOF solver source.
- `keyhole-cfd/openfoam/` — OpenFOAM case files (`blockMeshDict`, `controlDict`, etc.).
