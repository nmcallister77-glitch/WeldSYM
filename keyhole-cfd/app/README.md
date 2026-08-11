# Laser Keyhole OpenFOAM Runner

A standalone Windows GUI for configuring and running the `laserKeyholeVoF` OpenFOAM 2306 solver inside WSL2.

## Quick start

1. Make sure WSL2 is set up with OpenFOAM 2306 and that `laserKeyholeVoF` has been compiled at least once.
2. Make sure the case mesh and initial fields exist under `/home/<user>/welding-cases/keyhole-cfd` in WSL.
3. Double-click `RunLaserKeyholeApp.bat` in the repo root, or run:

```powershell
pythonw keyhole-cfd\app\laserkeyhole_app.pyw
```

## What the app does

- Loads common settings from `keyhole-cfd/config/simulation_master.yaml`.
- Lets you edit laser power, travel speed, focus radius, run time, time-step controls, and number of MPI cores.
- Saves those values back to `simulation_master.yaml` and patches the OpenFOAM case.
- Copies `config/`, `materials/`, `scripts/configure_case.py`, `system/`, and `constant/` into the WSL case.
- Runs `decomposePar -force`.
- Runs `mpirun --oversubscribe -np <N> laserKeyholeVoF -parallel`.
- Provides Stop, Reconstruct latest, and Rebuild solver buttons.

## Buttons

- **Save config**: write the fields back to `simulation_master.yaml` only.
- **Rebuild solver**: copy `laserKeyholeVoF` source to the WSL OpenFOAM tree and `wmake` it.
- **Configure + Decompose**: patch the case and generate the `processor*` directories.
- **Run simulation**: patch, decompose, and start the parallel run.
- **Stop**: terminate the currently running WSL command.
- **Reconstruct latest**: run `reconstructPar -latestTime` on the WSL case.

## Notes

- The app assumes the WSL distro is `Ubuntu-24.04` and the WSL case dir is `/home/nmcal/welding-cases/keyhole-cfd` (i.e., the OpenFOAM case is the top-level `keyhole-cfd` directory in WSL). Change these in the GUI if your setup differs.
- The mesh (`constant/polyMesh/`) is not copied from Windows; it is expected to already exist in the WSL case.
- The `0/` initial time directory is not overwritten by the app. Use the OpenFOAM `setFields` command in WSL if you need to regenerate initial metal/gas fields.
