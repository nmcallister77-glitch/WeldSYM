# Windows / WSL Deployment Guide — Keyhole Laser Welding CFD Stack

This guide installs and runs the full stack on **Windows 11** via **WSL2 (Ubuntu 22.04/24.04)**. Native Windows OpenFOAM builds exist but WSL2 + Linux binaries is the recommended production path.

---

## 1. Prerequisites (Windows Host)

| Requirement | Notes |
|-------------|-------|
| Windows 11 + WSL2 | `wsl --install -d Ubuntu-24.04` |
| 32+ GB RAM | AMR keyhole cases are memory-heavy |
| 8+ CPU cores | MPI decomposition |
| GPU optional | ParaView post-processing only |

Enable WSL2:

```powershell
wsl --install -d Ubuntu-24.04
wsl --update
```

Clone or copy the project into the Linux filesystem for I/O performance:

```powershell
# From PowerShell — copy project into WSL home
wsl mkdir -p ~/Projects
wsl cp -r /mnt/c/Users/nmcal/Projects/laser-welding-keyhole-cfd ~/Projects/
```

> **Tip:** Always run simulations under `~/Projects/` inside WSL, not `/mnt/c/`, to avoid 10× slower disk I/O.

---

## 2. Base Packages (inside WSL)

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential cmake git gfortran \
    openmpi-bin libopenmpi-dev \
    python3 python3-pip python3-venv \
    paraview-python3  # optional

pip3 install --user pyyaml numpy
```

---

## 3. OpenFOAM Installation

### Option A — Foundation v2312 (Recommended)

```bash
# Add OpenFOAM Foundation repository (see https://openfoam.org/download/release-2312/)
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt update
sudo apt install -y openfoam2312-default

# Persist environment
echo 'source /usr/lib/openfoam/openfoam2312/etc/bashrc' >> ~/.bashrc
source ~/.bashrc
foamVersion   # should print 2312
```

### Option B — ESI/OpenCFD v2412

Follow [OpenCFD download](https://www.openfoam.com/download/openfoam) and source `OpenFOAM-v2412/etc/bashrc`.

### Verify

```bash
which blockMesh snappyHexMesh wmake
blockMesh -help | head -1
```

---

## 4. Build laserKeyholeVoF Solver

```bash
PROJECT=~/Projects/laser-welding-keyhole-cfd
SOLVER_DST=$WM_PROJECT_USER_DIR/applications/solvers/laserKeyholeVoF

mkdir -p $WM_PROJECT_USER_DIR/applications/solvers
cp -r $PROJECT/solver/laserKeyholeVoF $SOLVER_DST
cd $SOLVER_DST

# Before wmake: copy alphaEqn from interFoam
INTERFOAM=$FOAM_SOLVERS/multiphase/interFoam
cp $INTERFOAM/alphaEqn.H $INTERFOAM/alphaControls.H $INTERFOAM/alphaEqnSubCycle.H . 2>/dev/null || true

wmake 2>&1 | tee build.log
which laserKeyholeVoF
```

If `wmake` fails on library paths, compare `Make/options` with your installed `interFoam/Make/options` and align `-I` / `-l` entries.

---

## 5. CalculiX + preCICE (Thermo-Mechanical Coupling)

### CalculiX

```bash
sudo apt install -y calculix-ccx calculix-cgx
# or build from source for preCICE adapter
ccx -v
```

### preCICE 3.x

```bash
sudo apt install -y libprecice3-dev precice-tools 2>/dev/null || {
  # Build from source if package unavailable
  git clone https://github.com/precice/precice.git
  cd precice && mkdir build && cd build
  cmake .. -DPRECICE_FEATURE_MPI_COMMUNICATION=ON
  make -j$(nproc) && sudo make install
}
precice-tools --version
```

### preCICE CalculiX Adapter

```bash
git clone https://github.com/precice/calculix-adapter.git
cd calculix-adapter
make CCX=ccx ADAPTER=PRECICE
# Produces ccx_preCICE wrapper — add to PATH
```

---

## 6. Case Setup & Mesh

```bash
cd ~/Projects/laser-welding-keyhole-cfd

# Configure from YAML
python3 scripts/configure_case.py --config config/simulation_master.yaml

# Optional STL surface
python3 scripts/generate_workpiece_stl.py

cd openfoam

# Background mesh + refinement
blockMesh
snappyHexMesh -overwrite
checkMesh -allTopology -allGeometry | tee checkMesh.log

# Inspect cell count
checkMesh | grep cells
```

Expected cell count after snappy: **2–20 M** cells depending on refinement levels; AMR adds more at runtime.

---

## 7. Running the Simulation

### CFD Only (smoke test)

```bash
cd ~/Projects/laser-welding-keyhole-cfd/openfoam
laserKeyholeVoF
# or parallel:
decomposePar
mpirun -np 8 laserKeyholeVoF -parallel
reconstructPar
```

### Full Coupled Pipeline

```bash
cd ~/Projects/laser-welding-keyhole-cfd
COUPLING=one_way NP_CFD=16 bash scripts/run_coupled_simulation.sh
```

For fully-coupled:

```bash
COUPLING=fully_coupled NP_CFD=24 NP_FEA=8 bash scripts/run_coupled_simulation.sh
```

---

## 8. Post-Processing on Windows

### ParaView (Windows native)

1. Install [ParaView 5.12+](https://www.paraview.org/download/).
2. Open VTK files from `\\wsl$\Ubuntu\home\<user>\Projects\laser-welding-keyhole-cfd\openfoam\postProcessing\`.
3. Or export Xdmf:

```bash
python3 scripts/export_vtk_xdmf.py openfoam/postProcessing
```

### Metrics JSON

```bash
python3 scripts/extract_metrics.py openfoam/postProcessing \
  --fea fea/calculix_thermomech.frd \
  --output postProcessing/metrics/final_metrics.json
```

View in Windows:

```powershell
type \\wsl$\Ubuntu\home\$env:USERNAME\Projects\laser-welding-keyhole-cfd\postProcessing\metrics\final_metrics.json
```

---

## 9. Environment Variables Reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `MATERIAL` | `Ti6Al4V` | Alloy selection |
| `COUPLING` | `one_way` | `one_way` or `fully_coupled` |
| `NP_CFD` | `32` | MPI ranks for OpenFOAM |
| `NP_FEA` | `8` | MPI ranks for CalculiX |
| `WM_PROJECT_DIR` | auto | OpenFOAM install root |
| `PRECICE_CONFIG` | `scripts/preCICE_config.xml` | Coupling config |

---

## 10. Troubleshooting

| Issue | Fix |
|-------|-----|
| `wmake: command not found` | `source $WM_PROJECT_DIR/etc/bashrc` |
| snappyHexMesh runs out of memory | Reduce `maxGlobalCells` or refinement levels |
| `laserKeyholeVoF: not found` | Rebuild solver; check `$FOAM_USER_APPBIN` in PATH |
| Slow I/O on `/mnt/c/` | Move case to `~/Projects/` inside WSL |
| preCICE socket error | Ensure `exchange-directory` exists and both participants start |
| Volume loss in VOF | Reduce `deltaT`; verify `compression 1` in `fvSchemes` |
| Keyhole collapses | Check recoil pressure units (Pa); refine keyholeCore zone |

---

## 11. Suggested Hardware Profiles

| Profile | CPU | RAM | Typical wall time (3 kW, 5 s weld) |
|---------|-----|-----|-------------------------------------|
| Dev / smoke | 8 cores | 32 GB | 24–72 h (coarse mesh) |
| Production | 32 cores | 128 GB | 8–24 h |
| HPC | 128+ cores | 512 GB | 2–6 h |

---

## 12. Quick Validation Workflow

```bash
# 1. Reduce domain and time for smoke test
# Edit config/simulation_master.yaml: endTime=0.001, num_rays=1000

# 2. Run 1 ms of physics
python3 scripts/configure_case.py
cd openfoam && mpirun -np 4 laserKeyholeVoF -parallel

# 3. Check metrics
python3 ../scripts/extract_metrics.py postProcessing

# 4. Visual check in ParaView — expect molten zone near focus
```

---

*Last updated: 2026-08-01 | OpenFOAM v2312 / preCICE 3.x / CalculiX 2.21*
