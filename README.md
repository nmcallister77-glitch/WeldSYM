# Weld Sim

Streamlit dashboard + Python tooling for laser-welding simulation: fast 2D thermal estimates with wobble, and a full 3D OpenFOAM VOF keyhole solver.

## Repository layout

Clone or copy this repo somewhere on your machine. Below, `<REPO>` stands for the repo root on your machine (e.g. `C:\Dev\WeldSYM` on Windows or `~/repos/WeldSYM` on Linux).

```text
app/gui.py              — Streamlit dashboard
src/weldsim/            — Python package (thermal FD solver, wobble path, material loader)
keyhole-cfd/            — OpenFOAM VOF case + custom laserKeyholeVoF solver
keyhole-cfd/materials/  — YAML material property tables
```

## 1. 2D thermal + wobble (any OS)

The 2D tooling is pure Python and runs on Windows, Linux and macOS. Change into the repo root:

```powershell
cd <REPO>
```

Install once:

```powershell
pip install -e ".[gui]"
```

Launch the dashboard:

```powershell
python run_gui.py
```

Then open http://localhost:8501 in your browser.

Use the **2D Thermal + Wobble** tab for fast laser-welding estimates, and the **3D Keyhole CFD** tab to prepare the OpenFOAM case.

## 2. 3D keyhole CFD (requires WSL2 + OpenFOAM)

The full 3D solver is a Linux/WSL2 application. `blockMesh`, `laserKeyholeVoF`, and the `python3` commands are **not Windows commands**, so do not run them in PowerShell directly.

### 2.1 Install WSL2 and a Linux distro

1. Open a PowerShell **administrator** window and run:

   ```powershell
   wsl --install
   ```

2. Reboot when prompted.
3. After reboot, WSL will install Ubuntu. If it does not, run:

   ```powershell
   wsl --install -d Ubuntu
   ```

4. Start WSL and set a username/password:

   ```powershell
   wsl
   ```

### 2.2 Install OpenFOAM in WSL2

Inside the WSL Ubuntu terminal, install OpenFOAM. For OpenFOAM v11 (example):

```bash
sudo apt update
sudo apt install -y openfoam11
```

Source the OpenFOAM environment in every WSL shell where you run OpenFOAM commands:

```bash
source /opt/openfoam11/etc/bashrc
```

### 2.3 Build the custom solver

The custom `laserKeyholeVoF` solver must be compiled in WSL2 inside `keyhole-cfd/solver/laserKeyholeVoF`.

### 2.4 Run the case

A Windows checkout is visible inside WSL under `/mnt/<drive>/...`; for example `C:\Dev\WeldSYM` becomes `/mnt/c/Dev/WeldSYM`. The **WSL runner** panel in the GUI performs this translation for you and shows the resulting path.

Open a WSL terminal and run:

```bash
cd <REPO-IN-WSL>/keyhole-cfd
source /opt/openfoam11/etc/bashrc

python3 scripts/configure_case.py
python3 scripts/generate_workpiece_stl.py

blockMesh
laserKeyholeVoF
```

For parallel (4 cores):

```bash
decomposePar
mpirun -np 4 laserKeyholeVoF -parallel
reconstructPar
```

### 2.5 From the Streamlit GUI

On the **3D Keyhole CFD** tab you can also:

- Generate the workpiece STL
- Configure the OpenFOAM case
- Preview the STL in 3D
- Open the **WSL runner** panel and run `blockMesh` / `laserKeyholeVoF` directly from the GUI

## CLI

```powershell
python -m weldsim.cli --power 1500 --speed 0.01 --t-end 2 --dt 0.05
```

Invalid or non-physical inputs (zero/negative power, speed or thickness) and time steps that
violate the explicit-scheme stability limit are reported as a single-line error and exit with
status 2 without writing output. The stability message includes the largest usable `--dt`.

## Tests

```bash
pytest -q
```
