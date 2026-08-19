# Weld Sim

Streamlit dashboard + Python tooling for laser-welding simulation. Enter process, material, geometry and wobble parameters and get the weld back: fusion zone and HAZ, penetration and welding mode, cooling rates and HAZ metallurgy, distortion and residual stress, and the wobble heat-concentration map.

Both solvers ship inside the Python package: a fast 2D thin-plate solve for screening, and a 3D through-thickness solve that measures penetration and the weld cross-section. Nothing external is required — no internet at runtime, no OpenFOAM, no WSL2, no compilation. OpenFOAM stays available as an optional export for anyone who wants resolved free-surface CFD.

## Repository layout

Clone or copy this repo somewhere on your machine. Below, `<REPO>` stands for the repo root on your machine (e.g. `C:\Dev\WeldSYM` on Windows or `~/repos/WeldSYM` on Linux).

```text
app/gui.py                     — Streamlit dashboard
src/weldsim/thermal/fd_solver.py — 2D thin-plate thermal solver
src/weldsim/thermal/solver3d.py  — 3D through-thickness thermal solver
src/weldsim/                   — weld metrics, keyhole, microstructure, distortion, wobble, report
keyhole-cfd/                   — optional OpenFOAM VOF case + laserKeyholeVoF solver
keyhole-cfd/materials/         — YAML material property tables
```

## 1. Weld simulation (any OS, offline)

Everything in this section is pure Python/NumPy and runs on Windows, Linux and macOS. Change into the repo root:

```powershell
cd <REPO>
```

Install once:

```powershell
pip install -e ".[gui]"
```

On a machine with no internet, build a wheelhouse on a connected machine first, copy it across, and install from it:

```powershell
pip wheel -w wheelhouse ".[gui]"          # connected machine
pip install --no-index --find-links wheelhouse -e ".[gui]"   # air-gapped machine
```

The app makes no network calls at runtime: material data comes from the YAML files in the repo, plots are rendered server-side by Matplotlib, and Streamlit telemetry is switched off in `.streamlit/config.toml`.

Launch the dashboard:

```powershell
python run_gui.py
```

Then open http://localhost:8501 in your browser.

Work on the **Weld simulation** page: set the parameters on the **Setup** tab, pick a solver, and read the answer on the **Weld result** tab.

### Choosing a solver

| | 2D thin plate | 3D through-thickness |
| --- | --- | --- |
| Run time on the default grid | ~1 s | ~10 s (progress bar, estimate shown before you run) |
| Penetration | energy balance over the melted channel | **measured** on the computed fusion boundary |
| Weld cross-section | peak-temperature profile across the weld | transverse and longitudinal sections with a depth axis |
| Also reports | — | root width, fusion area, HAZ depth, keyhole power share, full-penetration status |

The 3D solve adds a tapered volumetric keyhole source to the moving surface Gaussian, latent heat of fusion, surface convection/radiation and an evaporation cap.

### What the weld assessment gives you

| Output | Basis |
| --- | --- |
| Fusion-zone width/length, HAZ width, macro-section | Peak-temperature field vs. solidus and the material's HAZ limit |
| Penetration and welding mode (conduction / transition / keyhole) | 3D: the fusion boundary through the thickness. 2D: absorbed peak intensity vs. the ~1 MW/cm² keyhole threshold plus an energy balance over the melted channel |
| t8/5, cooling rate, HAZ phase fractions, hardness, carbon equivalent | Thermal cycle of each cell, with the CCT-style limits in the material YAML |
| Distortion (angular, transverse, longitudinal, bowing) and residual stress | Inherent-strain / shrinkage-force model for a single unrestrained pass |
| Wobble swept width, overlap, spot speed and heat-concentration map | Accumulated absorbed energy density along the oscillating beam track |

Every number is an engineering estimate, not a validated high-fidelity result. Neither
solver resolves the free surface, vapour recoil, melt flow or Marangoni convection — the
keyhole is an assumed tapered capillary — and distortion is an inherent-strain estimate
rather than thermo-mechanical FEA. The report prints explicit warnings when its
assumptions break down (boiling reached, fusion zone under-resolved, wobble faster than
the time step, energy balance inconsistent). For resolved keyhole physics use the optional
OpenFOAM export below, and for qualification-grade distortion a thermo-mechanical FE run.
The same report is available from the CLI with `--report weld.json`, and downloadable as
JSON from the GUI.

### Comparing against real coupons, and calibrating to them

The **Measured vs predicted** page closes the loop with the workshop. Weld a bracket of
coupons at a fixed focus, varying power and travel speed, cut, polish and etch them, then
enter the measured penetration and top-surface fusion width for each one. The page solves
the same parameters and reports per-coupon residuals, a parity plot, and the RMS error and
systematic bias.

Expect the first comparison to disagree on absolute penetration, because two things about
your machine and joint cannot be known in advance: how much beam power is actually
absorbed, and how the capillary narrows with depth. "Fit absorption and keyhole taper"
searches those two parameters for the pair that best reproduces your macro-sections, and
the fit downloads as YAML. It is an empirical calibration, valid for the material,
thickness and process window the coupons covered — not a general physical constant, and no
substitute for procedure qualification.

## 2. Optional: OpenFOAM export (requires Linux/WSL2 + OpenFOAM)

Nothing here is needed for the results above. It exports the same job as an OpenFOAM VOF case for resolved free-surface CFD. OpenFOAM is a separate native package that has to be installed and compiled outside this app; it has not been built or run in this repo's CI.

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

On the **OpenFOAM export (optional)** page you can also:

- Generate the workpiece STL
- Configure the OpenFOAM case
- Preview the STL in 3D
- Open the **WSL runner** panel and run `blockMesh` / `laserKeyholeVoF` directly from the GUI

## CLI

```powershell
# fast 2D screening
python -m weldsim.cli --power 1500 --speed 0.01 --t-end 2 --dt 0.05 --report weld.json

# 3D solve: measured penetration and cross-section
python -m weldsim.cli --solver 3d --nz 13 --power 1500 --speed 0.01 --t-end 2 --report weld.json
```

Invalid or non-physical inputs (zero/negative power, speed or thickness) and time steps that
violate the explicit-scheme stability limit are reported as a single-line error and exit with
status 2 without writing output. The stability message includes the largest usable `--dt`.

## Tests

```bash
pytest -q
```
