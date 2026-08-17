# README — Keyhole Laser Welding Multiphysics Simulation Package

Production-ready configuration for **3D transient keyhole-mode laser welding** coupled with **thermo-mechanical distortion analysis**.

## Package Structure

```
laser-welding-keyhole-cfd/
├── SYSTEM_PROMPT.md              # Master AI/engineer system prompt (full physics spec)
├── docs/
│   └── DEPLOYMENT_WSL_WINDOWS.md # Full Windows/WSL install + run guide
├── config/
│   ├── simulation_master.yaml    # Single source of truth for run parameters
│   └── physics_models.yaml       # Model registry
├── materials/
│   ├── Ti6Al4V.yaml              # Ti-6Al-4V temperature-dependent properties
│   └── S355_structural_steel.yaml
├── solver/
│   └── laserKeyholeVoF/          # Custom OpenFOAM solver (C++ skeleton)
├── openfoam/                     # OpenFOAM case (VOF + AMR + laser source)
│   └── system/
│       ├── blockMeshDict         # Background mesh (80×40×16 mm domain)
│       └── snappyHexMeshDict     # Keyhole/melt-pool refinement zones
├── fea/                          # CalculiX thermo-mechanical input
├── scripts/
│   ├── configure_case.py         # YAML → OpenFOAM patch
│   ├── generate_workpiece_stl.py # Optional triSurface generator
│   ├── extract_metrics.py        # Keyhole depth, 8/5 rate, distortion
│   ├── export_vtk_xdmf.py        # ParaView-ready time series
│   ├── preCICE_config.xml        # Fluid-solid coupling
│   └── run_coupled_simulation.sh # Production launcher
└── validation/
    └── metrics_definitions.yaml
```

## Prerequisites

| Component | Version |
|-----------|---------|
| OpenFOAM | v2312+ with custom `laserKeyholeVoF` solver |
| CalculiX | 2.21+ with preCICE adapter |
| preCICE | 3.x |
| Python | 3.9+ (PyYAML, numpy) |
| MPI | OpenMPI or MPICH |
| OS | Linux or **WSL2** (see `docs/DEPLOYMENT_WSL_WINDOWS.md`) |

> **Note:** `solver/laserKeyholeVoF/` is a buildable C++ skeleton extending `interFoam`. Copy to `$WM_PROJECT_USER_DIR/applications/solvers/` and run `wmake`. See `solver/laserKeyholeVoF/README.md` for integration steps.

## Quick Start

**Windows users:** follow [`docs/DEPLOYMENT_WSL_WINDOWS.md`](docs/DEPLOYMENT_WSL_WINDOWS.md) first.

```bash
# 1. Build solver (once)
cp -r solver/laserKeyholeVoF $WM_PROJECT_USER_DIR/applications/solvers/
cd $WM_PROJECT_USER_DIR/applications/solvers/laserKeyholeVoF && wmake

# 2. Configure case
python scripts/configure_case.py --config config/simulation_master.yaml

# 3. Generate mesh
cd openfoam && blockMesh && snappyHexMesh -overwrite && checkMesh

# 4. Run full pipeline
cd .. && COUPLING=one_way NP_CFD=32 bash scripts/run_coupled_simulation.sh

# 5. Post-process
python scripts/export_vtk_xdmf.py openfoam/postProcessing
python scripts/extract_metrics.py openfoam/postProcessing --fea fea/calculix_thermomech.frd
```

## Switching Alloys

In `config/simulation_master.yaml`:

```yaml
material:
  alloy: "S355"
  properties_file: "materials/S355_structural_steel.yaml"
```

For S355, enable phase transformation in FEA:

```yaml
fea:
  phase_transformation:
    enabled: true
```

## Key Physics Summary

- **Fluid:** Navier-Stokes + VOF + surface tension + Marangoni + buoyancy + Knudsen recoil pressure
- **Phase change:** Enthalpy-porosity melting/solidification + vaporization source
- **Optics:** Dual-cylinder Gaussian beam with multi-bounce Fresnel ray tracing
- **Mesh:** Dynamic AMR in keyhole capillary (10–25 µm target)
- **Structural:** Transient thermal-elastic-plastic FEA with fixture BCs
- **Outputs:** VTK/Xdmf fields + JSON metric time series

## Validation Metrics

| Metric | Symbol | Unit |
|--------|--------|------|
| Keyhole depth | d_kh | m |
| Pool width (top/root) | w_top, w_root | m |
| Cooling rate 8/5 | R_8/5 | K/s |
| Angular distortion | θ | deg |
| Longitudinal shrinkage | ε_L | — |
| Out-of-plane displacement | w_max | m |

See `validation/metrics_definitions.yaml` for extraction methods and tolerances.

## Using as an AI System Prompt

Load `SYSTEM_PROMPT.md` as the agent system prompt for automated case setup, run monitoring, and validation. The agent should always cross-reference `config/simulation_master.yaml` for numeric parameters.

## License

Engineering configuration templates — adapt for your organization's solver build and validation data.
