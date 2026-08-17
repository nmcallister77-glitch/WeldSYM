# laserKeyholeVoF — Build & Integration Guide

## Install into OpenFOAM User Tree

```bash
# After sourcing OpenFOAM (see docs/DEPLOYMENT_WSL_WINDOWS.md)
export FOAM_USER_APPBIN=$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin
export SOLVER_SRC=$WM_PROJECT_USER_DIR/applications/solvers

mkdir -p $SOLVER_SRC
cp -r /path/to/laser-welding-keyhole-cfd/solver/laserKeyholeVoF $SOLVER_SRC/
cd $SOLVER_SRC/laserKeyholeVoF
wmake
```

## Integration Checklist

Before first successful build, complete these stubs:

| File | Action |
|------|--------|
| `alphaEqnSubCycle.H` | Copy from `interFoam`; link `alphaEqn.H` |
| `rayTracing/rayTracer.C` | Implement isosurface intersection (cutCellIso or octree) |
| `createFields.H` | Align `rho`, `Cp` with `heRhoThermo` or custom thermo |
| `TEqn.H` | Wire `rho`, `Cp`, `alphaEff` from thermo model |
| `laserHeatSource.C` | Register `fvOptions` volumetric heat source |
| `Make/options` | Adjust library paths for your OpenFOAM version |

## OpenFOAM Version Notes

- **v2312 / v2406**: Uses `geometricVoF` — paths in `Make/options` match these releases.
- **v10 / v11**: Replace `immiscibleIncompressibleTwoPhaseMixture` includes if renamed.
- **ESI-OpenCFD**: Verify `dynamicRefineFvMesh` module is compiled.

## Verify Build

```bash
which laserKeyholeVoF
laserKeyholeVoF -help
cd $WM_PROJECT_USER_DIR/../laser-welding-keyhole-cfd/openfoam
laserKeyholeVoF
```

Expected: case starts, reads `laserHeatSource`, enters time loop (will fail without mesh).

## Solver Architecture

```
laserKeyholeVoF.C
├── rayTracer          → Fresnel multi-bounce → laserHeatFlux
├── laserHeatSource    → Gaussian beam + moving focus
├── recoilPressure     → Knudsen P_r(T) → momentum source
├── enthalpyPorosity   → f_l(T), Darcy drag, vapor source
└── interFoam core     → VOF + PIMPLE + turbulence
```

## preCICE Adapter (optional)

Add `-lpreciceAdapter` to `EXE_LIBS` and include adapter hooks in `createFields.H`
when enabling fully-coupled mode per `scripts/preCICE_config.xml`.
