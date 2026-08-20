"""High-level simulation API."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict

import numpy as np

from .errors import ValidationError
from .materials import Material, list_materials, load_material, material_from_params
from .thermal.fd_solver import PhaseModel, ThermalHistory, run_2d_fd_thermal
from .thermal.solver3d import Solution3D, run_3d_thermal
from .types import MaterialParams, WeldParams
from .weld_path import WeldPath, WobbleParams

#: Bounds on problem size. Raised to allow multi-day runs on large RAM machines.
#: 10 million cells is a 10x increase over the original 1 million while still
#: rejecting accidental 500 x 500 x 50 test grids and guarding against OOM.
MAX_CELLS = 10_000_000
MAX_STEPS = 10_000_000

#: Solvers the high-level API can dispatch to. The 2D thin-plate solve is fast
#: enough for parameter sweeps; the 3D solve resolves depth, so it is the one
#: that can report penetration and a transverse cross-section.
SOLVERS = ("2d", "3d")


@dataclass
class ThermalSimulationConfig:
    """Configuration for a transient thermal simulation."""

    nx: int = 51
    ny: int = 26
    Lx: float = 0.1  # m
    Ly: float = 0.05  # m
    t_end: float = 10.0  # s
    dt: float = 0.05  # s
    weld: WeldParams | None = None
    material: MaterialParams | Material = field(default_factory=MaterialParams)
    output_file: str | None = "results/temperature.csv"
    T1: float = 0.005  # effective thickness (m)
    path: WeldPath | None = None
    wobble: WobbleParams | None = None
    probe: tuple[float, float] | None = None
    plate_thickness: float | None = None  # m; defaults to T1
    top_thickness: float | None = (
        None  # m; top sheet in a 2t lap joint (defaults to plate_thickness)
    )
    store_3d_frames: bool = False  # keep 3D temperature snapshots for post-run animation
    frame_interval: float = 0.02  # seconds between stored 3D animation frames
    max_stored_frames: int = 1_000  # hard cap on stored 3D frames
    phase_change: bool = True  # latent heat, evaporation cap and surface losses
    solver: str = "2d"  # "2d" thin plate or "3d" through-thickness
    nz: int = 17  # grid points through the thickness, 3D solver only
    dt_3d: float | None = None  # s; None lets the 3D solver pick a stable step
    keyhole_taper: float | None = None  # tip/surface radius of the capillary, 3D only

    @property
    def thickness(self) -> float:
        """Plate thickness used for the mechanical models (m).

        ``T1`` is the depth the surface flux is spread over for the thermal
        solve, which is not necessarily the plate thickness once the beam only
        partially penetrates.
        """
        return self.T1 if self.plate_thickness is None else self.plate_thickness


def _to_material_params(material: MaterialParams | Material) -> MaterialParams:
    if isinstance(material, Material):
        return MaterialParams(
            k=material.thermal_conductivity,
            rho=material.density,
            cp=material.specific_heat,
            T0=material.T0,
        )
    return material


def _phase_model(config: ThermalSimulationConfig) -> PhaseModel | None:
    """Phase-change physics for the run, when the alloy data is available.

    Only a :class:`~weldsim.materials.Material` carries melting and boiling
    data; with bare :class:`~weldsim.types.MaterialParams` the solve stays pure
    conduction, which will overshoot for a concentrated beam.
    """
    material = config.material
    if not config.phase_change or not isinstance(material, Material):
        return None
    return PhaseModel(
        solidus=material.solidus,
        liquidus=material.liquidus,
        latent_heat=material.latent_heat_fusion,
        boiling=material.boiling,
    )


def validate_config(config: ThermalSimulationConfig) -> None:
    """Check that a configuration is physically and numerically meaningful.

    Raises
    ------
    ValidationError
        Naming the offending parameter and its allowed range.
    """
    positive = {
        "Lx (plate length)": config.Lx,
        "Ly (plate width)": config.Ly,
        "T1 (effective thickness)": config.T1,
        "dt (time step)": config.dt,
        "t_end (simulation time)": config.t_end,
    }
    for name, value in positive.items():
        if not value > 0:
            raise ValidationError(f"{name} must be greater than 0, got {value}.")

    if config.solver not in SOLVERS:
        raise ValidationError(
            f"Unknown solver {config.solver!r}; choose one of {', '.join(SOLVERS)}."
        )

    counts = [("nx", config.nx), ("ny", config.ny)]
    if config.solver == "3d":
        counts.append(("nz", config.nz))
    for name, count in counts:
        if count < 3:
            raise ValidationError(
                f"{name} must be at least 3 to have an interior node, got {count}."
            )

    cells = config.nx * config.ny
    shape = f"{config.nx}x{config.ny}"
    if config.solver == "3d":
        cells *= config.nz
        shape += f"x{config.nz}"
    if cells > MAX_CELLS:
        raise ValidationError(
            f"Grid of {shape} = {cells} cells exceeds the limit of "
            f"{MAX_CELLS}. Coarsen the mesh or shrink the domain."
        )

    steps = int(np.ceil(config.t_end / config.dt))
    if steps > MAX_STEPS:
        raise ValidationError(
            f"t_end / dt = {steps} time steps exceeds the limit of {MAX_STEPS}. "
            "Increase dt or shorten t_end."
        )

    mat = _to_material_params(config.material)
    material_props = (
        ("k (conductivity)", mat.k),
        ("rho (density)", mat.rho),
        ("cp (specific heat)", mat.cp),
    )
    for name, value in material_props:
        if not value > 0:
            raise ValidationError(f"Material {name} must be greater than 0, got {value}.")

    weld = config.weld
    if weld is None:
        return
    if not weld.power > 0:
        raise ValidationError(f"Power must be greater than 0 W, got {weld.power}.")
    if not 0 < weld.efficiency <= 1:
        raise ValidationError(f"Efficiency must be in (0, 1], got {weld.efficiency}.")
    if not weld.speed > 0:
        raise ValidationError(
            f"Travel speed must be greater than 0 m/s, got {weld.speed}. "
            "A stationary or reversing torch is not supported."
        )
    if not weld.sigma > 0:
        raise ValidationError(f"Beam sigma must be greater than 0 m, got {weld.sigma}.")


def run_thermal_simulation(
    config: ThermalSimulationConfig,
    on_progress: Callable[[float], None] | None = None,
    abort: Callable[[], bool] | None = None,
) -> Dict[str, Any]:
    """
    Run a transient thermal simulation with a moving heat source.

    ``config.solver`` selects the thin-plate 2D solve or the through-thickness
    3D solve; both return the same keys, so everything downstream — weld
    metrics, microstructure, distortion, the report — works either way.

    Parameters
    ----------
    on_progress : callable | None
        Progress callback taking a 0..1 fraction.
    abort : callable | None
        Should return True when the caller wants the simulation to stop. The
        solver checks this periodically and raises :class:`AbortError` if it
        returns True.

    Returns
    -------
    result : dict
        ``x``, ``y`` and the final surface temperature field ``T``, the optional
        probe history (``t``, ``T_probe``), and ``history`` — the
        :class:`~weldsim.thermal.fd_solver.ThermalHistory` holding the peak
        temperature and cooling-rate fields the weld metrics are built from. A
        3D run adds ``z`` and ``solution3d``, which carry the depth information.
    """
    if config.weld is None:
        config.weld = WeldParams(
            power=3000.0,
            efficiency=0.8,
            speed=0.005,
            start_pos=(0.01, config.Ly / 2),
            direction="x",
        )

    validate_config(config)

    mat = _to_material_params(config.material)

    solidus = config.material.solidus if isinstance(config.material, Material) else None
    phase = _phase_model(config)

    if config.solver == "3d":
        return _run_3d(config, phase, on_progress, abort)

    x, y, T, T_probe, history = run_2d_fd_thermal(
        nx=config.nx,
        ny=config.ny,
        Lx=config.Lx,
        Ly=config.Ly,
        t_end=config.t_end,
        dt=config.dt,
        weld=config.weld,
        material=mat,
        T0=mat.T0,
        h=config.T1,
        path=config.path,
        wobble=config.wobble,
        probe=config.probe,
        dwell_temp=solidus,
        extra_dwell_temps=_extra_dwell_temps(config.material),
        phase=phase,
        on_progress=on_progress,
        abort=abort,
    )

    if config.output_file is not None:
        os.makedirs(os.path.dirname(config.output_file) or ".", exist_ok=True)
        save_temperature_csv(config.output_file, x, y, T)

    result: Dict[str, Any] = {
        "x": x,
        "y": y,
        "T": T,
        "history": history,
        "top_thickness": config.top_thickness,
    }
    if T_probe is not None:
        result["t"] = np.arange(0, config.t_end, config.dt)
        result["T_probe"] = T_probe
    return result


def _extra_dwell_temps(material: Material | MaterialParams) -> tuple[float, ...]:
    """Thresholds the post-processing needs dwell for beyond the melt threshold.

    Grain coarsening depends on the time spent above the coarsening temperature,
    which is far below the solidus, so it has to be counted separately during
    the run.
    """
    if isinstance(material, Material) and material.grain_coarsening_temperature is not None:
        return (float(material.grain_coarsening_temperature),)
    return ()


def _run_3d(
    config: ThermalSimulationConfig,
    phase: PhaseModel | None,
    on_progress: Callable[[float], None] | None,
    abort: Callable[[], bool] | None,
) -> Dict[str, Any]:
    """Run the through-thickness solver and shape its output like the 2D one.

    The plan-view fields come from :meth:`Solution3D.to_history`, so the weld
    metrics, microstructure and distortion models are shared between solvers;
    ``solution3d`` carries the depth information they cannot express.
    """
    material = config.material
    if not isinstance(material, Material):
        material = material_from_params(material)
    assert config.weld is not None

    solution = run_3d_thermal(
        nx=config.nx,
        ny=config.ny,
        nz=config.nz,
        Lx=config.Lx,
        Ly=config.Ly,
        thickness=config.thickness,
        t_end=config.t_end,
        weld=config.weld,
        material=material,
        dt=config.dt_3d,
        T0=material.T0,
        path=config.path,
        wobble=config.wobble,
        phase=phase,
        keyhole_taper=config.keyhole_taper,
        extra_dwell_temps=_extra_dwell_temps(material),
        on_progress=on_progress,
        abort=abort,
        store_frames=config.store_3d_frames,
        frame_interval=config.frame_interval,
        max_frames=config.max_stored_frames,
    )

    surface = solution.T[:, :, 0]
    if config.output_file is not None:
        os.makedirs(os.path.dirname(config.output_file) or ".", exist_ok=True)
        save_temperature_csv(config.output_file, solution.x, solution.y, surface)

    return {
        "x": solution.x,
        "y": solution.y,
        "z": solution.z,
        "T": surface,
        "history": solution.to_history(),
        "solution3d": solution,
        "top_thickness": config.top_thickness,
    }


def save_temperature_csv(path: str, x: np.ndarray, y: np.ndarray, T: np.ndarray):
    """Save temperature field as a simple CSV (flattened grid)."""
    nx, ny = T.shape
    with open(path, "w", encoding="utf-8") as f:
        f.write("x_m,y_m,T_K\n")
        for i in range(nx):
            for j in range(ny):
                f.write(f"{x[i]:.6e},{y[j]:.6e},{T[i, j]:.3f}\n")


__all__ = [
    "Solution3D",
    "ThermalHistory",
    "ThermalSimulationConfig",
    "run_thermal_simulation",
    "validate_config",
    "save_temperature_csv",
    "ValidationError",
    "WeldParams",
    "MaterialParams",
    "Material",
    "list_materials",
    "load_material",
    "WeldPath",
    "WobbleParams",
]
