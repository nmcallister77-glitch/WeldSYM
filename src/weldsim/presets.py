"""Named presets for common weld scenarios.

A preset is a complete :class:`ThermalSimulationConfig` that a user can run with
one click in the GUI or via ``--preset`` on the CLI. Each preset is a starting
point, not a qualified procedure — the real machine-specific numbers (absorption
and keyhole taper especially) should be fitted on the Measured vs predicted page.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .materials import load_material
from .simulation import ThermalSimulationConfig
from .types import WeldParams
from .weld_path import WeldPath, WobbleParams


@dataclass
class Preset:
    """Human-readable description plus the parameters that define the scenario."""

    name: str
    description: str
    material: str
    material_at_temperature: float
    power: float
    efficiency: float
    speed: float
    sigma: float
    start: tuple[float, float]
    end: tuple[float, float]
    Lx: float
    Ly: float
    top_thickness_mm: float
    bottom_thickness_mm: float
    wobble_amp_um: float = 0.0
    wobble_freq_hz: float = 0.0
    wobble_pattern: str = "circle"
    solver: str = "3d"
    nx: int = 61
    ny: int = 81
    nz: int = 9
    t_end: float | None = None
    dt: float = 0.01
    probe: tuple[float, float] | None = None
    keyhole_taper: float | None = None
    custom_krcp: tuple[float, float, float, float] | None = None
    path_type: str = "line"  # line, circle, spiral
    center: tuple[float, float] = (0.0, 0.0)
    radius: float = 0.005
    turns: float = 1.0
    r_start: float = 0.0
    r_end: float = 0.005
    notes: list[str] = field(default_factory=list)

    @property
    def plate_thickness_mm(self) -> float:
        """Total stack thickness for the 2t lap joint."""
        return self.top_thickness_mm + self.bottom_thickness_mm

    @property
    def T1_mm(self) -> float:
        """Depth the surface flux is spread over: the full stack for a 2t joint."""
        return self.plate_thickness_mm

    def make_config(self) -> ThermalSimulationConfig:
        """Build a ready-to-run config from this preset."""
        if self.custom_krcp is not None:
            from .types import MaterialParams

            k, rho, cp, t0 = self.custom_krcp
            material: object = MaterialParams(k=k, rho=rho, cp=cp, T0=t0)
        else:
            material = load_material(self.material, self.material_at_temperature)

        if self.path_type == "circle":
            path = WeldPath.circle(
                center=self.center,
                radius=self.radius,
                speed=self.speed,
                turns=self.turns,
            )
        elif self.path_type == "spiral":
            path = WeldPath.spiral(
                center=self.center,
                r_start=self.r_start,
                r_end=self.r_end,
                speed=self.speed,
                turns=self.turns,
            )
        else:
            path = WeldPath(start=self.start, end=self.end, speed=self.speed)

        weld = WeldParams(
            power=self.power,
            efficiency=self.efficiency,
            speed=self.speed,
            start_pos=path.start,
            direction="x",
            sigma=self.sigma,
        )
        wobble = WobbleParams(
            amplitude=self.wobble_amp_um * 1e-6,
            frequency=self.wobble_freq_hz,
            pattern=self.wobble_pattern,
        )

        thickness = self.plate_thickness_mm / 1000.0
        t_end = self.t_end
        if t_end is None:
            # Long enough tail for t8/5 measurement; 1.3x weld duration in 3D.
            t_end = min(max(1.3 * path.length / self.speed, 0.1), 50.0)

        probe = self.probe
        if probe is None:
            mid_x = (path.start[0] + path.end[0]) / 2.0
            mid_y = (path.start[1] + path.end[1]) / 2.0
            probe = (mid_x, mid_y)

        return ThermalSimulationConfig(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            Lx=self.Lx,
            Ly=self.Ly,
            t_end=t_end,
            dt=self.dt,
            weld=weld,
            material=material,
            output_file=None,
            T1=thickness,
            plate_thickness=thickness,
            top_thickness=self.top_thickness_mm / 1000.0,
            path=path,
            wobble=wobble,
            probe=probe if self.solver == "2d" else None,
            solver=self.solver,
            keyhole_taper=self.keyhole_taper,
        )


#: Presets keyed by the name shown in the GUI / CLI.
PRESETS: dict[str, Preset] = {
    "2t copper lap joint": Preset(
        name="2t copper lap joint",
        description=(
            "Two copper sheets in a 2t lap stack, welded from above with a "
            "high-brightness NIR laser. The beam passes through the top sheet "
            "and into the bottom one. Total thickness = top + bottom."
        ),
        material="Copper",
        material_at_temperature=1000.0,
        power=4000.0,
        efficiency=0.35,
        speed=0.03,
        sigma=100e-6,
        start=(0.005, 0.005),
        end=(0.025, 0.005),
        Lx=0.04,
        Ly=0.02,
        top_thickness_mm=1.0,
        bottom_thickness_mm=1.0,
        wobble_amp_um=0.0,
        wobble_freq_hz=0.0,
        wobble_pattern="circle",
        solver="3d",
        nx=61,
        ny=81,
        nz=9,
        t_end=0.6,
        dt=0.01,
        keyhole_taper=0.4,
        notes=[
            "Copper is highly reflective: 35 % absorption is a starting guess and should be "
            "calibrated on a bracket of coupons.",
            "The 2t stack is two 1 mm sheets by default; set top and bottom thickness "
            "independently for dissimilar-gauge lap joints.",
            "The small domain and coarse through-thickness grid keep the 3D solve fast "
            "despite copper's high thermal diffusivity.",
        ],
    ),
    "1t copper butt weld": Preset(
        name="1t copper butt weld",
        description="Single 1 mm copper sheet, conduction-mode butt weld.",
        material="Copper",
        material_at_temperature=1000.0,
        power=3000.0,
        efficiency=0.30,
        speed=0.04,
        sigma=80e-6,
        start=(0.005, 0.005),
        end=(0.025, 0.005),
        Lx=0.04,
        Ly=0.01,
        top_thickness_mm=0.0,
        bottom_thickness_mm=1.0,
        wobble_amp_um=0.0,
        wobble_freq_hz=0.0,
        wobble_pattern="circle",
        solver="3d",
        nx=61,
        ny=41,
        nz=9,
        t_end=0.6,
        dt=0.01,
        keyhole_taper=0.5,
        notes=["A thin copper butt joint with no wobble."],
    ),
    "2t aluminum lap joint": Preset(
        name="2t aluminum lap joint",
        description="Two 1.2 mm aluminum sheets in a 2t lap stack.",
        material="Aluminum",
        material_at_temperature=1000.0,
        power=3500.0,
        efficiency=0.40,
        speed=0.04,
        sigma=120e-6,
        start=(0.005, 0.005),
        end=(0.025, 0.005),
        Lx=0.04,
        Ly=0.02,
        top_thickness_mm=1.2,
        bottom_thickness_mm=1.2,
        wobble_amp_um=0.0,
        wobble_freq_hz=0.0,
        wobble_pattern="circle",
        solver="3d",
        nx=61,
        ny=81,
        nz=11,
        t_end=0.5,
        dt=0.01,
        keyhole_taper=0.5,
        notes=[
            "Aluminum 6061 is reflective and conductive; efficiency should be calibrated.",
            "1.2 mm + 1.2 mm lap stack.",
        ],
    ),
    "aluminum circle weld": Preset(
        name="aluminum circle weld",
        description="Circular seam on a 2 mm aluminum plate.",
        material="Aluminum",
        material_at_temperature=1000.0,
        power=3000.0,
        efficiency=0.40,
        speed=0.05,
        sigma=100e-6,
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        Lx=0.02,
        Ly=0.02,
        top_thickness_mm=0.0,
        bottom_thickness_mm=2.0,
        wobble_amp_um=0.0,
        wobble_freq_hz=0.0,
        wobble_pattern="circle",
        path_type="circle",
        center=(0.01, 0.01),
        radius=0.006,
        turns=1.0,
        solver="3d",
        nx=61,
        ny=61,
        nz=9,
        t_end=1.0,
        dt=0.01,
        keyhole_taper=0.5,
        notes=["A full circular weld on a 2 mm 6061 aluminum plate."],
    ),
    "copper spiral weld": Preset(
        name="copper spiral weld",
        description="Spiral weld from center out on a 1 mm copper plate.",
        material="Copper",
        material_at_temperature=1000.0,
        power=2500.0,
        efficiency=0.35,
        speed=0.04,
        sigma=100e-6,
        start=(0.0, 0.0),
        end=(0.0, 0.0),
        Lx=0.02,
        Ly=0.02,
        top_thickness_mm=0.0,
        bottom_thickness_mm=1.0,
        wobble_amp_um=0.0,
        wobble_freq_hz=0.0,
        wobble_pattern="circle",
        path_type="spiral",
        center=(0.01, 0.01),
        r_start=0.001,
        r_end=0.007,
        turns=1.5,
        solver="3d",
        nx=61,
        ny=61,
        nz=9,
        t_end=1.0,
        dt=0.01,
        keyhole_taper=0.4,
        notes=["Archimedean spiral weld on a 1 mm copper plate."],
    ),
}
