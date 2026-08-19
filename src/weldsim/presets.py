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

        weld = WeldParams(
            power=self.power,
            efficiency=self.efficiency,
            speed=self.speed,
            start_pos=self.start,
            direction="x",
            sigma=self.sigma,
        )
        path = WeldPath(start=self.start, end=self.end, speed=self.speed)
        wobble = WobbleParams(
            amplitude=self.wobble_amp_um * 1e-6,
            frequency=self.wobble_freq_hz,
            pattern=self.wobble_pattern,
        )

        thickness = self.plate_thickness_mm / 1000.0
        t_end = self.t_end
        if t_end is None:
            # Long enough tail for t8/5 measurement; 1.3x weld duration in 3D.
            length = (
                (self.end[0] - self.start[0]) ** 2 + (self.end[1] - self.start[1]) ** 2
            ) ** 0.5
            t_end = min(max(1.3 * length / self.speed, 0.1), 50.0)

        probe = self.probe
        if probe is None:
            mid_x = (self.start[0] + self.end[0]) / 2.0
            mid_y = (self.start[1] + self.end[1]) / 2.0
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
    )
}
