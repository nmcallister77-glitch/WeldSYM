"""Compare predictions against measured macro-sections, and calibrate to them.

A conduction solve with an assumed capillary gets the trends right but cannot
know two things about a particular machine and joint: how much of the beam power
actually ends up in the plate, and how the capillary narrows with depth. Those
are exactly the two knobs an engineer can pin down by welding coupons, cutting
and etching them, and measuring the fusion boundary.

So the workflow here is:

1. Weld a bracket of coupons, varying power and travel speed at a fixed focus.
2. Measure penetration and top-surface fusion width on each macro-section.
3. :func:`compare` predicts the same quantities and reports the residuals.
4. :func:`calibrate` searches absorption efficiency and keyhole taper for the
   pair that best reproduces the measurements, and :func:`save_calibration`
   stores it so later runs start from the calibrated values.

The result is an empirical fit valid over the process window it was measured in,
for that material and thickness — not a general-purpose physical constant.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Iterable, Sequence

import numpy as np
import yaml

from .errors import ValidationError
from .materials import Material
from .simulation import ThermalSimulationConfig, run_thermal_simulation
from .thermal.solver3d import KEYHOLE_TIP_TAPER, Solution3D
from .types import WeldParams
from .weld_path import WeldPath

#: Absorption efficiencies tried when calibrating (fraction of beam power).
EFFICIENCY_CANDIDATES = (0.25, 0.4, 0.55, 0.7, 0.85, 0.95)

#: Capillary tip/surface radius ratios tried when calibrating.
TAPER_CANDIDATES = (0.2, 0.4, 0.6, 0.85)


@dataclass
class Coupon:
    """One welded and sectioned coupon: the parameters used and what was measured.

    Lengths are metres, so a 2.4 mm penetration is ``0.0024``.
    """

    label: str
    power: float  # W
    speed: float  # m/s
    thickness: float  # m
    penetration: float  # m, measured depth of the fusion boundary
    fusion_width: float | None = None  # m, at the top surface
    sigma: float = 0.0002  # m, beam 1/e² radius used for the coupon

    def validate(self) -> None:
        checks = {
            "power": self.power,
            "speed": self.speed,
            "thickness": self.thickness,
            "sigma": self.sigma,
        }
        for name, value in checks.items():
            if not value > 0:
                raise ValidationError(
                    f"Coupon {self.label!r}: {name} must be greater than 0, got {value}."
                )
        if self.penetration < 0:
            raise ValidationError(f"Coupon {self.label!r}: penetration cannot be negative.")
        if self.penetration > self.thickness * 1.001:
            raise ValidationError(
                f"Coupon {self.label!r}: measured penetration {self.penetration * 1e3:.2f} mm "
                f"exceeds the {self.thickness * 1e3:.2f} mm plate."
            )


@dataclass
class Mesh:
    """Grid and weld length used for the comparison runs.

    Deliberately coarse: a calibration search runs dozens of solves, and the
    measured quantities are boundary positions, which converge well before the
    temperature field does.
    """

    nx: int = 61
    ny: int = 31
    nz: int = 11
    weld_length: float = 0.02  # m of weld actually simulated
    width: float = 0.012  # m of plate across the weld


@dataclass
class CouponResidual:
    """Measured versus predicted geometry for one coupon."""

    label: str
    measured_penetration: float
    predicted_penetration: float
    measured_fusion_width: float | None
    predicted_fusion_width: float
    full_penetration: bool

    @property
    def penetration_error(self) -> float:
        """Predicted minus measured (m); positive means the model is too deep."""
        return self.predicted_penetration - self.measured_penetration

    @property
    def penetration_error_percent(self) -> float | None:
        if self.measured_penetration <= 0:
            return None
        return 100.0 * self.penetration_error / self.measured_penetration

    @property
    def width_error(self) -> float | None:
        if self.measured_fusion_width is None:
            return None
        return self.predicted_fusion_width - self.measured_fusion_width


@dataclass
class Comparison:
    """Residuals over a set of coupons for one (efficiency, taper) pair."""

    efficiency: float
    keyhole_taper: float
    residuals: list[CouponResidual] = field(default_factory=list)

    @property
    def penetration_rms(self) -> float:
        """Root-mean-square penetration error (m)."""
        errors = [r.penetration_error for r in self.residuals]
        return math.sqrt(sum(e**2 for e in errors) / len(errors)) if errors else 0.0

    @property
    def penetration_bias(self) -> float:
        """Mean signed penetration error (m): the systematic over/under-shoot."""
        errors = [r.penetration_error for r in self.residuals]
        return sum(errors) / len(errors) if errors else 0.0

    @property
    def width_rms(self) -> float | None:
        errors = [r.width_error for r in self.residuals if r.width_error is not None]
        if not errors:
            return None
        return math.sqrt(sum(e**2 for e in errors) / len(errors))

    def cost(self) -> float:
        """Objective the calibration minimises: relative penetration error.

        Relative, so a 3 mm and a 0.5 mm coupon carry the same weight, and width
        is folded in at half weight because it is the less reliable measurement
        (it depends on where the top bead was polished back to).
        """
        terms: list[float] = []
        for r in self.residuals:
            if r.measured_penetration > 0:
                terms.append((r.penetration_error / r.measured_penetration) ** 2)
            if r.measured_fusion_width:
                terms.append(0.5 * (r.width_error / r.measured_fusion_width) ** 2)
        return math.sqrt(sum(terms) / len(terms)) if terms else float("inf")


@dataclass
class Calibration:
    """Best-fitting absorption efficiency and keyhole taper for a coupon set."""

    efficiency: float
    keyhole_taper: float
    material: str
    cost: float  # relative RMS error after fitting
    baseline_cost: float  # the same, with the uncalibrated defaults
    comparison: Comparison
    coupons: list[Coupon]
    created: str = ""

    @property
    def improvement(self) -> float:
        """Fractional reduction in the objective, 0 if the fit did not help."""
        if self.baseline_cost <= 0:
            return 0.0
        return max(0.0, 1.0 - self.cost / self.baseline_cost)


def _predict(
    coupon: Coupon,
    material: Material,
    efficiency: float,
    keyhole_taper: float,
    mesh: Mesh,
) -> Solution3D:
    """Run the 3D solve for one coupon's parameters."""
    coupon.validate()
    length = mesh.weld_length
    Lx = length + 8.0 * coupon.sigma
    weld = WeldParams(
        power=coupon.power,
        efficiency=efficiency,
        speed=coupon.speed,
        start_pos=(4.0 * coupon.sigma, mesh.width / 2.0),
        direction="x",
        sigma=coupon.sigma,
    )
    config = ThermalSimulationConfig(
        nx=mesh.nx,
        ny=mesh.ny,
        nz=mesh.nz,
        Lx=Lx,
        Ly=mesh.width,
        t_end=length / coupon.speed,
        dt=0.001,
        weld=weld,
        material=material,
        output_file=None,
        T1=coupon.thickness,
        plate_thickness=coupon.thickness,
        solver="3d",
        keyhole_taper=keyhole_taper,
        path=WeldPath(
            start=(4.0 * coupon.sigma, mesh.width / 2.0),
            end=(4.0 * coupon.sigma + length, mesh.width / 2.0),
            speed=coupon.speed,
        ),
    )
    result = run_thermal_simulation(config)
    return result["solution3d"]


def compare(
    coupons: Sequence[Coupon],
    material: Material,
    efficiency: float,
    keyhole_taper: float = KEYHOLE_TIP_TAPER,
    mesh: Mesh | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Comparison:
    """Predict each coupon's macro-section and report the residuals."""
    if not coupons:
        raise ValidationError("Add at least one measured coupon to compare against.")
    mesh = mesh or Mesh()
    residuals: list[CouponResidual] = []
    for i, coupon in enumerate(coupons):
        solution = _predict(coupon, material, efficiency, keyhole_taper, mesh)
        section = solution.section_metrics()
        residuals.append(
            CouponResidual(
                label=coupon.label,
                measured_penetration=coupon.penetration,
                predicted_penetration=section.penetration,
                measured_fusion_width=coupon.fusion_width,
                predicted_fusion_width=section.width_top,
                full_penetration=section.full_penetration,
            )
        )
        if on_progress is not None:
            on_progress((i + 1) / len(coupons))
    return Comparison(efficiency=efficiency, keyhole_taper=keyhole_taper, residuals=residuals)


def calibrate(
    coupons: Sequence[Coupon],
    material: Material,
    efficiencies: Iterable[float] = EFFICIENCY_CANDIDATES,
    tapers: Iterable[float] = TAPER_CANDIDATES,
    baseline_efficiency: float = 0.8,
    mesh: Mesh | None = None,
    on_progress: Callable[[float], None] | None = None,
) -> Calibration:
    """Fit absorption efficiency and keyhole taper to measured macro-sections.

    A grid search rather than a gradient method: the objective is evaluated by a
    transient solve, so it is noisy at the mesh level and the useful range of
    both parameters is narrow and bounded. Cost is one solve per coupon per
    combination, which is why :class:`Mesh` defaults to a coarse grid.
    """
    mesh = mesh or Mesh()
    grid = [(e, t) for t in tapers for e in efficiencies]
    if not grid:
        raise ValidationError("Give at least one efficiency and one taper to try.")

    best: Comparison | None = None
    for i, (efficiency, taper) in enumerate(grid):
        trial = compare(coupons, material, efficiency, taper, mesh)
        if best is None or trial.cost() < best.cost():
            best = trial
        if on_progress is not None:
            on_progress((i + 1) / len(grid))
    assert best is not None

    baseline = compare(coupons, material, baseline_efficiency, KEYHOLE_TIP_TAPER, mesh)
    return Calibration(
        efficiency=best.efficiency,
        keyhole_taper=best.keyhole_taper,
        material=material.name,
        cost=best.cost(),
        baseline_cost=baseline.cost(),
        comparison=best,
        coupons=list(coupons),
        created=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def calibration_yaml(calibration: Calibration) -> str:
    """Serialise a calibration, including the coupons it was fitted to."""
    payload = {
        "material": calibration.material,
        "efficiency": float(calibration.efficiency),
        "keyhole_taper": float(calibration.keyhole_taper),
        "cost": float(calibration.cost),
        "baseline_cost": float(calibration.baseline_cost),
        "created": calibration.created,
        "coupons": [
            {
                "label": c.label,
                "power": c.power,
                "speed": c.speed,
                "thickness": c.thickness,
                "penetration": c.penetration,
                "fusion_width": c.fusion_width,
                "sigma": c.sigma,
            }
            for c in calibration.coupons
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False)


def save_calibration(path: str, calibration: Calibration) -> None:
    """Write a calibration to YAML so later runs can start from it."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(calibration_yaml(calibration))


def load_calibration(path: str) -> tuple[float, float, dict]:
    """Read a saved calibration: ``(efficiency, keyhole_taper, metadata)``."""
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    try:
        return float(data["efficiency"]), float(data["keyhole_taper"]), data
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"{path} is not a WeldSYM calibration file: {exc}") from exc


def sensitivity(
    coupon: Coupon,
    material: Material,
    efficiencies: Sequence[float] = EFFICIENCY_CANDIDATES,
    keyhole_taper: float = KEYHOLE_TIP_TAPER,
    mesh: Mesh | None = None,
) -> np.ndarray:
    """Predicted penetration (m) for one coupon over a range of efficiencies.

    Useful for showing how sharply penetration responds to absorption before
    trusting a fit to a handful of coupons.
    """
    mesh = mesh or Mesh()
    return np.array(
        [
            _predict(coupon, material, efficiency, keyhole_taper, mesh)
            .section_metrics()
            .penetration
            for efficiency in efficiencies
        ]
    )


__all__ = [
    "Calibration",
    "Comparison",
    "Coupon",
    "CouponResidual",
    "Mesh",
    "calibrate",
    "calibration_yaml",
    "compare",
    "load_calibration",
    "save_calibration",
    "sensitivity",
    "EFFICIENCY_CANDIDATES",
    "TAPER_CANDIDATES",
]
