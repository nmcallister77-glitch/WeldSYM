"""Distortion and residual-stress estimates from the weld thermal cycle.

This is the inherent-strain (shrinkage-force) method, the classical fast
alternative to a thermo-elastic-plastic FE run. The argument is:

1. Material heated past the temperature where its yield strength is exhausted by
   thermal expansion yields in compression, and cannot recover that strain on
   cooling. What is left behind is the *inherent strain*.
2. Integrating the inherent strain over the weld cross-section gives a
   shrinkage force, and its offset from the plate's neutral axis gives a bending
   moment.
3. Shortening, transverse shrinkage, angular distortion and bowing all follow
   from that force and moment by elastic beam relations.

The inherent-strain saturation used here is Okerblom's one-dimensional bar
result: nothing below the yield temperature difference, growing to a plateau of
``α·ΔT_y`` once the peak temperature is twice that.

Because the thermal model is a thin-plate one, the through-thickness
distribution is assumed rather than computed: the plastic zone is taken to
extend from the top surface down to the fusion depth. That makes angular
distortion vanish for a full-penetration pass, which understates a real V-groove
weld — the numbers are for comparing parameter sets, not for qualification.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .materials import Material
from .thermal.fd_solver import ThermalHistory


@dataclass
class DistortionEstimate:
    """Distortion and residual stress for one weld pass."""

    shrinkage_force: float  # N
    longitudinal_shrinkage: float  # m, shortening along the weld
    transverse_shrinkage: float  # m, closure across the weld
    angular_distortion: float  # deg, rotation between the two halves
    bowing_deflection: float  # m, mid-span camber out of plane
    plastic_zone_width: float  # m, total width that yielded
    plastic_zone_depth: float  # m, assumed depth of the yielded layer
    peak_tensile_stress: float  # Pa, longitudinal residual stress at the weld
    balancing_compressive_stress: float  # Pa, in the surrounding plate
    yield_temperature_rise: float  # K, ΔT that first causes yielding
    notes: list[str] = field(default_factory=list)

    @property
    def longitudinal_shrinkage_mm(self) -> float:
        return self.longitudinal_shrinkage * 1e3

    @property
    def transverse_shrinkage_mm(self) -> float:
        return self.transverse_shrinkage * 1e3

    @property
    def bowing_deflection_mm(self) -> float:
        return self.bowing_deflection * 1e3


def inherent_strain(
    T_peak: np.ndarray,
    material: Material,
) -> tuple[np.ndarray, float]:
    """Residual (inherent) strain magnitude per cell, and the yield ΔT.

    Returns
    -------
    strain : np.ndarray
        Magnitude of the compressive plastic strain left after cooling (-).
    delta_T_yield : float
        Temperature rise at which yielding starts, ``σ_y / (E·α)`` (K).
    """
    mech = material.mechanical
    alpha = mech.thermal_expansion
    delta_T_yield = mech.yield_strain / alpha

    delta_T = np.maximum(T_peak - material.T0, 0.0)
    strain = np.clip(alpha * (delta_T - delta_T_yield), 0.0, alpha * delta_T_yield)
    return strain, delta_T_yield


def estimate_distortion(
    x: np.ndarray,
    y: np.ndarray,
    history: ThermalHistory,
    material: Material,
    plate_thickness: float,
    plate_width: float,
    plate_length: float,
    penetration: float | None = None,
) -> DistortionEstimate:
    """Estimate distortion and residual stress for a single weld pass.

    Parameters
    ----------
    plate_thickness, plate_width, plate_length : float
        Plate dimensions (m). Width is across the weld, length along it.
    penetration : float | None
        Fusion depth (m) from the keyhole estimate; sets the depth of the
        yielded layer. Defaults to the full thickness, i.e. a through-thickness
        plastic zone with no angular distortion.
    """
    mech = material.mechanical
    E = mech.youngs_modulus
    dy = float(y[1] - y[0]) if y.size > 1 else 0.0

    strain, delta_T_yield = inherent_strain(history.T_peak, material)

    # Take the transverse section through the hottest station: that is where the
    # shrinkage force is highest and where a macro-section would be cut.
    station = int(np.unravel_index(np.argmax(history.T_peak), history.T_peak.shape)[0])
    strain_section = strain[station, :]

    depth = plate_thickness if penetration is None else min(penetration, plate_thickness)
    depth = max(depth, 0.0)

    # Shrinkage force: the elastic force needed to pull the yielded layer back to
    # the length of the surrounding plate.
    strain_integral = float(strain_section.sum()) * dy  # m of "missing" length
    force = E * strain_integral * depth

    plastic_width = float((strain_section > 0).sum()) * dy
    strain_mean = strain_integral / plastic_width if plastic_width > 0 else 0.0

    plate_area = plate_width * plate_thickness
    longitudinal = force * plate_length / (E * plate_area) if plate_area > 0 else 0.0

    # Transverse closure: the same missing length, acting across the weld, scaled
    # by how much of the thickness actually yielded.
    depth_fraction = depth / plate_thickness if plate_thickness > 0 else 0.0
    transverse = strain_integral * depth_fraction

    # Angular distortion: the yielded layer sits above the mid-plane, so its
    # contraction bends the plate. Curvature of a strip carrying strain over the
    # top `depth`, times the width over which it acts.
    if plate_thickness > 0 and 0.0 < depth < plate_thickness:
        curvature = 6.0 * strain_mean * depth * (plate_thickness - depth) / plate_thickness**3
    else:
        curvature = 0.0
    angular_deg = math.degrees(curvature * plastic_width)

    # Bowing: the shrinkage force offset from the neutral axis bends the plate
    # along its length.
    second_moment = plate_width * plate_thickness**3 / 12.0
    eccentricity = max(0.0, (plate_thickness - depth) / 2.0)
    if second_moment > 0:
        bowing = (force * eccentricity) * plate_length**2 / (8.0 * E * second_moment)
    else:
        bowing = 0.0

    # Residual stress: the weld itself ends up at yield in tension, balanced by
    # compression spread over the remaining cross-section.
    tensile = mech.yield_stress
    remaining_area = max(plate_area - plastic_width * depth, 1e-12)
    compressive = force / remaining_area

    notes: list[str] = []
    if strain_integral == 0.0:
        notes.append(
            "No point in the plate got hot enough to yield, so this pass leaves "
            "no residual distortion."
        )
    if depth >= plate_thickness and plate_thickness > 0:
        notes.append(
            "The yielded layer is assumed to run through the full thickness, so "
            "angular distortion is reported as zero; a partial-penetration pass "
            "or a V-groove would bow the plate toward the weld."
        )
    if compressive > mech.yield_stress:
        notes.append(
            "The balancing compressive stress exceeds yield, meaning the plate "
            "would buckle rather than simply shorten: expect out-of-plane "
            "distortion well above the bowing figure."
        )
    if plate_length > 0 and longitudinal / plate_length > 1e-3:
        notes.append(
            "Longitudinal shortening exceeds 0.1 % of the plate length: allow for "
            "it in the cut length or restrain the assembly."
        )

    return DistortionEstimate(
        shrinkage_force=force,
        longitudinal_shrinkage=longitudinal,
        transverse_shrinkage=transverse,
        angular_distortion=angular_deg,
        bowing_deflection=bowing,
        plastic_zone_width=plastic_width,
        plastic_zone_depth=depth,
        peak_tensile_stress=tensile,
        balancing_compressive_stress=compressive,
        yield_temperature_rise=delta_T_yield,
        notes=notes,
    )


__all__ = ["DistortionEstimate", "estimate_distortion", "inherent_strain"]
