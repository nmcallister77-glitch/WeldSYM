"""Penetration-depth and welding-mode estimates for a focused beam.

The 2D thermal solver is a thin-plate model: it cannot resolve a keyhole, which
is a three-dimensional vapour capillary. What it can do is tell us the absorbed
intensity and the melt width, and those are enough for the standard engineering
estimate of penetration depth and of which welding mode the parameters sit in.

Two regimes matter:

conduction mode
    The surface stays below the boiling point. The pool is shallow and roughly
    hemispherical, so depth follows the melt width.
keyhole mode
    Intensity is high enough to boil the surface; recoil pressure opens a
    capillary and depth is set by how much metal the beam can melt per unit
    length of travel.

For the resolved free-surface treatment, run the OpenFOAM ``laserKeyholeVoF``
solver in ``keyhole-cfd/``; this module is the fast estimate used for parameter
selection.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .materials import Material
from .optics import KeyholeAbsorptance, fresnel_absorptivity, keyhole_absorptance
from .types import WeldParams

GAS_CONSTANT = 8.314462618

#: Absorbed intensity above which metals form a keyhole, ~1 MW/cm². A widely
#: used engineering threshold for laser welding of steel and titanium; it is a
#: transition band rather than a sharp limit, hence TRANSITION_BAND below.
KEYHOLE_THRESHOLD_INTENSITY = 1.0e10  # W/m²

#: How far below the threshold intensity the transition regime reaches.
TRANSITION_BAND = 3.0

#: Fraction of absorbed energy that goes into melting rather than conduction
#: losses. Keyhole welding is efficient because the beam is delivered inside the
#: capillary; conduction-mode welding wastes far more into the plate. The
#: theoretical ceiling for a moving source is ~0.48 (Fuerschbach).
MELTING_EFFICIENCY_KEYHOLE = 0.40
MELTING_EFFICIENCY_CONDUCTION = 0.20


@dataclass
class KeyholeEstimate:
    """Welding mode and penetration estimate for one parameter set."""

    mode: str  # "conduction", "transition" or "keyhole"
    peak_intensity: float  # W/m², absorbed, at the beam centre
    threshold_intensity: float  # W/m²
    depth: float  # m, estimated penetration below the surface
    width: float  # m, width the depth estimate assumes for the molten channel
    aspect_ratio: float  # depth / width
    melting_efficiency: float  # fraction of absorbed power used for melting
    full_penetration: bool
    plate_thickness: float  # m
    notes: list[str] = field(default_factory=list)
    flat_absorptivity: float | None = None
    keyhole_absorptance: float | None = None
    recoil_onset_temperature: float | None = None
    capillary_pressure: float | None = None

    @property
    def depth_mm(self) -> float:
        return self.depth * 1e3

    @property
    def intensity_mw_per_cm2(self) -> float:
        """Peak absorbed intensity in the units process engineers quote (MW/cm²)."""
        return self.peak_intensity * 1e-10


def peak_intensity(weld: WeldParams, wobble_amplitude: float = 0.0) -> float:
    """Absorbed intensity at the beam centre (W/m²).

    A Gaussian of standard deviation ``sigma`` carrying power ``P`` peaks at
    ``P / (2 π σ²)``. Wobble spreads the same power over a wider swept track, so
    the time-averaged intensity a point sees falls by the ratio of the swept
    width to the beam width — this is the mechanism that lets wobble weld
    gap-bridging joints without cutting through them.
    """
    intensity = weld.absorbed_power / (2.0 * math.pi * weld.sigma**2)
    if wobble_amplitude > 0.0:
        spread = (2.0 * weld.sigma + 2.0 * wobble_amplitude) / (2.0 * weld.sigma)
        intensity /= spread
    return intensity


def recoil_pressure(temperature: float, material: Material) -> float:
    """Estimate vapour recoil pressure at the specified surface temperature."""
    if material.vapor_molar_mass is None:
        raise ValueError(f"Material {material.name!r} has no vapour molar mass")
    if temperature <= 0.0 or material.boiling <= 0.0:
        raise ValueError("temperature and boiling temperature must be positive")
    exponent = (
        material.latent_heat_vaporization
        * material.vapor_molar_mass
        * (temperature - material.boiling)
        / (GAS_CONSTANT * temperature * material.boiling)
    )
    return material.recoil_prefactor * material.ambient_pressure * math.exp(exponent)


def recoil_onset_temperature(material: Material, beam_radius: float) -> float | None:
    """Return the temperature at which recoil balances beam-scale capillary pressure."""
    if material.vapor_molar_mass is None:
        return None
    if beam_radius <= 0.0:
        raise ValueError("beam_radius must be positive")
    capillary_pressure = 2.0 * material.surface_tension / beam_radius
    reference_pressure = material.recoil_prefactor * material.ambient_pressure
    denominator = material.latent_heat_vaporization * material.vapor_molar_mass
    if capillary_pressure <= 0.0 or reference_pressure <= 0.0 or denominator <= 0.0:
        raise ValueError("recoil and capillary properties must be positive")
    inverse_temperature = 1.0 / material.boiling - (
        GAS_CONSTANT * math.log(capillary_pressure / reference_pressure) / denominator
    )
    if inverse_temperature <= 0.0:
        raise ValueError("recoil onset temperature is not positive")
    return 1.0 / inverse_temperature


def keyhole_threshold_intensity(material: Material, weld: WeldParams) -> float:
    """Return an engineering estimate, not an exact keyhole threshold."""
    onset = recoil_onset_temperature(material, weld.sigma)
    if onset is None:
        return KEYHOLE_THRESHOLD_INTENSITY
    peclet = weld.speed * weld.sigma / material.thermal_diffusivity
    return (
        material.thermal_conductivity
        * (onset - material.T0)
        / (weld.sigma * math.sqrt(math.pi / 2.0))
        * math.sqrt(1.0 + peclet)
    )


def estimate_keyhole(
    weld: WeldParams,
    material: Material,
    plate_thickness: float,
    wobble_amplitude: float = 0.0,
    fusion_width: float | None = None,
) -> KeyholeEstimate:
    """Estimate welding mode and penetration depth.

    Parameters
    ----------
    plate_thickness : float
        Plate thickness (m); used to flag full penetration and to cap depth.
    wobble_amplitude : float
        Beam oscillation amplitude (m). Widens the channel and lowers intensity.
    fusion_width : float | None
        Measured fusion-zone width from the thermal solution (m). When given it
        sets the channel width, which is more faithful than the beam diameter.

    Returns
    -------
    KeyholeEstimate
    """
    intensity = peak_intensity(weld, wobble_amplitude)
    threshold = keyhole_threshold_intensity(material, weld)

    if intensity >= threshold:
        mode = "keyhole"
        efficiency = MELTING_EFFICIENCY_KEYHOLE
    elif intensity >= threshold / TRANSITION_BAND:
        mode = "transition"
        efficiency = 0.5 * (MELTING_EFFICIENCY_KEYHOLE + MELTING_EFFICIENCY_CONDUCTION)
    else:
        mode = "conduction"
        efficiency = MELTING_EFFICIENCY_CONDUCTION

    # Channel width: the beam plus its wobble sweep, unless the thermal solution
    # measured a wider pool.
    width = 2.0 * weld.sigma + 2.0 * wobble_amplitude
    if fusion_width is not None and fusion_width > width:
        width = fusion_width

    # Energy balance per unit length of weld: the melted cross-section is the
    # absorbed energy per unit length divided by the enthalpy needed to melt
    # unit volume. Dividing by the channel width turns that area into a depth.
    energy_per_length = efficiency * weld.absorbed_power / weld.speed  # J/m
    melt_enthalpy = material.density * material.melting_enthalpy  # J/m³
    depth = energy_per_length / (melt_enthalpy * width)

    notes: list[str] = []
    if mode == "conduction":
        # A conduction pool cannot be deeper than it is wide; it is a shallow
        # dish, not a capillary.
        cap = 0.5 * width
        if depth > cap:
            notes.append(
                "Conduction-mode depth capped at half the pool width: without a "
                "keyhole the pool cannot be deeper than a hemisphere."
            )
            depth = cap

    full_penetration = depth >= plate_thickness
    if full_penetration:
        notes.append(
            f"Estimated depth ({depth * 1e3:.2f} mm) reaches through the "
            f"{plate_thickness * 1e3:.2f} mm plate: expect full penetration and "
            "check for dropout or root sag."
        )
        depth = plate_thickness

    if wobble_amplitude > 0.0:
        notes.append(
            f"Wobble widens the channel to {width * 1e3:.2f} mm and cuts peak "
            "intensity, which trades penetration for gap bridging."
        )
    if intensity > 100.0 * threshold:
        notes.append(
            "Intensity is two orders of magnitude above the keyhole threshold: "
            "expect a deep unstable keyhole with spatter and porosity risk."
        )

    flat_absorptivity = None
    predicted_absorptance = None
    if material.optical_n is not None and material.optical_k is not None:
        flat_absorptivity = fresnel_absorptivity(1.0, material.optical_n, material.optical_k)
        if mode == "conduction":
            predicted_absorptance = flat_absorptivity
        else:
            predicted_absorptance = keyhole_absorptance(
                material.optical_n,
                material.optical_k,
                depth=depth,
                mouth_radius=weld.sigma,
            ).absorptance
        if abs(weld.efficiency - predicted_absorptance) > 0.15:
            notes.append(
                f"Fresnel ray trace predicts ~{predicted_absorptance * 100.0:.0f} % "
                f"absorption for this keyhole (flat melt "
                f"{flat_absorptivity * 100.0:.0f} %); efficiency is set to "
                f"{weld.efficiency * 100.0:.0f} %. Calibrate against a macro-section "
                "before trusting depth."
            )

    onset_temperature = recoil_onset_temperature(material, weld.sigma)
    capillary_pressure = 2.0 * material.surface_tension / weld.sigma

    return KeyholeEstimate(
        mode=mode,
        peak_intensity=intensity,
        threshold_intensity=threshold,
        depth=depth,
        width=width,
        aspect_ratio=depth / width if width > 0 else 0.0,
        melting_efficiency=efficiency,
        full_penetration=full_penetration,
        plate_thickness=plate_thickness,
        notes=notes,
        flat_absorptivity=flat_absorptivity,
        keyhole_absorptance=predicted_absorptance,
        recoil_onset_temperature=onset_temperature,
        capillary_pressure=capillary_pressure,
    )


__all__ = [
    "KeyholeEstimate",
    "estimate_keyhole",
    "peak_intensity",
    "recoil_pressure",
    "recoil_onset_temperature",
    "keyhole_threshold_intensity",
    "fresnel_absorptivity",
    "keyhole_absorptance",
    "KeyholeAbsorptance",
    "GAS_CONSTANT",
    "KEYHOLE_THRESHOLD_INTENSITY",
]
