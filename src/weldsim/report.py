"""One weld assessment, assembled from the thermal solution.

:func:`build_report` turns a finished thermal run into the answers a welding
engineer actually wants — bead and HAZ size, penetration, microstructure and
hardness, distortion, and what the wobble setting did to the heat concentration
— so the GUI and the CLI present the same numbers from the same code.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict

import numpy as np

from .distortion import DistortionEstimate, estimate_distortion
from .keyhole import KeyholeEstimate, estimate_keyhole
from .materials import Material
from .microstructure import MicrostructureResult, predict_microstructure
from .simulation import ThermalSimulationConfig
from .types import MaterialParams
from .weld_metrics import WeldMetrics, compute_weld_metrics
from .wobble_analysis import WobbleAnalysis, analyse_wobble


@dataclass
class WeldReport:
    """Everything derived from one thermal run."""

    material_name: str
    heat_input: float  # J/mm absorbed
    metrics: WeldMetrics
    keyhole: KeyholeEstimate
    microstructure: MicrostructureResult | None
    distortion: DistortionEstimate | None
    wobble: WobbleAnalysis | None
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view, with arrays dropped rather than expanded."""

        def convert(value: Any) -> Any:
            if is_dataclass(value) and not isinstance(value, type):
                return {k: convert(v) for k, v in asdict(value).items()}
            if isinstance(value, np.ndarray):
                return None
            if isinstance(value, (np.floating, np.integer)):
                return value.item()
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        data = convert(self)
        assert isinstance(data, dict)
        data["metrics"].pop("profile", None)
        return data

    def summary_lines(self) -> list[str]:
        """Compact text report, as printed by the CLI."""
        m = self.metrics
        k = self.keyhole
        lines = [
            f"Material:            {self.material_name}",
            f"Heat input:          {self.heat_input:.1f} J/mm (absorbed)",
            f"Peak temperature:    {m.peak_temperature:.0f} K",
            f"Welding mode:        {k.mode} ({k.intensity_mw_per_cm2:.2f} MW/cm² absorbed)",
            f"Penetration:         {k.depth_mm:.2f} mm"
            + (" (full penetration)" if k.full_penetration else ""),
            f"Fusion zone:         {m.fusion_width_mm:.2f} mm wide, "
            f"{m.fusion_length * 1e3:.1f} mm long",
            f"HAZ:                 {m.haz_width_mm:.2f} mm per side",
            f"Melt dwell:          {m.melt_dwell:.2f} s above solidus",
        ]
        if m.t_8_5 is not None:
            lines.append(f"HAZ cooling:         t8/5 = {m.t_8_5:.2f} s")
        if m.cooling_rate is not None:
            lines.append(f"                     {m.cooling_rate:.0f} K/s at 800 °C")

        micro = self.microstructure
        if micro is not None and micro.phases:
            phases = ", ".join(
                f"{name} {fraction * 100:.0f}%"
                for name, fraction in micro.phases.items()
                if fraction > 0.005
            )
            lines.append(f"HAZ microstructure:  {phases}")
            if micro.hardness_hv is not None:
                lines.append(
                    f"HAZ hardness:        {micro.hardness_hv:.0f} HV "
                    f"(parent {micro.base_hardness_hv:.0f} HV)"
                )
            if micro.carbon_equivalent is not None:
                lines.append(f"Carbon equivalent:   {micro.carbon_equivalent:.2f} (IIW)")

        dist = self.distortion
        if dist is not None:
            lines += [
                f"Shrinkage force:     {dist.shrinkage_force / 1e3:.1f} kN",
                f"Longitudinal:        {dist.longitudinal_shrinkage_mm:.3f} mm shortening",
                f"Transverse:          {dist.transverse_shrinkage_mm:.3f} mm closure",
                f"Angular distortion:  {dist.angular_distortion:.3f} deg",
                f"Bowing:              {dist.bowing_deflection_mm:.3f} mm",
                f"Residual stress:     {dist.peak_tensile_stress / 1e6:.0f} MPa tensile at "
                f"the weld, {dist.balancing_compressive_stress / 1e6:.0f} MPa compressive",
            ]

        wob = self.wobble
        if wob is not None and math.isfinite(wob.pitch):
            lines += [
                f"Wobble track:        {wob.swept_width_mm:.2f} mm swept, "
                f"{wob.pitch_mm:.2f} mm per loop, {wob.overlap_ratio * 100:.0f}% overlap",
                f"Heat concentration:  peak {wob.peak_energy_density / 1e6:.1f} MJ/m², "
                f"{wob.concentration_ratio:.1f}× the track mean, "
                f"{wob.peak_reduction * 100:.0f}% below a straight beam",
            ]

        if self.warnings:
            lines.append("")
            lines += [f"! {w}" for w in self.warnings]
        return lines


def build_report(
    config: ThermalSimulationConfig,
    result: Dict[str, Any],
) -> WeldReport:
    """Assemble a :class:`WeldReport` from a config and its thermal result.

    Requires a :class:`~weldsim.materials.Material` (not a bare
    :class:`~weldsim.types.MaterialParams`), because the metallurgical and
    mechanical models need the alloy data that only the material library carries.
    """
    material = config.material
    if not isinstance(material, Material):
        material = _material_from_params(material)

    x, y = result["x"], result["y"]
    history = result["history"]
    weld = config.weld
    assert weld is not None, "run_thermal_simulation always populates config.weld"

    metrics = compute_weld_metrics(x, y, history, material, weld)

    keyhole = estimate_keyhole(
        weld,
        material,
        plate_thickness=config.thickness,
        wobble_amplitude=config.wobble.amplitude if config.wobble else 0.0,
        fusion_width=metrics.fusion_width or None,
    )

    microstructure = predict_microstructure(
        x, y, history, material, metrics.t_8_5, metrics.cooling_rate
    )

    distortion = estimate_distortion(
        x,
        y,
        history,
        material,
        plate_thickness=config.thickness,
        plate_width=config.Ly,
        plate_length=config.Lx,
        penetration=keyhole.depth,
    )

    wobble = None
    if config.path is not None:
        wobble = analyse_wobble(
            config.path,
            config.wobble or _no_wobble(),
            weld,
            config.T1,
            x,
            y,
            t_end=min(config.t_end, config.path.duration),
        )

    warnings = list(metrics.warnings)
    warnings += _energy_consistency_warnings(config, material, metrics)
    warnings += microstructure.warnings
    warnings += keyhole.notes
    warnings += distortion.notes
    if wobble is not None:
        warnings += wobble.notes

    return WeldReport(
        material_name=material.name,
        heat_input=metrics.heat_input,
        metrics=metrics,
        keyhole=keyhole,
        microstructure=microstructure,
        distortion=distortion,
        wobble=wobble,
        warnings=warnings,
    )


def _energy_consistency_warnings(
    config: ThermalSimulationConfig,
    material: Material,
    metrics: WeldMetrics,
) -> list[str]:
    """Check the melted volume against the energy available to melt it.

    The thin-plate solver has neither latent heat nor surface losses, so it can
    report a molten cross-section that the absorbed energy could not possibly
    have produced. Comparing the two is the cheapest way to tell the engineer
    when the fusion width is an upper bound rather than a prediction.
    """
    weld = config.weld
    if weld is None or not metrics.melted or weld.speed <= 0:
        return []
    melted_area = metrics.fusion_width * config.T1  # m², assumed full thickness
    energy_needed = melted_area * material.density * material.melting_enthalpy  # J/m
    energy_available = weld.absorbed_power / weld.speed  # J/m
    if energy_available <= 0:
        return []
    ratio = energy_needed / energy_available
    if ratio <= 0.5:
        return []
    return [
        f"Melting the reported cross-section would need {ratio * 100:.0f} % of the "
        "absorbed energy, and a real weld spends most of it heating the "
        "surrounding plate. Treat the fusion width as an upper bound: reduce the "
        "effective thickness to the expected penetration, or work from the "
        "energy-balance depth instead."
    ]


def _no_wobble() -> Any:
    from .weld_path import WobbleParams

    return WobbleParams(amplitude=0.0, frequency=0.0)


def _material_from_params(params: MaterialParams) -> Material:
    """Wrap bare thermal properties as a Material so the reports still run.

    Phase-change and alloy data are unknown in this case, so the defaults of
    :class:`~weldsim.materials.Material` apply and the metallurgical output is
    generic rather than alloy-specific.
    """
    return Material(
        name="Custom (thermal properties only)",
        density=params.rho,
        thermal_conductivity=params.k,
        specific_heat=params.cp,
        T0=params.T0,
    )


__all__ = ["WeldReport", "build_report"]
