"""Streamlit GUI for Weld Sim — weld assessment, wobble, and OpenFOAM export."""

from __future__ import annotations

import csv
import io
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the '3d' projection

from weldsim.calibration import (
    EFFICIENCY_CANDIDATES,
    TAPER_CANDIDATES,
    Calibration,
    Comparison,
    Coupon,
    Mesh,
    calibrate,
    calibration_yaml,
    compare,
)
from weldsim.errors import WeldSimError
from weldsim.materials import Material, list_materials, load_material
from weldsim.report import WeldReport, build_report
from weldsim.simulation import (
    Solution3D,
    ThermalSimulationConfig,
    run_thermal_simulation,
)
from weldsim.thermal.solver3d import DT_SAFETY, stable_dt
from weldsim.types import WeldParams
from weldsim.wobble_analysis import energy_density_map, sampling_window
from weldsim.weld_path import (
    WeldPath,
    WobbleParams,
    beam_trajectory,
    heat_signature,
    wobble_animation_gif,
)


def _show(fig: plt.Figure) -> None:
    """Draw a figure and drop it: pyplot keeps every figure alive otherwise."""
    st.pyplot(fig)
    plt.close(fig)


def _plot_temperature_2d(x: np.ndarray, y: np.ndarray, T: np.ndarray) -> plt.Figure:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots(figsize=(6, 4))
    cmap = ax.pcolormesh(X * 1e3, Y * 1e3, T, shading="auto", cmap="inferno")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperature field (K)")
    fig.colorbar(cmap, ax=ax, label="T (K)")
    return fig


def _plot_temperature_3d(x: np.ndarray, y: np.ndarray, T: np.ndarray) -> plt.Figure:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(X * 1e3, Y * 1e3, T, cmap=cm.inferno, linewidth=0, antialiased=True)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("T (K)")
    ax.set_title("3D temperature surface")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="T (K)")
    return fig


def _plot_heat_signature(
    x: np.ndarray,
    y: np.ndarray,
    Q: np.ndarray,
    x_traj: np.ndarray | None = None,
    y_traj: np.ndarray | None = None,
) -> plt.Figure:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots(figsize=(7, 5))
    # log scale makes low-level wobble tracks visible
    Qplot = Q.copy()
    Qplot[Qplot == 0] = Qplot[Qplot > 0].min() * 1e-6
    cmap = ax.pcolormesh(
        X * 1e3,
        Y * 1e3,
        Qplot,
        shading="auto",
        cmap="hot",
        norm=plt.matplotlib.colors.LogNorm(vmin=Qplot[Qplot > 0].min(), vmax=Qplot.max()),
    )
    if x_traj is not None and y_traj is not None:
        ax.plot(x_traj * 1e3, y_traj * 1e3, "c-", alpha=0.4, lw=0.5, label="beam path")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Heat signature (J/m³) — log scale")
    fig.colorbar(cmap, ax=ax, label="Q (J/m³)")
    ax.legend()
    return fig


def _plot_temperature_profile(
    x: np.ndarray,
    T: np.ndarray,
    xlabel: str,
    title: str,
    material: Material,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x * 1e3, T, "k-")
    ax.axhline(material.solidus, color="orange", ls="--", label="solidus")
    ax.axhline(material.liquidus, color="red", ls="--", label="liquidus")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("T (K)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def _plot_probe_history(
    t: np.ndarray,
    T_probe: np.ndarray,
    material: Material,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, T_probe, "b-")
    ax.axhline(material.solidus, color="orange", ls="--", label="solidus")
    ax.axhline(material.liquidus, color="red", ls="--", label="liquidus")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("T (K)")
    ax.set_title("Time-temperature history at probe")
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def _plot_beam_path(
    x_traj: np.ndarray,
    y_traj: np.ndarray,
    Lx: float,
    Ly: float,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x_traj * 1e3, y_traj * 1e3, "r-", lw=0.5, alpha=0.7)
    ax.scatter(x_traj[0] * 1e3, y_traj[0] * 1e3, c="green", s=50, label="start")
    ax.scatter(x_traj[-1] * 1e3, y_traj[-1] * 1e3, c="blue", s=50, label="end")
    ax.set_xlim(0, Lx * 1e3)
    ax.set_ylim(0, Ly * 1e3)
    ax.set_aspect("equal")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Wobbled beam path")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)
    return fig


def _plot_zone_map(
    x: np.ndarray,
    y: np.ndarray,
    T_peak: np.ndarray,
    material: Material,
) -> plt.Figure:
    """Plan view of the weld: fusion zone, HAZ bands and unaffected plate."""
    X, Y = np.meshgrid(x, y, indexing="ij")

    # Contiguous temperature bands from cold plate up to the fusion zone, so the
    # colour bar reads as a legend of metallurgical zones rather than kelvin.
    levels = [float(T_peak.min()), material.haz_outer_temperature]
    labels = ["Unaffected"]
    for zone in sorted(material.haz_zones, key=lambda z: z.t_min):
        upper = min(zone.t_max, material.solidus)
        if upper > levels[-1]:
            levels.append(upper)
            labels.append(zone.name)
    if material.solidus > levels[-1]:
        levels.append(material.solidus)
        labels.append("HAZ" if not material.haz_zones else "Near fusion boundary")
    levels.append(max(material.liquidus, float(T_peak.max()) + 1.0))
    labels.append("Fusion zone")

    fig, ax = plt.subplots(figsize=(7, 4))
    filled = ax.contourf(X * 1e3, Y * 1e3, T_peak, levels=levels, cmap="inferno")
    ax.contour(X * 1e3, Y * 1e3, T_peak, levels=[material.solidus], colors="cyan", linewidths=1.2)
    cbar = fig.colorbar(filled, ax=ax)
    cbar.set_ticks([(levels[i] + levels[i + 1]) / 2 for i in range(len(levels) - 1)])
    cbar.set_ticklabels(labels)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Weld zones from peak temperature (cyan = fusion boundary)")
    return fig


#: Cell updates per second the 3D solver sustains on a typical laptop core,
#: measured on the default grid. Only used to warn about long runs.
CELL_UPDATE_RATE = 3.0e7


def _estimate_3d_runtime(
    nx: int,
    ny: int,
    nz: int,
    Lx: float,
    Ly: float,
    thickness: float,
    t_end: float,
    material: Material,
) -> float:
    """Rough wall-clock estimate (s) for a 3D solve, from its stable time step."""
    dx, dy = Lx / max(nx - 1, 1), Ly / max(ny - 1, 1)
    dz = thickness / max(nz - 1, 1)
    alpha = material.thermal_conductivity / (material.density * material.specific_heat)
    dt = DT_SAFETY * stable_dt(alpha, dx, dy, dz)
    return nx * ny * nz * math.ceil(t_end / dt) / CELL_UPDATE_RATE


def _plot_cross_section(solution: Solution3D) -> plt.Figure:
    """Transverse macro-section from the 3D solve: the weld as it would be cut."""
    index = solution.section_index()
    T_peak = solution.section(index)
    Y, Z = np.meshgrid(solution.y * 1e3, solution.z * 1e3, indexing="ij")

    fig, ax = plt.subplots(figsize=(7, 4))
    filled = ax.contourf(Y, Z, T_peak, levels=24, cmap="inferno")
    ax.contour(Y, Z, T_peak, levels=[solution.solidus], colors="cyan", linewidths=1.4)
    ax.contour(Y, Z, T_peak, levels=[solution.haz_limit], colors="lime", linewidths=1.0)
    fig.colorbar(filled, ax=ax, label="Peak T (K)")
    ax.invert_yaxis()
    ax.set_xlabel("y across the weld (mm)")
    ax.set_ylabel("depth below the surface (mm)")
    ax.set_title(
        f"Cross-section at x = {solution.x[index] * 1e3:.1f} mm "
        "(cyan = fusion boundary, green = HAZ limit)"
    )
    ax.set_aspect("equal")
    return fig


def _plot_longitudinal_section(solution: Solution3D) -> plt.Figure:
    """Depth profile along the weld centreline, so the run-in and crater show up."""
    centre = int(np.argmax(solution.T_peak.max(axis=(0, 2))))
    T_peak = solution.T_peak[:, centre, :]
    X, Z = np.meshgrid(solution.x * 1e3, solution.z * 1e3, indexing="ij")

    fig, ax = plt.subplots(figsize=(7, 3))
    filled = ax.contourf(X, Z, T_peak, levels=24, cmap="inferno")
    ax.contour(X, Z, T_peak, levels=[solution.solidus], colors="cyan", linewidths=1.4)
    fig.colorbar(filled, ax=ax, label="Peak T (K)")
    ax.invert_yaxis()
    ax.set_xlabel("x along the weld (mm)")
    ax.set_ylabel("depth (mm)")
    ax.set_title(f"Longitudinal section at y = {solution.y[centre] * 1e3:.1f} mm")
    return fig


def _plot_macro_section(report: WeldReport, material: Material) -> plt.Figure:
    """Transverse macro-section: peak temperature across the weld with the zones marked."""
    profile = report.metrics.profile
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(profile.y * 1e3, profile.T_peak, "k-", lw=1.5, label="peak temperature")
    ax.axhline(material.liquidus, color="red", ls="--", lw=1, label="liquidus")
    ax.axhline(material.solidus, color="orange", ls="--", lw=1, label="solidus")
    ax.axhline(material.haz_outer_temperature, color="green", ls="--", lw=1, label="HAZ limit")
    ax.fill_between(
        profile.y * 1e3,
        material.solidus,
        profile.T_peak,
        where=profile.T_peak >= material.solidus,
        color="red",
        alpha=0.2,
    )
    ax.set_xlabel("y across the weld (mm)")
    ax.set_ylabel("Peak T (K)")
    ax.set_title(f"Macro-section at x = {profile.x * 1e3:.1f} mm")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig


def _plot_phases(phases: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 2.4))
    names = list(phases)
    values = [phases[n] * 100 for n in names]
    ax.barh(names, values, color="steelblue")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Volume fraction (%)")
    ax.set_title("Predicted HAZ constitution")
    for i, v in enumerate(values):
        ax.text(min(v + 1, 92), i, f"{v:.0f}%", va="center", fontsize=8)
    ax.grid(True, axis="x", alpha=0.3)
    return fig


def _plot_energy_density(x: np.ndarray, y: np.ndarray, E: np.ndarray) -> plt.Figure:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots(figsize=(7, 4))
    mesh = ax.pcolormesh(X * 1e3, Y * 1e3, E * 1e-6, shading="auto", cmap="hot")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Absorbed energy density (MJ/m²)")
    fig.colorbar(mesh, ax=ax, label="MJ/m²")
    return fig


def _plot_wobble_concentration(y: np.ndarray, E: np.ndarray) -> plt.Figure:
    """Energy density across the track, which is what sets penetration uniformity."""
    station = int(np.argmax(E.max(axis=1)))
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(y * 1e3, E[station, :] * 1e-6, "r-")
    ax.set_xlabel("y across the weld (mm)")
    ax.set_ylabel("Energy density (MJ/m²)")
    ax.set_title("Heat concentration across the wobble track")
    ax.grid(True, alpha=0.3)
    return fig


def _process_metrics(
    x: np.ndarray,
    y: np.ndarray,
    T: np.ndarray,
    material: Material,
    weld: WeldParams,
) -> dict:
    T_max = float(T.max())
    T_min = float(T.min())
    liquidus = material.liquidus
    solidus = material.solidus
    melted = T >= liquidus
    pool_width = 0.0
    pool_length = 0.0
    if melted.any():
        # Approximate pool dimensions from the melted cells
        Xm, Ym = np.meshgrid(x, y, indexing="ij")
        x_melt = Xm[melted]
        y_melt = Ym[melted]
        if len(x_melt) > 1:
            pool_length = float(np.ptp(x_melt))
            pool_width = float(np.ptp(y_melt))

    metrics = {
        "T_peak (K)": T_max,
        "T_min (K)": T_min,
        "Pool length (mm)": pool_length * 1e3,
        "Pool width (mm)": pool_width * 1e3,
        "Melted cells": int(melted.sum()),
        "Liquidus (K)": liquidus,
        "Solidus (K)": solidus,
        "Line energy (J/mm)": weld.line_energy_j_per_mm,
        "Power density (MW/m²)": weld.power_density_w_per_m2 / 1e6,
        "Spot area (mm²)": weld.spot_area_m2 * 1e6,
    }
    return metrics


def _run_python_script(script: str, *args: str) -> tuple[int, str, str]:
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / script)]
    cmd.extend(args)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    return proc.returncode, proc.stdout, proc.stderr


def _get_material(mat_name: str, op_temp: float, custom_krc: tuple | None = None) -> Material:
    if custom_krc is not None:
        k, rho, cp, t0 = custom_krc
        return Material(
            name="Custom",
            density=rho,
            thermal_conductivity=k,
            specific_heat=cp,
            T0=t0,
            solidus=1700.0,
            liquidus=1800.0,
        )
    return load_material(mat_name, op_temp)


def _render_weld_report(
    report: WeldReport,
    result: dict,
    config: ThermalSimulationConfig,
) -> None:
    """The engineering answer: weld size, penetration, metallurgy and distortion."""
    material = config.material
    assert isinstance(material, Material)
    m, k = report.metrics, report.keyhole
    weld = config.weld
    assert weld is not None
    x, y = result["x"], result["y"]

    for warning in report.warnings:
        st.warning(warning)

    st.subheader("Weld")
    cols = st.columns(4)
    cols[0].metric("Penetration", f"{report.penetration * 1e3:.2f} mm", k.mode + " mode")
    cols[1].metric("Fusion zone width", f"{m.fusion_width_mm:.2f} mm")
    cols[2].metric("HAZ width", f"{m.haz_width_mm:.2f} mm", "per side")
    cols[3].metric("Heat input", f"{report.heat_input:.1f} J/mm", "absorbed")
    cols = st.columns(4)
    cols[0].metric("Peak temperature", f"{m.peak_temperature:.0f} K")
    cols[1].metric("Melt dwell", f"{m.melt_dwell:.2f} s", "above solidus")
    cols[2].metric("t8/5", f"{m.t_8_5:.2f} s" if m.t_8_5 is not None else "n/a")
    cols[3].metric(
        "Cooling rate",
        f"{m.cooling_rate:.0f} K/s" if m.cooling_rate is not None else "n/a",
        "at 800 °C",
    )

    c1, c2 = st.columns(2)
    with c1:
        _show(_plot_zone_map(x, y, result["history"].T_peak, material))
    with c2:
        _show(_plot_macro_section(report, material))

    solution = result.get("solution3d")
    section = report.section
    st.subheader("Penetration and welding mode")
    cols = st.columns(3)
    cols[0].metric("Absorbed intensity", f"{k.intensity_mw_per_cm2:.2f} MW/cm²")
    cols[1].metric(
        "Depth / width",
        f"{section.aspect_ratio:.2f}" if section is not None else f"{k.aspect_ratio:.2f}",
    )
    cols[2].metric("Full penetration", "yes" if report.full_penetration else "no")

    if solution is not None and section is not None:
        cols = st.columns(4)
        cols[0].metric("Fusion area", f"{section.fusion_area * 1e6:.2f} mm²", "cross-section")
        cols[1].metric("Root width", f"{section.root_width * 1e3:.2f} mm")
        cols[2].metric("HAZ depth", f"{section.haz_depth * 1e3:.2f} mm")
        cols[3].metric("Keyhole power share", f"{solution.keyhole_fraction * 100:.0f} %")
        c1, c2 = st.columns(2)
        with c1:
            _show(_plot_cross_section(solution))
        with c2:
            _show(_plot_longitudinal_section(solution))
        st.caption(
            f"Penetration is measured on the fusion boundary of the {solution.T.shape[0]}×"
            f"{solution.T.shape[1]}×{solution.T.shape[2]} 3D solve "
            f"({solution.steps} steps of {solution.dt * 1e3:.2f} ms). Mode is set by "
            "comparing absorbed intensity with the ~1 MW/cm² keyhole threshold, and the "
            "capillary is an assumed tapered channel: melt flow, recoil pressure and the "
            "free surface are not resolved, which is what the optional OpenFOAM export is for."
        )
    else:
        st.caption(
            "Mode is set by comparing absorbed intensity with the ~1 MW/cm² keyhole "
            "threshold; depth comes from an energy balance over the melted channel. "
            "Switch the solver to **3D through-thickness** on the **Setup** tab to "
            "measure penetration and see the weld cross-section."
        )

    micro = report.microstructure
    if micro is not None:
        st.subheader("HAZ microstructure")
        if micro.phases:
            c1, c2 = st.columns([2, 1])
            with c1:
                _show(_plot_phases(micro.phases))
            with c2:
                if micro.hardness_hv is not None:
                    st.metric(
                        "Predicted HAZ hardness",
                        f"{micro.hardness_hv:.0f} HV",
                        f"parent {micro.base_hardness_hv:.0f} HV",
                    )
                if micro.carbon_equivalent is not None:
                    st.metric("Carbon equivalent (IIW)", f"{micro.carbon_equivalent:.2f}")
                st.metric("Coarse-grained band", f"{micro.coarse_grain_width * 1e3:.2f} mm")
                if micro.coarse_grain_dwell is not None:
                    st.metric(
                        "Time above coarsening temperature",
                        f"{micro.coarse_grain_dwell:.2f} s",
                        help=(
                            "Longest time any coarse-grained HAZ point stayed above "
                            "the grain-coarsening temperature — not the melt dwell."
                        ),
                    )
        if micro.bands:
            st.dataframe(
                [
                    {
                        "Zone": band.name,
                        "Peak T (K)": f"{band.t_min:.0f} – {band.t_max:.0f}",
                        "Width per side (mm)": round(band.width_mm, 3),
                        "Area (mm²)": round(band.area * 1e6, 2),
                        "Significance": band.note,
                    }
                    for band in micro.bands
                ],
                width="stretch",
                hide_index=True,
            )
        st.caption(
            f"Model: {micro.model}. Zone limits and transformation data come from the "
            "material YAML — replace them with the CCT data for your plate for "
            "anything beyond parameter screening."
        )

    dist = report.distortion
    if dist is not None:
        st.subheader("Distortion and residual stress")
        cols = st.columns(4)
        cols[0].metric("Angular distortion", f"{dist.angular_distortion:.3f} °")
        cols[1].metric("Transverse shrinkage", f"{dist.transverse_shrinkage_mm:.3f} mm")
        cols[2].metric("Longitudinal shrinkage", f"{dist.longitudinal_shrinkage_mm:.3f} mm")
        cols[3].metric("Bowing", f"{dist.bowing_deflection_mm:.3f} mm")
        cols = st.columns(3)
        cols[0].metric("Shrinkage force", f"{dist.shrinkage_force / 1e3:.1f} kN")
        cols[1].metric("Residual tension at weld", f"{dist.peak_tensile_stress / 1e6:.0f} MPa")
        cols[2].metric(
            "Balancing compression", f"{dist.balancing_compressive_stress / 1e6:.0f} MPa"
        )
        st.caption(
            "Inherent-strain (shrinkage-force) estimate for a single unrestrained "
            f"pass on a {config.Lx * 1e3:.0f} × {config.Ly * 1e3:.0f} × "
            f"{config.thickness * 1e3:.1f} mm plate. Restraint, tacking and multi-pass "
            "sequencing change these numbers substantially."
        )

    wob = report.wobble
    if wob is not None:
        st.subheader("Wobble heat concentration")
        cols = st.columns(4)
        cols[0].metric("Swept width", f"{wob.swept_width_mm:.2f} mm")
        cols[1].metric(
            "Advance per loop",
            f"{wob.pitch_mm:.3f} mm" if math.isfinite(wob.pitch) else "no wobble",
        )
        cols[2].metric("Footprint overlap", f"{wob.overlap_ratio * 100:.0f} %")
        cols[3].metric("Track uniformity", f"{wob.uniformity * 100:.0f} %")
        cols = st.columns(3)
        cols[0].metric("Peak energy density", f"{wob.peak_energy_density * 1e-6:.1f} MJ/m²")
        cols[1].metric("Peak / track mean", f"{wob.concentration_ratio:.2f}×")
        cols[2].metric("Peak vs straight beam", f"−{wob.peak_reduction * 100:.0f} %")
        assert config.path is not None
        energy = energy_density_map(
            config.path,
            config.wobble or WobbleParams(amplitude=0.0, frequency=0.0),
            weld,
            config.T1,
            x,
            y,
            t_end=min(config.t_end, config.path.duration),
        )
        c1, c2 = st.columns(2)
        with c1:
            _show(_plot_energy_density(x, y, energy))
        with c2:
            _show(_plot_wobble_concentration(y, energy))
        cols = st.columns(2)
        cols[0].metric("Spot speed (mean)", f"{wob.mean_beam_speed:.2f} m/s")
        cols[1].metric("Spot speed (peak)", f"{wob.peak_beam_speed:.2f} m/s")

    st.download_button(
        label="Download weld assessment (JSON)",
        data=json.dumps(report.as_dict(), indent=2).encode("utf-8"),
        file_name="weld_report.json",
        mime="application/json",
    )


def _page_thermal_and_wobble():
    st.header("Weld simulation")
    st.caption(
        "Enter the process, material and wobble parameters; the app solves the weld "
        "locally and reports fusion zone, penetration, HAZ metallurgy, distortion and "
        "heat concentration. No internet or external solver needed."
    )

    tab_setup, tab_wobble, tab_weld, tab_thermal = st.tabs(
        ["Setup", "Wobble signature", "Weld result", "Thermal field"]
    )

    with tab_setup:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Material")
            mat_options = list_materials()
            mat_name = st.selectbox("Material", mat_options)
            op_temp = st.slider(
                "Operating temperature for props (K)",
                293.0,
                3000.0,
                800.0,
                50.0,
            )
            use_custom = st.checkbox("Use custom k/rho/cp")
            if use_custom:
                k_c = st.number_input("k (W/m·K)", 1.0, 500.0, 50.0)
                rho_c = st.number_input("rho (kg/m³)", 500.0, 30000.0, 7850.0)
                cp_c = st.number_input("cp (J/kg·K)", 100.0, 5000.0, 500.0)
                t0_c = st.number_input("T0 (K)", 200.0, 500.0, 300.0)
                material = Material(
                    name="Custom",
                    density=rho_c,
                    thermal_conductivity=k_c,
                    specific_heat=cp_c,
                    T0=t0_c,
                    solidus=1700.0,
                    liquidus=1800.0,
                )
            else:
                material = _get_material(mat_name, op_temp)
            st.write(
                f"**{material.name}**: k={material.thermal_conductivity:.1f}, "
                f"rho={material.density:.0f}, cp={material.specific_heat:.0f}"
            )

        with c2:
            st.subheader("Laser / process")
            power = st.slider("Power (W)", 100.0, 8000.0, 1500.0, 50.0)
            efficiency = st.slider("Efficiency", 0.1, 1.0, 0.8, 0.05)
            speed = st.slider("Travel speed (m/s)", 0.001, 0.05, 0.01, 0.001)
            beam_radius = st.slider("Beam 1/e² radius (µm)", 50.0, 1000.0, 500.0, 25.0)
            sigma = beam_radius * 1e-6

        st.subheader("Weld path")
        c3, c4 = st.columns(2)
        with c3:
            x0 = st.number_input("Start x (m)", 0.0, 0.5, 0.01, 0.001)
            y0 = st.number_input("Start y (m)", 0.0, 0.2, 0.025, 0.001)
        with c4:
            x1 = st.number_input("End x (m)", 0.0, 0.5, 0.07, 0.001)
            y1 = st.number_input("End y (m)", 0.0, 0.2, 0.025, 0.001)
        length = math.hypot(x1 - x0, y1 - y0)
        if length <= 0.0:
            st.error("The weld path has zero length. Move the end point away from the start.")
            return
        st.write(f"Path length: **{length*1e3:.2f} mm**, travel time: **{length/speed:.3f} s**")

        st.subheader("Wobble")
        c5, c6, c7 = st.columns(3)
        with c5:
            wobble_amp = st.slider("Wobble amplitude (µm)", 0.0, 1000.0, 400.0, 25.0)
        with c6:
            # Low default so individual loops are resolvable in the path preview;
            # production wobble is typically several hundred Hz.
            wobble_freq = st.slider("Wobble frequency (Hz)", 0, 2000, 20, 10)
        with c7:
            pattern = st.selectbox(
                "Pattern",
                ["circle", "line", "figure8", "infinity"],
                format_func=lambda p: {
                    "circle": "Circle",
                    "line": "Line / Sine",
                    "figure8": "Figure-8",
                    "infinity": "Infinity / Lemniscate",
                }[p],
            )

        st.subheader("Plate & mesh")
        c8, c9 = st.columns(2)
        with c8:
            Lx = st.number_input("Plate length Lx (m)", 0.01, 0.5, 0.08, 0.01)
            Ly = st.number_input("Plate width Ly (m)", 0.01, 0.2, 0.05, 0.001)
            plate_thickness = st.number_input(
                "Plate thickness (mm)", 0.1, 50.0, 3.0, 0.1, key="plate_thickness_mm"
            )
            T1 = st.number_input("Heat-spreading depth h (m)", 0.0005, 0.05, 0.003, 0.0005)
            st.caption(
                "The thermal model spreads the beam flux over depth h. Set it to the "
                "expected penetration for a partial-penetration pass, or to the plate "
                "thickness for a fully penetrating one."
            )
        with c9:
            solver = st.radio(
                "Solver",
                ["3d", "2d"],
                format_func=lambda s: {
                    "3d": "3D through-thickness (measures penetration)",
                    "2d": "2D thin plate (fast screening)",
                }[s],
                help=(
                    "The 3D solve resolves the depth, so penetration and the weld "
                    "cross-section are measured rather than estimated. It runs "
                    "inside the app; no external solver is needed."
                ),
            )
            nx = st.slider("Grid points X", 21, 201, 81, 2)
            ny = st.slider("Grid points Y", 11, 101, 41, 2)
            nz = st.slider("Grid points through thickness", 5, 61, 13, 2, disabled=solver == "2d")
            # Long enough for the measured section to cool through 500 C, so t8/5
            # is available. The 3D solve pays for every extra second of simulated
            # time, so it gets the shorter tail.
            tail = 1.3 if solver == "3d" else 2.0
            t_end = st.number_input(
                "Simulation time (s)",
                0.1,
                50.0,
                float(min(max(tail * length / speed, 0.1), 50.0)),
                0.1,
                key=f"t_end_{solver}",
            )
            dt = st.number_input("Time step (s)", 0.001, 0.5, 0.01, 0.001, disabled=solver == "3d")
            if solver == "3d":
                seconds = _estimate_3d_runtime(
                    nx, ny, nz, Lx, Ly, plate_thickness / 1e3, t_end, material
                )
                st.caption(
                    "The 3D solve picks its own stable time step. Estimated run time "
                    f"**{seconds:.0f} s** — shorten the weld, the simulated time or the "
                    "mesh if that is too slow."
                )

        st.subheader("Thermal probe")
        st.caption("Defaults to the mid-point of the weld path, where the torch passes over it.")
        c10, c11 = st.columns(2)
        with c10:
            px = st.number_input("Probe x (m)", 0.0, Lx, min((x0 + x1) / 2, Lx), 0.001)
        with c11:
            py = st.number_input("Probe y (m)", 0.0, Ly, min((y0 + y1) / 2, Ly), 0.001)

        weld = WeldParams(
            power=power,
            efficiency=efficiency,
            speed=speed,
            start_pos=(x0, y0),
            direction="x",
            sigma=sigma,
        )
        path = WeldPath(start=(x0, y0), end=(x1, y1), speed=speed)
        wobble = WobbleParams(
            amplitude=wobble_amp * 1e-6,
            frequency=wobble_freq,
            pattern=pattern,
        )

        st.divider()
        go, run = st.columns([1, 2])
        with go:
            preview_pressed = st.button("Go (draw heat signature)", type="primary", width="stretch")
        with run:
            run_label = "Run 3D weld simulation" if solver == "3d" else "Run 2D weld simulation"
            run_pressed = st.button(run_label, width="stretch")

        st.session_state["weld"] = weld
        st.session_state["path"] = path
        st.session_state["wobble"] = wobble
        st.session_state["material"] = material
        st.session_state["Lx"] = Lx
        st.session_state["Ly"] = Ly
        st.session_state["nx"] = nx
        st.session_state["ny"] = ny
        st.session_state["t_end"] = t_end
        st.session_state["dt"] = dt
        st.session_state["T1"] = T1

        if preview_pressed or run_pressed:
            with st.spinner("Computing beam path & heat signature..."):
                x = np.linspace(0, Lx, nx)
                y = np.linspace(0, Ly, ny)
                t_max = min(t_end, path.duration * 1.2)
                # The sampling has to follow the wobble frequency, or a fast
                # oscillation is sampled at one phase and draws as a straight line.
                t_from, t_to, heat_dt = sampling_window(t_max, wobble, min(0.002, dt))
                Q = heat_signature(
                    path=path,
                    wobble=wobble,
                    power=power,
                    efficiency=efficiency,
                    sigma=sigma,
                    h=T1,
                    x=x,
                    y=y,
                    t_end=t_to,
                    dt=heat_dt,
                    t_start=t_from,
                )
                x_traj, y_traj = beam_trajectory(
                    path=path,
                    wobble=wobble,
                    t_end=t_to,
                    dt=min(heat_dt / 2, 0.0005),
                    t_start=t_from,
                )
            st.session_state["Q"] = Q
            st.session_state["x_traj"] = x_traj
            st.session_state["y_traj"] = y_traj
            st.session_state["x"] = x
            st.session_state["y"] = y

        if run_pressed:
            config = ThermalSimulationConfig(
                nx=nx,
                ny=ny,
                nz=nz,
                Lx=Lx,
                Ly=Ly,
                t_end=t_end,
                dt=dt,
                weld=weld,
                material=material,
                output_file=None,
                T1=T1,
                plate_thickness=plate_thickness / 1e3,
                path=path,
                wobble=wobble,
                probe=(px, py) if solver == "2d" else None,
                solver=solver,
            )
            bar = st.progress(0.0, text="Running thermal simulation...")

            def report_progress(fraction: float) -> None:
                bar.progress(
                    min(fraction, 1.0),
                    text=f"Running 3D thermal simulation... {fraction * 100:.0f}%",
                )

            try:
                result = run_thermal_simulation(config, on_progress=report_progress)
                report = build_report(config, result)
            except WeldSimError as exc:
                st.error(str(exc))
                st.session_state.pop("thermal_result", None)
                st.session_state.pop("weld_report", None)
                result = None
            finally:
                bar.empty()
            if result is not None:
                st.session_state["thermal_result"] = result
                st.session_state["weld_report"] = report
                st.session_state["thermal_config"] = config
                st.session_state["thermal_material"] = material
                st.session_state["thermal_weld"] = weld
                st.session_state["thermal_x1"] = x1
                st.session_state["thermal_y1"] = y1

    # --- Wobble tab (always rendered, data from session state) ---
    with tab_wobble:
        if "Q" in st.session_state:
            Q = st.session_state["Q"]
            x_traj = st.session_state["x_traj"]
            y_traj = st.session_state["y_traj"]
            x = st.session_state["x"]
            y = st.session_state["y"]
            Lx = st.session_state["Lx"]
            Ly = st.session_state["Ly"]
            _show(_plot_beam_path(x_traj, y_traj, Lx, Ly))
            _show(_plot_heat_signature(x, y, Q, x_traj, y_traj))

            with st.expander("Run animation", expanded=True):
                anim_col, anim_params = st.columns([1, 3])
                with anim_col:
                    run_anim = st.button("Run animation", type="secondary", width="stretch")
                with anim_params:
                    fps = st.slider("FPS", 1, 30, 10, 1)
                    anim_dt = st.slider("Anim time step (s)", 0.02, 0.2, 0.1, 0.01)
                    trail = st.slider("Trail length (s)", 0.01, 0.2, 0.05, 0.01)
                if run_anim or "wobble_gif" in st.session_state:
                    if run_anim:
                        with st.spinner("Rendering animation..."):
                            # Fixed coarse grid for fast, accurate wobble rendering
                            x_anim = np.linspace(0, Lx, 41)
                            y_anim = np.linspace(0, Ly, 21)
                            t_end_anim = min(st.session_state.get("t_end", 6.0), path.duration)
                            gif = wobble_animation_gif(
                                path=path,
                                wobble=wobble,
                                power=st.session_state["weld"].power,
                                efficiency=st.session_state["weld"].efficiency,
                                sigma=st.session_state["weld"].sigma,
                                h=st.session_state["T1"],
                                x=x_anim,
                                y=y_anim,
                                t_end=t_end_anim,
                                frame_dt=anim_dt,
                                trail_time=trail,
                                heat_dt=sampling_window(trail, wobble, 0.002)[2],
                                fps=fps,
                                gif_width=400,
                            )
                            st.session_state["wobble_gif"] = gif
                    st.image(
                        st.session_state["wobble_gif"],
                        caption="Wobbled weld animation — brighter = more dwell time",
                    )
        else:
            st.info(
                "Click **Go (draw heat signature)** on the **Setup** tab to generate "
                "the wobble preview."
            )

    # --- Weld result tab: what the parameters actually produce ---
    with tab_weld:
        if "weld_report" in st.session_state:
            _render_weld_report(
                st.session_state["weld_report"],
                st.session_state["thermal_result"],
                st.session_state["thermal_config"],
            )
        else:
            st.info("Run a simulation on the **Setup** tab to get the weld assessment.")

    # --- Thermal tab (always rendered, data from session state) ---
    with tab_thermal:
        if "thermal_result" in st.session_state:
            result = st.session_state["thermal_result"]
            material = st.session_state["thermal_material"]
            weld = st.session_state["thermal_weld"]
            x1 = st.session_state["thermal_x1"]
            y1 = st.session_state["thermal_y1"]
            x, y, T = result["x"], result["y"], result["T"]

            metrics = _process_metrics(x, y, T, material, weld)
            st.subheader("Process metrics")
            cols = st.columns(4)
            for i, (k, v) in enumerate(metrics.items()):
                with cols[i % 4]:
                    st.metric(k, f"{v:.3f}" if isinstance(v, float) else v)

            c1, c2 = st.columns(2)
            with c1:
                _show(_plot_temperature_2d(x, y, T))
            with c2:
                _show(_plot_temperature_3d(x, y, T))

            with st.expander("Temperature profiles", expanded=True):
                prof1, prof2, prof3 = st.tabs(["Longitudinal", "Transverse", "Probe history"])
                with prof1:
                    j_mid = int(round(y1 / (y[1] - y[0])))
                    j_mid = min(max(j_mid, 0), len(y) - 1)
                    _show(
                        _plot_temperature_profile(
                            x, T[:, j_mid], "x (mm)", f"T along y = {y[j_mid]*1e3:.1f} mm", material
                        )
                    )
                with prof2:
                    i_end = int(round(x1 / (x[1] - x[0])))
                    i_end = min(max(i_end, 0), len(x) - 1)
                    _show(
                        _plot_temperature_profile(
                            y,
                            T[i_end, :],
                            "y (mm)",
                            f"T across x = {x[i_end]*1e3:.1f} mm",
                            material,
                        )
                    )
                with prof3:
                    if "t" in result and "T_probe" in result:
                        _show(_plot_probe_history(result["t"], result["T_probe"], material))
                    else:
                        st.info("Probe history not available.")

            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["x_m", "y_m", "T_K"])
            for i in range(len(x)):
                for j in range(len(y)):
                    writer.writerow([f"{x[i]:.6e}", f"{y[j]:.6e}", f"{T[i, j]:.3f}"])
            csv_bytes = buf.getvalue().encode("utf-8")
            st.download_button(
                label="Download temperature CSV",
                data=csv_bytes,
                file_name="temperature.csv",
                mime="text/csv",
            )
        else:
            st.info("Run a simulation on the **Setup** tab to see results.")


COUPON_COLUMNS = [
    "Label",
    "Power (W)",
    "Speed (mm/s)",
    "Thickness (mm)",
    "Beam radius (mm)",
    "Measured penetration (mm)",
    "Measured fusion width (mm)",
]

EXAMPLE_COUPONS = [
    ["P2000-v20", 2000.0, 20.0, 3.0, 0.2, 1.5, 1.2],
    ["P3000-v20", 3000.0, 20.0, 3.0, 0.2, 2.2, 1.4],
    ["P2000-v40", 2000.0, 40.0, 3.0, 0.2, 1.0, 0.9],
]


def _coupons_from_rows(rows) -> list[Coupon]:
    """Build coupons from the measurement editor, skipping blank rows."""
    coupons: list[Coupon] = []
    for i, row in enumerate(rows):
        power = row.get("Power (W)")
        penetration = row.get("Measured penetration (mm)")
        if not power or penetration is None:
            continue
        width = row.get("Measured fusion width (mm)")
        coupons.append(
            Coupon(
                label=str(row.get("Label") or f"coupon {i + 1}"),
                power=float(power),
                speed=float(row.get("Speed (mm/s)") or 0.0) / 1e3,
                thickness=float(row.get("Thickness (mm)") or 0.0) / 1e3,
                penetration=float(penetration) / 1e3,
                fusion_width=float(width) / 1e3 if width else None,
                sigma=float(row.get("Beam radius (mm)") or 0.0) / 1e3,
            )
        )
    return coupons


def _plot_measured_vs_predicted(comparison: Comparison) -> plt.Figure:
    """Parity plot: points on the diagonal mean the model matches the macros."""
    measured = [r.measured_penetration * 1e3 for r in comparison.residuals]
    predicted = [r.predicted_penetration * 1e3 for r in comparison.residuals]
    fig, ax = plt.subplots(figsize=(5, 5))
    limit = max(measured + predicted + [0.1]) * 1.15
    ax.plot([0, limit], [0, limit], "k--", lw=1, label="perfect agreement")
    ax.plot([0, limit], [0, 1.2 * limit], color="0.7", lw=0.8)
    ax.plot([0, limit], [0, 0.8 * limit], color="0.7", lw=0.8, label="\u00b120%")
    ax.scatter(measured, predicted, s=60, color="tab:red", zorder=3)
    for r, m, p in zip(comparison.residuals, measured, predicted):
        ax.annotate(r.label, (m, p), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Measured penetration (mm)")
    ax.set_ylabel("Predicted penetration (mm)")
    ax.set_title("Macro-section parity")
    ax.legend(loc="lower right", fontsize=8)
    return fig


def _residual_rows(comparison: Comparison) -> list[dict]:
    rows = []
    for r in comparison.residuals:
        error = r.penetration_error_percent
        measured_width = r.measured_fusion_width
        rows.append(
            {
                "Coupon": r.label,
                "Measured depth (mm)": round(r.measured_penetration * 1e3, 2),
                "Predicted depth (mm)": round(r.predicted_penetration * 1e3, 2),
                "Depth error (%)": None if error is None else round(error, 1),
                "Measured width (mm)": (
                    None if measured_width is None else round(measured_width * 1e3, 2)
                ),
                "Predicted width (mm)": round(r.predicted_fusion_width * 1e3, 2),
                "Full penetration": r.full_penetration,
            }
        )
    return rows


def _calibration_controls(coupons: list[Coupon], material: Material, mesh: Mesh) -> None:
    """Compare / fit buttons, and store whatever they produce in the session."""
    left, right = st.columns(2)
    with left:
        efficiency = st.slider(
            "Absorption efficiency to compare at", 0.1, 1.0, 0.8, 0.05, key="cal_eff"
        )
        compare_now = st.button("Compare with current settings", width="stretch")
    with right:
        st.caption(
            f"Fitting solves every coupon at every trial setting: "
            f"{len(coupons)} coupons x {len(EFFICIENCY_CANDIDATES) * len(TAPER_CANDIDATES)} "
            "trials, so expect it to take a while."
        )
        fit_now = st.button("Fit absorption and keyhole taper", type="primary", width="stretch")

    if not (compare_now or fit_now):
        return

    progress = st.progress(0.0, text="Solving coupons...")
    try:
        if fit_now:
            calibration = calibrate(
                coupons,
                material,
                mesh=mesh,
                baseline_efficiency=efficiency,
                on_progress=lambda f: progress.progress(f, text="Searching absorption/taper..."),
            )
            st.session_state["calibration"] = calibration
            st.session_state["comparison"] = calibration.comparison
        else:
            st.session_state["comparison"] = compare(
                coupons,
                material,
                efficiency,
                mesh=mesh,
                on_progress=lambda f: progress.progress(f, text="Solving coupons..."),
            )
            st.session_state.pop("calibration", None)
    except WeldSimError as exc:
        st.error(str(exc))
    finally:
        progress.empty()


def _show_calibration(calibration: Calibration) -> None:
    m1, m2, m3 = st.columns(3)
    m1.metric("Fitted absorption efficiency", f"{calibration.efficiency:.2f}")
    m2.metric(
        "Fitted keyhole taper",
        f"{calibration.keyhole_taper:.2f}",
        help="Capillary radius at its tip as a fraction of its radius at the surface.",
    )
    m3.metric(
        "Relative error after fitting",
        f"{calibration.cost * 100:.1f}%",
        delta=f"{-calibration.improvement * 100:.0f}% vs uncalibrated",
        delta_color="inverse",
    )
    st.download_button(
        "Download calibration (YAML)",
        data=calibration_yaml(calibration).encode("utf-8"),
        file_name=f"calibration_{calibration.material}.yaml",
        mime="text/yaml",
    )
    st.caption(
        "Enter the fitted efficiency on the **Weld simulation** page to run calibrated "
        "predictions. The fit holds only for the material, thickness and process window "
        "the coupons covered."
    )
    if calibration.efficiency >= max(EFFICIENCY_CANDIDATES):
        st.warning(
            "The fit ran into the top of the absorption range, so the model still cannot "
            "reach the measured depths: suspect a tighter focus, a deeper capillary, or "
            "macros from a different beam setup."
        )


def _page_calibration():
    st.header("Measured vs predicted")
    st.info(
        "Weld a bracket of coupons at a fixed focus, varying power and travel speed, then "
        "cut, polish and etch them and measure the fusion boundary. Entering those "
        "measurements here shows how far the model is off, and fits the two things it "
        "cannot know about your machine: how much beam power is absorbed, and how the "
        "capillary narrows with depth."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        material_name = st.selectbox("Material", list_materials(), key="cal_material")
    with c2:
        nz = st.slider("Grid points through thickness", 7, 21, 11, 2, key="cal_nz")
    with c3:
        weld_length = st.number_input(
            "Simulated weld length (mm)",
            min_value=5.0,
            max_value=60.0,
            value=20.0,
            step=5.0,
            help="Long enough for the sectioned position to be quasi-steady.",
        )

    st.subheader("Coupon measurements")
    edited = st.data_editor(
        [dict(zip(COUPON_COLUMNS, row)) for row in EXAMPLE_COUPONS],
        num_rows="dynamic",
        width="stretch",
        key="coupon_editor",
    )

    try:
        material = load_material(material_name)
        coupons = _coupons_from_rows(edited)
        for coupon in coupons:
            coupon.validate()
    except (WeldSimError, FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    if not coupons:
        st.warning("Enter at least one coupon with a power and a measured penetration.")
        return

    _calibration_controls(coupons, material, Mesh(nz=nz, weld_length=weld_length / 1e3))

    calibration = st.session_state.get("calibration")
    if calibration is not None:
        _show_calibration(calibration)

    comparison = st.session_state.get("comparison")
    if comparison is None:
        st.info("Enter your measurements, then compare or fit.")
        return

    st.subheader("Residuals")
    st.dataframe(_residual_rows(comparison), width="stretch")
    a, b = st.columns(2)
    with a:
        _show(_plot_measured_vs_predicted(comparison))
    with b:
        st.metric("Penetration RMS error", f"{comparison.penetration_rms * 1e3:.2f} mm")
        st.metric(
            "Penetration bias",
            f"{comparison.penetration_bias * 1e3:+.2f} mm",
            help="Positive means the model predicts deeper welds than you measured.",
        )
        width_rms = comparison.width_rms
        if width_rms is not None:
            st.metric("Fusion width RMS error", f"{width_rms * 1e3:.2f} mm")
        st.caption(
            "Calibration removes systematic bias. Scatter that survives the fit is either "
            "physics this model does not carry — melt flow, vapour loss, focus drift — or "
            "spread in the sectioning itself."
        )


def _page_keyhole_cfd():
    st.header("Optional: high-fidelity OpenFOAM export")
    st.info(
        "Nothing on this page is needed for a weld assessment — the **Weld simulation** "
        "page solves the weld through the thickness on its own. This page exports the "
        "same job as an OpenFOAM case for anyone who wants resolved free-surface CFD: "
        "vapour recoil, melt flow and a true keyhole shape, which the built-in solver "
        "approximates with a fixed capillary. OpenFOAM is a separate native package "
        "(Linux or WSL2) that has to be installed and compiled outside this app."
    )

    root = Path(__file__).resolve().parent.parent
    keyhole_root = root / "keyhole-cfd"

    if not keyhole_root.exists():
        st.error("`keyhole-cfd/` directory not found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("OpenFOAM case")
        if st.button("Generate workpiece STL", width="stretch"):
            with st.spinner("Generating STL..."):
                rc, out, err = _run_python_script("keyhole-cfd/scripts/generate_workpiece_stl.py")
            if rc == 0:
                st.success("STL generated")
                st.code(out, language="text")
            else:
                st.error(f"Failed:\n{err}")
    with c2:
        st.subheader("Case config")
        if st.button("Configure OpenFOAM case", width="stretch"):
            with st.spinner("Configuring case..."):
                rc, out, err = _run_python_script("keyhole-cfd/scripts/configure_case.py")
            if rc == 0:
                st.success("Case configured")
                st.code(out, language="text")
            else:
                st.error(f"Failed:\n{err}")

    stl_path = keyhole_root / "openfoam" / "triSurface" / "workpiece.stl"
    if stl_path.exists():
        st.subheader("3D workpiece preview")
        try:
            import pyvista as pv

            mesh = pv.read(stl_path)
            st.write(f"Mesh: {mesh.n_points} points, {mesh.n_cells} cells")

            col_a, col_b = st.columns([1, 3])
            with col_a:
                color = st.color_picker("Part color", "#C0C0C0")
                show_edges = st.checkbox("Show edges", True)
            with col_b:
                plotter = pv.Plotter(off_screen=True, window_size=(800, 450))
                plotter.add_mesh(mesh, color=color, show_edges=show_edges)
                plotter.add_axes()
                png = plotter.screenshot()
                st.image(png, width="stretch")
        except Exception as e:
            st.warning(f"Could not render STL: {e}")
    else:
        st.info("Click **Generate workpiece STL** to create the 3D workpiece.")

    st.subheader("Build & run the case (OpenFOAM install required)")
    st.code(
        """cd ~/WeldSYM/keyhole-cfd
python3 scripts/configure_case.py
python3 scripts/generate_workpiece_stl.py
blockMesh
laserKeyholeVoF
# or parallel:
# decomposePar; mpirun -np 4 laserKeyholeVoF -parallel; reconstructPar
""",
        language="bash",
    )

    with st.expander("WSL runner (experimental)", expanded=False):
        st.write("Run the OpenFOAM mesh and solver directly from WSL if a distro is installed.")

        # Detect WSL distros
        if "wsl_distros" not in st.session_state:
            try:
                proc = subprocess.run(["wsl", "-l", "-v"], capture_output=True, text=True)
                st.session_state["wsl_distros"] = proc.stdout
            except Exception as e:
                st.session_state["wsl_distros"] = f"Error: {e}"
        st.code(st.session_state["wsl_distros"], language="text")

        # Convert Windows path to WSL /mnt path
        posix = keyhole_root.as_posix()
        if len(posix) >= 2 and posix[1] == ":":
            drive = posix[0].lower()
            wsl_path = f"/mnt/{drive}{posix[2:]}"
        else:
            wsl_path = str(keyhole_root)
        st.write(f"**WSL path:** `{wsl_path}`")

        openfoam_bashrc = st.text_input("OpenFOAM bashrc path in WSL", "/opt/openfoam11/etc/bashrc")

        def _wsl_cmd(cmd: str) -> list[str]:
            return [
                "wsl",
                "-e",
                "bash",
                "-c",
                f"source {openfoam_bashrc} 2>/dev/null || true; cd {wsl_path}/openfoam && {cmd}",
            ]

        def _run_in_wsl(cmd: str, timeout: int) -> None:
            """Run an OpenFOAM command through WSL, reporting failures in the page."""
            with st.spinner(f"Running {cmd} in WSL..."):
                try:
                    proc = subprocess.run(
                        _wsl_cmd(cmd), capture_output=True, text=True, timeout=timeout
                    )
                except FileNotFoundError:
                    st.error(
                        "`wsl` was not found on this machine. The 3D keyhole solver "
                        "needs Windows with WSL2 (or a native Linux OpenFOAM install); "
                        "run the commands above manually there."
                    )
                    return
                except subprocess.TimeoutExpired:
                    st.warning(
                        f"{cmd} did not finish within {timeout}s and was stopped. "
                        "Run it directly in WSL for a long solve."
                    )
                    return
                except OSError as exc:
                    st.error(f"Could not start {cmd} through WSL: {exc}")
                    return
            if proc.returncode == 0:
                st.success(f"{cmd} completed")
            else:
                st.error(f"{cmd} failed (exit {proc.returncode}): {proc.stderr}")
            st.code(proc.stdout or proc.stderr, language="text")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Run blockMesh in WSL", width="stretch"):
                _run_in_wsl("blockMesh", timeout=120)
        with col_b:
            if st.button("Run laserKeyholeVoF in WSL", width="stretch"):
                _run_in_wsl("laserKeyholeVoF", timeout=30)


def _page_docs():
    st.header("Documentation")
    st.markdown("""
        **Weld simulation — runs entirely inside this app, offline**
        - Two solvers, both pure Python/NumPy: a fast 2D thin-plate solve for screening,
          and a 3D through-thickness solve that resolves the depth
        - Moving Gaussian surface source plus a tapered volumetric keyhole source,
          latent heat of fusion, surface convection/radiation and an evaporation cap
        - Material library (Ti-6Al-4V, S355) with temperature-dependent properties
        - Wobble patterns: circle, line/sine, figure-8, infinity, with the beam position
          sub-sampled so high frequencies are not aliased
        - Outputs: fusion zone and HAZ geometry, penetration, welding mode, t8/5 and
          cooling rate, HAZ phase fractions and hardness, distortion and residual stress,
          and the wobble heat-concentration map
        - In 3D mode penetration, root width, fusion area and HAZ depth are measured on
          the computed fusion boundary; in 2D mode penetration is an energy-balance estimate
        - Run `python run_gui.py` from the repo root, or `weldsim --solver 3d` on the CLI

        **What the built-in solver does not do**
        - No free-surface motion, vapour recoil, melt flow or Marangoni convection: the
          keyhole is an assumed tapered capillary, not a solved cavity
        - Distortion is an inherent-strain estimate, not thermo-mechanical FEA
        - Treat every number as parameter screening, not weld-procedure qualification

        **Measured vs predicted**
        - Enter the penetration and fusion width measured on etched macro-sections and see
          per-coupon residuals, a parity plot, RMS error and systematic bias
        - Fits the two machine-specific unknowns — absorption efficiency and keyhole taper
          — to your coupons, and exports the fit as YAML
        - Empirical: valid over the material, thickness and process window you measured,
          and not a substitute for procedure qualification

        **Optional: OpenFOAM export (advanced)**
        - Exports the same job as an OpenFOAM VOF case (`laserKeyholeVoF`) with
          enthalpy-porosity melting, recoil pressure and laser ray tracing
        - OpenFOAM is a separate native package: install and compile it on Linux or WSL2
        - Not required for any of the results above
        """)


def main():
    st.set_page_config(
        page_title="Weld Sim",
        page_icon="",
        layout="wide",
    )

    st.sidebar.title("Weld Sim")
    st.sidebar.caption("Laser welding simulation dashboard")

    page = st.sidebar.radio(
        "Navigation",
        ["Weld simulation", "Measured vs predicted", "OpenFOAM export (optional)", "Docs"],
    )

    if page == "Weld simulation":
        _page_thermal_and_wobble()
    elif page == "Measured vs predicted":
        _page_calibration()
    elif page == "OpenFOAM export (optional)":
        _page_keyhole_cfd()
    elif page == "Docs":
        _page_docs()


if __name__ == "__main__":
    main()
