"""Streamlit GUI for Weld Sim — weld assessment, wobble, and OpenFOAM export."""

from __future__ import annotations

import csv
import io
import json
import math
import queue
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  registers the '3d' projection

import plots

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
from dataclasses import replace

from weldsim.errors import AbortError, WeldSimError
from weldsim.materials import (
    Material,
    list_materials,
    load_material,
    material_liquidus,
    material_temperatures,
)
from weldsim.presets import PRESETS, Preset
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
)


def _show(fig: plt.Figure | go.Figure) -> None:
    """Render a Matplotlib or Plotly figure and close it if it is a pyplot."""
    if isinstance(fig, go.Figure):
        _apply_2d_plot_settings(fig)
        plot_width = st.session_state.get("plot_width", 0) or None
        plot_height = st.session_state.get("plot_height", 0) or None
        kwargs: dict[str, Any] = {"theme": None}
        if plot_width is not None:
            kwargs["width"] = int(plot_width)
        if plot_height is not None:
            kwargs["height"] = int(plot_height)
        st.plotly_chart(fig, **kwargs)
    else:
        st.pyplot(fig)
        plt.close(fig)


def _apply_2d_plot_settings(fig: go.Figure) -> None:
    """Apply user-chosen colorscale, contrast and (for heat/surface maps) T clips."""
    colorscale = st.session_state.get("plot_colorscale")
    tmin = st.session_state.get("plot_tmin", 0.0) or None
    tmax = st.session_state.get("plot_tmax", 0.0) or None
    if not (colorscale or tmin or tmax):
        return
    for trace in fig.data:
        if trace.type in ("heatmap", "contour"):
            if colorscale:
                trace.colorscale = colorscale
            if tmin is not None:
                trace.zmin = float(tmin)
            if tmax is not None:
                trace.zmax = float(tmax)
        elif trace.type == "surface":
            if colorscale:
                trace.colorscale = colorscale
            if tmin is not None:
                trace.cmin = float(tmin)
            if tmax is not None:
                trace.cmax = float(tmax)


_RUN_LOG_PATH = Path(__file__).resolve().parent.parent / "completed_runs.json"


def _load_run_log() -> list[dict[str, Any]]:
    if not _RUN_LOG_PATH.exists():
        return []
    with _RUN_LOG_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def _append_run_log(entry: dict[str, Any]) -> None:
    log = _load_run_log()
    log.append(entry)
    _RUN_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _RUN_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)


def _solver_worker(
    config: ThermalSimulationConfig,
    msg_queue: queue.Queue,
    abort_event: threading.Event,
) -> None:
    def on_progress(fraction: float) -> None:
        msg_queue.put(("progress", fraction))

    try:
        result = run_thermal_simulation(
            config,
            on_progress=on_progress,
            abort=abort_event.is_set,
        )
        msg_queue.put(("result", result))
    except AbortError as exc:
        msg_queue.put(("aborted", str(exc)))
    except WeldSimError as exc:
        msg_queue.put(("error", str(exc)))
    except Exception as exc:
        msg_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _simulation_running() -> bool:
    thread = st.session_state.get("run_thread")
    return thread is not None and thread.is_alive()


def _process_run_queue() -> None:
    q = st.session_state.get("run_queue")
    if q is None:
        return
    prev_status = st.session_state.get("run_status", "running")
    progress = st.session_state.get("run_progress", 0.0)
    result = None
    status = prev_status
    message: str | None = st.session_state.get("run_message")
    while not q.empty():
        kind, payload = q.get_nowait()
        if kind == "progress":
            progress = float(payload)
        elif kind == "result":
            result = payload
            status = "completed"
        elif kind in ("aborted", "error"):
            status = kind
            message = str(payload)
    st.session_state["run_progress"] = progress
    if prev_status != "running" or status == "running":
        # Already finalised or still running; only update the progress bar.
        return
    # First time we see the final outcome.
    st.session_state["run_status"] = status
    st.session_state["run_message"] = message
    st.session_state.pop("run_thread", None)
    if status == "completed" and result is not None:
        config = st.session_state.get("run_config")
        if config is not None:
            report = build_report(config, result)
            st.session_state["thermal_result"] = result
            st.session_state["weld_report"] = report
            st.session_state["thermal_config"] = config
            st.session_state["thermal_material"] = config.material
            st.session_state["thermal_weld"] = config.weld
            if config.path is not None:
                st.session_state["thermal_x1"] = config.path.end[0]
                st.session_state["thermal_y1"] = config.path.end[1]
            _append_run_log(
                {
                    "time": datetime.now().isoformat(),
                    "status": "completed",
                    "solver": config.solver,
                    "power": config.weld.power if config.weld else None,
                    "speed": config.weld.speed if config.weld else None,
                    "efficiency": config.weld.efficiency if config.weld else None,
                    "material": (
                        config.material.name if isinstance(config.material, Material) else None
                    ),
                    "nx": config.nx,
                    "ny": config.ny,
                    "nz": config.nz,
                    "t_end": config.t_end,
                }
            )
    elif status == "error":
        _append_run_log(
            {
                "time": datetime.now().isoformat(),
                "status": "error",
                "message": message,
                "solver": st.session_state.get("run_config", ThermalSimulationConfig()).solver,
            }
        )
    elif status == "aborted":
        _append_run_log(
            {
                "time": datetime.now().isoformat(),
                "status": "aborted",
                "solver": st.session_state.get("run_config", ThermalSimulationConfig()).solver,
            }
        )


def _wobble_calculator_html(
    power_w: float,
    speed_m_s: float,
    wobble: WobbleParams,
    sigma_m: float,
    view_width_mm: float = 30.0,
) -> str:
    """Load the standalone wobble calculator and seed it with current setup values."""
    html_path = Path(__file__).with_name("wobble_calculator.html")
    html = html_path.read_text(encoding="utf-8")

    spot_size_mm = 2.0 * sigma_m * 1e3
    view_width_mm = max(view_width_mm, 30.0)

    html = html.replace(
        "const SPOT_SIZE_MM = 0.30;",
        f"const SPOT_SIZE_MM = {spot_size_mm:.4f};",
    )
    html = html.replace(
        "const viewWidthMM = 30;",
        f"const viewWidthMM = {view_width_mm:.2f};",
    )

    pattern = wobble.pattern.lower()
    pattern_map = {
        "circle": "circle",
        "line": "sine",
        "sine": "sine",
        "figure8": "figure8",
        "figure_8": "figure8",
        "infinity": "infinity",
        "lemniscate": "infinity",
    }
    html_pattern = pattern_map.get(pattern, "circle")

    amp_mm = wobble.amplitude * 2.0 * 1e3
    freq = wobble.frequency
    if freq <= 0.0:
        html_pattern = "sine"
        freq = 5.0
        amp_mm = 0.0
    else:
        freq = max(freq, 5.0)

    speed_mm_s = speed_m_s * 1e3
    amp_mm = max(0.0, min(amp_mm, 10.0))
    freq = min(freq, 50000.0)
    speed_mm_s = max(1.0, min(speed_mm_s, 300.0))

    init_script = f"""
<script>
(function() {{
  function sync(id, numId, value) {{
    const el = document.getElementById(id);
    const num = document.getElementById(numId);
    if (el) el.value = value;
    if (num) num.value = value;
  }}
  sync('amplitude', 'num-amplitude', {amp_mm:.4f});
  sync('frequency', 'num-frequency', {freq:.1f});
  sync('power', 'num-power', {power_w:.1f});
  sync('speed', 'num-speed', {speed_mm_s:.2f});
  const patternEl = document.getElementById('pattern');
  if (patternEl) patternEl.value = {html_pattern!r};
  const modEl = document.getElementById('modType');
  if (modEl) modEl.value = 'none';
  const palEl = document.getElementById('colorPalette');
  if (palEl) palEl.value = 'ironbow';
  if (typeof onParamChange === 'function') onParamChange();
}})();
</script>
"""

    html = html.replace("</body>", init_script + "\n</body>")
    return html


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
        _show(plots.plot_zone_map(x, y, result["history"].T_peak, material))
    with c2:
        _show(plots.plot_macro_section(report, material))

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
        if report.keyhole_depth is not None:
            st.metric("Keyhole depth", f"{report.keyhole_depth * 1e3:.2f} mm")
        if report.interface is not None and report.interface.n_stations > 0:
            st.metric(
                "Interface width (min/mean/max)",
                (
                    f"{report.interface.interface_width_min_mm:.2f} / "
                    f"{report.interface.interface_width_mean_mm:.2f} / "
                    f"{report.interface.interface_width_max_mm:.2f} mm"
                ),
                f"at {report.interface.interface_depth_mm:.2f} mm depth",
            )
        c1, c2 = st.columns(2)
        with c1:
            _show(plots.plot_cross_section(solution))
        with c2:
            _show(plots.plot_longitudinal_section(solution))
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
                _show(plots.plot_phases(micro.phases))
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
            _show(plots.plot_energy_density(x, y, energy))
        with c2:
            _show(plots.plot_wobble_concentration(y, energy))
        cols = st.columns(2)
        cols[0].metric("Spot speed (mean)", f"{wob.mean_beam_speed:.2f} m/s")
        cols[1].metric("Spot speed (peak)", f"{wob.peak_beam_speed:.2f} m/s")

    st.download_button(
        label="Download weld assessment (JSON)",
        data=json.dumps(report.as_dict(), indent=2).encode("utf-8"),
        file_name="weld_report.json",
        mime="application/json",
    )


def _load_preset_into_setup(preset: Preset, top_mm: float, bottom_mm: float) -> None:
    """Populate the Setup tab widgets from a preset so the user can adjust them before running."""
    st.session_state["mat_name"] = preset.material
    st.session_state["op_temp_last_mat"] = preset.material
    temps = material_temperatures(preset.material)
    t_min, t_max = float(min(temps)), float(max(temps))
    st.session_state["op_temp"] = float(min(max(preset.material_at_temperature, t_min), t_max))
    st.session_state["use_custom"] = False
    st.session_state["power"] = float(preset.power)
    st.session_state["efficiency"] = float(preset.efficiency)
    st.session_state["speed"] = float(preset.speed)
    st.session_state["spot_size_mm"] = float(preset.sigma * 2e3)
    st.session_state["path_type"] = preset.path_type
    st.session_state["center_x_mm"] = float(preset.center[0]) * 1e3
    st.session_state["center_y_mm"] = float(preset.center[1]) * 1e3
    st.session_state["radius_mm"] = float(preset.radius) * 1e3
    st.session_state["r_start_mm"] = float(preset.r_start) * 1e3
    st.session_state["r_end_mm"] = float(preset.r_end) * 1e3
    st.session_state["turns_path"] = float(preset.turns)

    if preset.path_type == "circle":
        _preset_path: WeldPath = WeldPath.circle(
            center=preset.center,
            radius=preset.radius,
            speed=preset.speed,
            turns=preset.turns,
        )
    elif preset.path_type == "spiral":
        _preset_path = WeldPath.spiral(
            center=preset.center,
            r_start=preset.r_start,
            r_end=preset.r_end,
            speed=preset.speed,
            turns=preset.turns,
        )
    else:
        _preset_path = WeldPath(start=preset.start, end=preset.end, speed=preset.speed)

    st.session_state["start_x_mm"] = float(_preset_path.start[0]) * 1e3
    st.session_state["start_y_mm"] = float(_preset_path.start[1]) * 1e3
    st.session_state["end_x_mm"] = float(_preset_path.end[0]) * 1e3
    st.session_state["end_y_mm"] = float(_preset_path.end[1]) * 1e3
    st.session_state["wobble_amp"] = float(preset.wobble_amp_um)
    st.session_state["wobble_freq"] = float(preset.wobble_freq_hz)
    st.session_state["wobble_pattern"] = preset.wobble_pattern
    st.session_state["Lx_mm"] = float(preset.Lx) * 1e3
    st.session_state["Ly_mm"] = float(preset.Ly) * 1e3

    total_mm = top_mm + bottom_mm
    st.session_state["plate_thickness_mm"] = float(total_mm)
    st.session_state["T1_mm"] = float(total_mm)
    st.session_state["top_thickness_mm"] = float(top_mm)

    st.session_state["solver"] = preset.solver
    st.session_state["nx"] = int(preset.nx)
    st.session_state["ny"] = int(preset.ny)
    st.session_state["nz"] = int(preset.nz)

    if preset.t_end is not None:
        t_end = float(preset.t_end)
    else:
        t_end = min(max(1.3 * _preset_path.duration, 0.1), 50.0)
    st.session_state["t_end"] = t_end
    st.session_state["dt"] = float(preset.dt)

    probe = preset.probe
    if probe is None:
        probe = _preset_path.nominal_position(_preset_path.duration / 2.0)
    st.session_state["probe_x_mm"] = float(probe[0]) * 1e3
    st.session_state["probe_y_mm"] = float(probe[1]) * 1e3


def _run_preset(preset: Preset) -> None:
    """Run a configured preset and populate the shared session state used by the result tabs."""
    try:
        config = replace(
            preset.make_config(),
            store_3d_frames=True,
            frame_interval=0.02,
        )
    except Exception as exc:
        st.error(f"Could not load preset: {exc}")
        return

    material = config.material
    weld = config.weld
    path = config.path
    wobble = config.wobble or WobbleParams(0.0, 0.0, "circle")
    assert isinstance(material, Material)
    assert weld is not None and path is not None

    # Heat-signature preview (used by the Wobble signature tab).
    x = np.linspace(0.0, config.Lx, config.nx)
    y = np.linspace(0.0, config.Ly, config.ny)
    t_max = min(config.t_end, path.duration * 1.2)
    t_from, t_to, heat_dt = sampling_window(t_max, wobble, min(0.002, config.dt))
    Q = heat_signature(
        path=path,
        wobble=wobble,
        power=weld.power,
        efficiency=weld.efficiency,
        sigma=weld.sigma,
        h=config.T1,
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
        dt=min(heat_dt / 2.0, 0.0005),
        t_start=t_from,
    )

    bar = st.progress(0.0, text=f"Running {preset.name} preset...")

    def report_progress(fraction: float) -> None:
        bar.progress(
            min(fraction, 1.0),
            text=f"Running {preset.name}... {fraction * 100:.0f}%",
        )

    try:
        result = run_thermal_simulation(config, on_progress=report_progress)
        report = build_report(config, result)
    except WeldSimError as exc:
        st.error(str(exc))
        return
    finally:
        bar.empty()

    # Populate the state used by the Wobble / Weld result / Thermal field tabs.
    st.session_state["Q"] = Q
    st.session_state["x_traj"] = x_traj
    st.session_state["y_traj"] = y_traj
    st.session_state["x"] = x
    st.session_state["y"] = y
    st.session_state["Lx"] = config.Lx
    st.session_state["Ly"] = config.Ly
    st.session_state["Lx_mm"] = config.Lx * 1e3
    st.session_state["Ly_mm"] = config.Ly * 1e3
    st.session_state["weld"] = weld
    st.session_state["path"] = path
    st.session_state["wobble"] = wobble
    st.session_state["material"] = material
    st.session_state["t_end"] = config.t_end
    st.session_state["T1"] = config.T1
    st.session_state["T1_mm"] = config.T1 * 1e3

    st.session_state["op_temp_last_mat"] = preset.material
    temps = material_temperatures(preset.material)
    t_min, t_max = float(min(temps)), float(max(temps))
    st.session_state["op_temp"] = float(min(max(preset.material_at_temperature, t_min), t_max))

    st.session_state["path_type"] = preset.path_type
    st.session_state["center_x_mm"] = float(preset.center[0]) * 1e3
    st.session_state["center_y_mm"] = float(preset.center[1]) * 1e3
    st.session_state["radius_mm"] = float(preset.radius) * 1e3
    st.session_state["r_start_mm"] = float(preset.r_start) * 1e3
    st.session_state["r_end_mm"] = float(preset.r_end) * 1e3
    st.session_state["turns_path"] = float(preset.turns)
    st.session_state["start_x_mm"] = float(path.start[0]) * 1e3
    st.session_state["start_y_mm"] = float(path.start[1]) * 1e3
    st.session_state["end_x_mm"] = float(path.end[0]) * 1e3
    st.session_state["end_y_mm"] = float(path.end[1]) * 1e3

    st.session_state["thermal_result"] = result
    st.session_state["weld_report"] = report
    st.session_state["thermal_config"] = config
    st.session_state["thermal_material"] = material
    st.session_state["thermal_weld"] = weld
    st.session_state["thermal_x1"] = path.end[0]
    st.session_state["thermal_y1"] = path.end[1]

    st.success(f"{preset.name} completed — see Weld result / Thermal field tabs.")


def _page_thermal_and_wobble():
    st.header("Weld simulation")
    st.caption(
        "Enter the process, material and wobble parameters; the app solves the weld "
        "locally and reports fusion zone, penetration, HAZ metallurgy, distortion and "
        "heat concentration. No internet or external solver needed."
    )

    # --- Presets: one-click runs for common configurations ---
    if PRESETS:
        with st.expander("Presets", expanded=False):
            preset_options = ["(none)"] + list(PRESETS.keys())
            preset_name = st.selectbox("Quick-start preset", preset_options)

            if preset_name != "(none)":
                preset = PRESETS[preset_name]

                # Sync the stackup inputs when the selected preset changes.
                last = st.session_state.get("_last_preset")
                if last != preset_name:
                    st.session_state["_last_preset"] = preset_name
                    st.session_state["_preset_top"] = preset.top_thickness_mm
                    st.session_state["_preset_bottom"] = preset.bottom_thickness_mm

                st.markdown(f"**{preset_name}** — {preset.description}")
                for note in preset.notes:
                    st.caption(note)

                c_top, c_bot = st.columns(2)
                with c_top:
                    top_mm = st.number_input(
                        "Top sheet thickness (mm)",
                        0.05,
                        10.0,
                        key="_preset_top",
                    )
                with c_bot:
                    bottom_mm = st.number_input(
                        "Bottom sheet thickness (mm)",
                        0.05,
                        10.0,
                        key="_preset_bottom",
                    )

                params = {
                    "Power (W)": preset.power,
                    "Efficiency": preset.efficiency,
                    "Speed (m/s)": preset.speed,
                    "Spot size (mm)": round(preset.sigma * 2e3, 2),
                    "Total stack thickness (mm)": top_mm + bottom_mm,
                    "Heat-spreading depth T1 (mm)": top_mm + bottom_mm,
                    "Solver": preset.solver,
                    "Grid": f"{preset.nx} × {preset.ny} × {preset.nz}",
                }
                st.json(params)

                col_run, col_load = st.columns(2)
                with col_run:
                    if st.button(f"Run {preset_name}", type="primary"):
                        configured = replace(
                            preset,
                            top_thickness_mm=top_mm,
                            bottom_thickness_mm=bottom_mm,
                        )
                        _run_preset(configured)
                with col_load:
                    if st.button("Load into setup", type="secondary"):
                        _load_preset_into_setup(preset, top_mm, bottom_mm)
                        st.success("Preset loaded into the Setup tab. Adjust and click Run.")

    tab_setup, tab_wobble, tab_weld, tab_thermal = st.tabs(
        ["Setup", "Wobble signature", "Weld result", "Thermal field"]
    )

    with tab_setup:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Material")
            mat_options = list_materials()
            mat_name = st.selectbox("Material", mat_options, key="mat_name")

            temps = material_temperatures(mat_name)
            t_min, t_max = float(min(temps)), float(max(temps))
            liquidus = material_liquidus(mat_name)
            nearest_to_liquidus = float(min(temps, key=lambda t: abs(t - liquidus)))

            last_mat = st.session_state.get("op_temp_last_mat")
            if last_mat != mat_name or "op_temp" not in st.session_state:
                st.session_state["op_temp"] = nearest_to_liquidus
                st.session_state["op_temp_last_mat"] = mat_name
            else:
                current = st.session_state["op_temp"]
                if current < t_min or current > t_max:
                    st.session_state["op_temp"] = nearest_to_liquidus

            op_temp = st.slider(
                "Evaluate material properties at (K)",
                t_min,
                t_max,
                step=1.0,
                key="op_temp",
            )
            st.caption(
                "The solver uses one constant k, rho, cp set sampled from the "
                "material's temperature table at this temperature."
            )
            use_custom = st.checkbox("Use custom k/rho/cp", key="use_custom")
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
            power = st.slider("Power (W)", 100.0, 8000.0, 1500.0, 50.0, key="power")
            efficiency = st.slider("Efficiency", 0.1, 1.0, 0.8, 0.05, key="efficiency")
            speed = st.slider("Travel speed (m/s)", 0.001, 0.05, 0.01, 0.001, key="speed")
            spot_size_mm = st.number_input(
                "Spot size (mm)", 0.05, 5.0, 1.0, 0.05, key="spot_size_mm"
            )
            sigma = spot_size_mm * 1e-3 / 2.0

        st.subheader("Weld path")
        path_type = st.selectbox(
            "Path type",
            ["line", "circle", "spiral"],
            key="path_type",
            format_func=lambda p: {"line": "Line", "circle": "Circle", "spiral": "Spiral"}[p],
        )

        if path_type == "line":
            c3, c4 = st.columns(2)
            with c3:
                x0_mm = st.number_input("Start x (mm)", 0.0, 500.0, 10.0, 0.1, key="start_x_mm")
                y0_mm = st.number_input("Start y (mm)", 0.0, 200.0, 25.0, 0.1, key="start_y_mm")
            with c4:
                x1_mm = st.number_input("End x (mm)", 0.0, 500.0, 70.0, 0.1, key="end_x_mm")
                y1_mm = st.number_input("End y (mm)", 0.0, 200.0, 25.0, 0.1, key="end_y_mm")
            x0, y0 = x0_mm * 1e-3, y0_mm * 1e-3
            x1, y1 = x1_mm * 1e-3, y1_mm * 1e-3
            path = WeldPath(start=(x0, y0), end=(x1, y1), speed=speed)
        elif path_type == "circle":
            c3, c4, c5 = st.columns(3)
            with c3:
                cx_mm = st.number_input("Center x (mm)", 0.0, 500.0, 40.0, 0.1, key="center_x_mm")
                cy_mm = st.number_input("Center y (mm)", 0.0, 200.0, 25.0, 0.1, key="center_y_mm")
            with c4:
                radius_mm = st.number_input("Radius (mm)", 0.1, 100.0, 5.0, 0.1, key="radius_mm")
            with c5:
                turns_path = st.number_input("Turns", 0.1, 10.0, 1.0, 0.1, key="turns_path")
            path = WeldPath.circle(
                center=(cx_mm * 1e-3, cy_mm * 1e-3),
                radius=radius_mm * 1e-3,
                speed=speed,
                turns=turns_path,
            )
        else:  # spiral
            c3, c4, c5 = st.columns(3)
            with c3:
                cx_mm = st.number_input("Center x (mm)", 0.0, 500.0, 40.0, 0.1, key="center_x_mm")
                cy_mm = st.number_input("Center y (mm)", 0.0, 200.0, 25.0, 0.1, key="center_y_mm")
            with c4:
                r_start_mm = st.number_input(
                    "Start radius (mm)", 0.0, 100.0, 1.0, 0.1, key="r_start_mm"
                )
                r_end_mm = st.number_input("End radius (mm)", 0.0, 100.0, 7.0, 0.1, key="r_end_mm")
            with c5:
                turns_path = st.number_input("Turns", 0.1, 10.0, 1.5, 0.1, key="turns_path")
            path = WeldPath.spiral(
                center=(cx_mm * 1e-3, cy_mm * 1e-3),
                r_start=r_start_mm * 1e-3,
                r_end=r_end_mm * 1e-3,
                speed=speed,
                turns=turns_path,
            )

        if path.length <= 0.0:
            st.error("The weld path has zero length. Increase the radius or end radius.")
            return
        st.write(
            f"Path length: **{path.length*1e3:.2f} mm**, " f"travel time: **{path.duration:.3f} s**"
        )

        st.subheader("Wobble")
        c5, c6, c7 = st.columns(3)
        with c5:
            wobble_amp = st.slider(
                "Wobble amplitude (µm)", 0.0, 1000.0, 400.0, 25.0, key="wobble_amp"
            )
        with c6:
            # Low default so individual loops are resolvable in the path preview;
            # production wobble is typically several hundred Hz.
            wobble_freq = st.slider("Wobble frequency (Hz)", 0, 2000, 20, 10, key="wobble_freq")
        with c7:
            pattern = st.selectbox(
                "Pattern",
                ["circle", "line", "figure8", "infinity"],
                key="wobble_pattern",
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
            Lx_mm = st.number_input("Plate length Lx (mm)", 1.0, 500.0, 80.0, 1.0, key="Lx_mm")
            Ly_mm = st.number_input("Plate width Ly (mm)", 1.0, 200.0, 50.0, 1.0, key="Ly_mm")
            Lx, Ly = Lx_mm * 1e-3, Ly_mm * 1e-3
            plate_thickness = st.number_input(
                "Plate thickness (mm)", 0.1, 50.0, 3.0, 0.1, key="plate_thickness_mm"
            )
            T1_mm = st.number_input("Heat-spreading depth h (mm)", 0.1, 50.0, 3.0, 0.1, key="T1_mm")
            T1 = T1_mm * 1e-3
            top_thickness_mm = st.number_input(
                "Top sheet thickness (mm) — 0 for single plate",
                0.0,
                50.0,
                0.0,
                0.1,
                key="top_thickness_mm",
                help=(
                    "For a 2t lap joint, set the top sheet thickness here and the total "
                    "stack thickness in Plate thickness. The interface plane and width "
                    "are reported at this depth."
                ),
            )
            st.caption(
                "The thermal model spreads the beam flux over depth h. Set it to the "
                "expected penetration for a partial-penetration pass, or to the plate "
                "thickness for a fully penetrating one."
            )
        with c9:
            solver = st.radio(
                "Solver",
                ["3d", "2d"],
                key="solver",
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
            nx = st.slider("Grid points X", 21, 201, 81, 2, key="nx")
            ny = st.slider("Grid points Y", 11, 101, 41, 2, key="ny")
            nz = st.slider(
                "Grid points through thickness", 5, 61, 13, 2, disabled=solver == "2d", key="nz"
            )
            # Long enough for the measured section to cool through 500 C, so t8/5
            # is available. The 3D solve pays for every extra second of simulated
            # time, so it gets the shorter tail.
            tail = 1.3 if solver == "3d" else 2.0
            t_end = st.number_input(
                "Simulation time (s)",
                0.1,
                50.0,
                float(min(max(tail * path.duration, 0.1), 50.0)),
                0.1,
                key="t_end",
            )
            dt = st.number_input(
                "Time step (s)", 0.001, 0.5, 0.01, 0.001, disabled=solver == "3d", key="dt"
            )
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
        mid_t = path.duration / 2.0 if path.duration > 0.0 else 0.0
        mid_x_mm, mid_y_mm = (x * 1e3 for x in path.nominal_position(mid_t))
        px_default = min(mid_x_mm, Lx_mm)
        py_default = min(mid_y_mm, Ly_mm)

        px_kwargs: dict[str, Any] = {
            "min_value": 0.0,
            "max_value": Lx_mm,
            "step": 0.1,
            "key": "probe_x_mm",
        }
        if "probe_x_mm" in st.session_state:
            st.session_state["probe_x_mm"] = min(st.session_state["probe_x_mm"], Lx_mm)
        else:
            px_kwargs["value"] = px_default
        py_kwargs: dict[str, Any] = {
            "min_value": 0.0,
            "max_value": Ly_mm,
            "step": 0.1,
            "key": "probe_y_mm",
        }
        if "probe_y_mm" in st.session_state:
            st.session_state["probe_y_mm"] = min(st.session_state["probe_y_mm"], Ly_mm)
        else:
            py_kwargs["value"] = py_default

        c10, c11 = st.columns(2)
        with c10:
            px_mm = st.number_input("Probe x (mm)", **px_kwargs)
        with c11:
            py_mm = st.number_input("Probe y (mm)", **py_kwargs)
        px, py = px_mm * 1e-3, py_mm * 1e-3

        weld = WeldParams(
            power=power,
            efficiency=efficiency,
            speed=speed,
            start_pos=path.start,
            direction="x",
            sigma=sigma,
        )
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
            if _simulation_running():
                run_label = "Check status"
            else:
                run_label = "Run 3D weld simulation" if solver == "3d" else "Run 2D weld simulation"
            run_pressed = st.button(run_label, key="run_thermal_button", width="stretch")

        st.session_state["weld"] = weld
        st.session_state["path"] = path
        st.session_state["wobble"] = wobble
        st.session_state["material"] = material

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
            st.session_state["show_3d_animation"] = False
            top_thickness_m = (
                top_thickness_mm / 1e3 if 0.0 < top_thickness_mm < plate_thickness else None
            )
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
                top_thickness=top_thickness_m,
                store_3d_frames=solver == "3d",
                frame_interval=0.02,
                max_stored_frames=1_000,
                path=path,
                wobble=wobble,
                probe=(px, py) if solver == "2d" else None,
                solver=solver,
            )
            if _simulation_running():
                st.info(
                    "A simulation is already running. Refresh to check progress or click Abort."
                )
            else:
                q = queue.Queue()
                event = threading.Event()
                thread = threading.Thread(
                    target=_solver_worker,
                    args=(config, q, event),
                    daemon=True,
                )
                thread.start()
                st.session_state["run_thread"] = thread
                st.session_state["run_queue"] = q
                st.session_state["run_abort"] = event
                st.session_state["run_config"] = config
                st.session_state["run_status"] = "running"
                st.session_state["run_progress"] = 0.0
                st.session_state.pop("run_message", None)
                st.session_state.pop("weld_report", None)
                st.session_state.pop("thermal_result", None)
                st.success("Simulation started in the background. Click Refresh to poll status.")

    # --- Wobble tab: embedded standalone wobble calculator ---
    with tab_wobble:
        weld_wobble = st.session_state.get("weld")
        wobble_state = st.session_state.get("wobble")
        if weld_wobble is not None and wobble_state is not None:
            view_width = st.session_state.get("Lx_mm", 80.0) + 6.0
            html = _wobble_calculator_html(
                power_w=weld_wobble.power,
                speed_m_s=weld_wobble.speed,
                wobble=wobble_state,
                sigma_m=weld_wobble.sigma,
                view_width_mm=view_width,
            )
            components.html(html, height=850, scrolling=True)
        else:
            st.info(
                "Set power, speed and wobble on the **Setup** tab and click "
                "**Run** to load the wobble calculator."
            )

    # --- Weld result tab: what the parameters actually produce ---
    with tab_weld:
        _process_run_queue()

        if _simulation_running():
            progress = st.session_state.get("run_progress", 0.0)
            st.progress(min(progress, 1.0), text=f"Running... {progress * 100:.0f}%")
            c1, c2 = st.columns(2)
            with c1:
                if st.button("Abort", type="secondary", key="abort_run_button"):
                    event = st.session_state.get("run_abort")
                    if event is not None:
                        event.set()
                    st.warning("Abort requested. Click Refresh to see the final status.")
            with c2:
                st.button("Refresh", key="refresh_run_button")
        else:
            status = st.session_state.get("run_status")
            if status == "completed" and "weld_report" in st.session_state:
                _render_weld_report(
                    st.session_state["weld_report"],
                    st.session_state["thermal_result"],
                    st.session_state["thermal_config"],
                )
            elif status == "error":
                st.error(st.session_state.get("run_message", "Simulation failed."))
            elif status == "aborted":
                st.warning("Simulation aborted by user.")
            elif status is None:
                st.info("Run a simulation on the **Setup** tab to get the weld assessment.")

        # Simulation log
        with st.expander("Simulation log", expanded=False):
            log = _load_run_log()
            if log:
                st.dataframe(log)
            else:
                st.info("No completed simulations yet.")

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

            with st.expander("2D plot settings (resolution & contrast)", expanded=False):
                pw, ph = st.columns(2)
                with pw:
                    st.slider("Plot width (px)", 0, 1600, 0, 50, key="plot_width")
                with ph:
                    st.slider("Plot height (px)", 0, 1200, 0, 50, key="plot_height")
                st.selectbox(
                    "Colorscale",
                    ["Inferno", "Hot", "Plasma", "Jet", "Viridis", "Magma"],
                    index=0,
                    key="plot_colorscale",
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(
                        "T low clip (K, 0 = auto)", 0.0, 5000.0, 0.0, 10.0, key="plot_tmin"
                    )
                with c2:
                    st.number_input(
                        "T high clip (K, 0 = auto)", 0.0, 8000.0, 0.0, 10.0, key="plot_tmax"
                    )

            c1, c2 = st.columns(2)
            with c1:
                _show(plots.plot_temperature_2d(x, y, T))
            with c2:
                solution3d = result.get("solution3d")
                if solution3d is not None:
                    _show(
                        plots.plot_weld_3d(
                            solution3d,
                            material,
                            path=st.session_state.get("path"),
                            wobble=st.session_state.get("wobble"),
                            top_thickness=result.get("top_thickness"),
                        )
                    )
                else:
                    _show(plots.plot_temperature_surface(x, y, T))

            with st.expander("3D weld animation", expanded=False):
                solution3d = result.get("solution3d")
                if solution3d is not None:
                    c1, c2 = st.columns(2)
                    with c1:
                        anim_time_scale = st.slider(
                            "Playback time scale",
                            0.1,
                            20.0,
                            1.0,
                            0.1,
                            key="anim_time_scale",
                        )
                    with c2:
                        anim_heat_filter = st.selectbox(
                            "Heat signature filter",
                            ["Melt pool", "Melt pool + HAZ", "HAZ only"],
                            key="anim_heat_filter",
                        )
                    c3, c4 = st.columns(2)
                    with c3:
                        anim_max_frames = st.slider(
                            "Max frames", 30, 300, 120, 10, key="anim_max_frames"
                        )
                    with c4:
                        anim_surface_count = st.slider(
                            "Volume surfaces", 2, 30, 8, 1, key="anim_surface_count"
                        )
                    if st.button("Go", key="go_3d_animation"):
                        st.session_state["show_3d_animation"] = True
                    if st.session_state.get("show_3d_animation"):
                        if st.button("Hide animation", key="hide_3d_animation"):
                            st.session_state["show_3d_animation"] = False
                        else:
                            with st.spinner("Building 3D animation..."):
                                _show(
                                    plots.plot_weld_3d_animation(
                                        solution3d,
                                        material,
                                        path=st.session_state.get("path"),
                                        wobble=st.session_state.get("wobble"),
                                        top_thickness=result.get("top_thickness"),
                                        time_scale=anim_time_scale,
                                        max_frames=int(anim_max_frames),
                                        heat_filter=str(anim_heat_filter),
                                        surface_count=int(anim_surface_count),
                                    )
                                )
                else:
                    st.info("3D animation is only available for 3D through-thickness runs.")

            with st.expander("Temperature profiles", expanded=True):
                prof1, prof2, prof3 = st.tabs(["Longitudinal", "Transverse", "Probe history"])
                with prof1:
                    j_mid = int(round(y1 / (y[1] - y[0])))
                    j_mid = min(max(j_mid, 0), len(y) - 1)
                    _show(
                        plots.plot_temperature_profile(
                            x, T[:, j_mid], "x (mm)", f"T along y = {y[j_mid]*1e3:.1f} mm", material
                        )
                    )
                with prof2:
                    i_end = int(round(x1 / (x[1] - x[0])))
                    i_end = min(max(i_end, 0), len(x) - 1)
                    _show(
                        plots.plot_temperature_profile(
                            y,
                            T[i_end, :],
                            "y (mm)",
                            f"T across x = {x[i_end]*1e3:.1f} mm",
                            material,
                        )
                    )
                with prof3:
                    if "t" in result and "T_probe" in result:
                        _show(plots.plot_probe_history(result["t"], result["T_probe"], material))
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
        _show(plots.plot_measured_vs_predicted(comparison))
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
