"""Streamlit GUI for Weld Sim — 2D thermal + wobble + 3D keyhole CFD."""

from __future__ import annotations

import csv
import io
import math
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from weldsim.materials import Material, list_materials, load_material
from weldsim.simulation import ThermalSimulationConfig, run_thermal_simulation
from weldsim.types import MaterialParams, WeldParams
from weldsim.weld_path import (
    WeldPath,
    WobbleParams,
    beam_trajectory,
    heat_signature,
    wobble_animation_gif,
)


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
    surf = ax.plot_surface(
        X * 1e3, Y * 1e3, T, cmap=cm.inferno, linewidth=0, antialiased=True
    )
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


def _page_thermal_and_wobble():
    st.header("2D Thermal + Wobble Calculator")

    tab_setup, tab_wobble, tab_thermal = st.tabs(["Setup", "Wobble signature", "Thermal result"])

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
            st.write(f"**{material.name}**: k={material.thermal_conductivity:.1f}, "
                     f"rho={material.density:.0f}, cp={material.specific_heat:.0f}")

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
        st.write(f"Path length: **{length*1e3:.2f} mm**, travel time: **{length/speed:.3f} s**")

        st.subheader("Wobble")
        c5, c6, c7 = st.columns(3)
        with c5:
            wobble_amp = st.slider("Wobble amplitude (µm)", 0.0, 1000.0, 100.0, 25.0)
        with c6:
            wobble_freq = st.slider("Wobble frequency (Hz)", 0, 2000, 100, 10)
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
            T1 = st.number_input("Effective thickness h (m)", 0.0005, 0.05, 0.005, 0.0005)
        with c9:
            nx = st.slider("Grid points X", 21, 201, 81, 2)
            ny = st.slider("Grid points Y", 11, 101, 41, 2)
            t_end = st.number_input("Simulation time (s)", 0.1, 50.0, length / speed, 0.1)
            dt = st.number_input("Time step (s)", 0.001, 0.5, 0.01, 0.001)

        st.subheader("Thermal probe")
        c10, c11 = st.columns(2)
        with c10:
            px = st.number_input("Probe x (m)", 0.0, Lx, (x1 + 0.005), 0.001)
        with c11:
            py = st.number_input("Probe y (m)", 0.0, Ly, y1, 0.001)

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
            preview_pressed = st.button("Go (draw heat signature)", type="primary", width='stretch')
        with run:
            run_pressed = st.button("Run full 2D thermal simulation", width='stretch')

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
                Q = heat_signature(
                    path=path,
                    wobble=wobble,
                    power=power,
                    efficiency=efficiency,
                    sigma=sigma,
                    h=T1,
                    x=x,
                    y=y,
                    t_end=t_max,
                    dt=min(0.002, dt),
                )
                x_traj, y_traj = beam_trajectory(
                    path=path,
                    wobble=wobble,
                    t_end=t_max,
                    dt=min(0.0005, dt / 2),
                )
            st.session_state["Q"] = Q
            st.session_state["x_traj"] = x_traj
            st.session_state["y_traj"] = y_traj
            st.session_state["x"] = x
            st.session_state["y"] = y

        if run_pressed:
            with st.spinner("Running thermal simulation..."):
                config = ThermalSimulationConfig(
                    nx=nx,
                    ny=ny,
                    Lx=Lx,
                    Ly=Ly,
                    t_end=t_end,
                    dt=dt,
                    weld=weld,
                    material=material,
                    output_file=None,
                    T1=T1,
                    path=path,
                    wobble=wobble,
                    probe=(px, py),
                )
                st.session_state["thermal_result"] = run_thermal_simulation(config)
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
            st.pyplot(_plot_beam_path(x_traj, y_traj, Lx, Ly))
            st.pyplot(_plot_heat_signature(x, y, Q, x_traj, y_traj))

            with st.expander("Run animation", expanded=True):
                anim_col, anim_params = st.columns([1, 3])
                with anim_col:
                    run_anim = st.button("Run animation", type="secondary", width='stretch')
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
                                heat_dt=0.002,
                                fps=fps,
                                gif_width=400,
                            )
                            st.session_state["wobble_gif"] = gif
                    st.image(st.session_state["wobble_gif"], caption="Wobbled weld animation — brighter = more dwell time")
        else:
            st.info("Click **Go (draw heat signature)** on the **Setup** tab to generate the wobble preview.")

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
                st.pyplot(_plot_temperature_2d(x, y, T))
            with c2:
                st.pyplot(_plot_temperature_3d(x, y, T))

            with st.expander("Temperature profiles", expanded=True):
                prof1, prof2, prof3 = st.tabs(["Longitudinal", "Transverse", "Probe history"])
                with prof1:
                    j_mid = int(round(y1 / (y[1] - y[0])))
                    j_mid = min(max(j_mid, 0), len(y) - 1)
                    st.pyplot(_plot_temperature_profile(x, T[:, j_mid], "x (mm)", f"T along y = {y[j_mid]*1e3:.1f} mm", material))
                with prof2:
                    i_end = int(round(x1 / (x[1] - x[0])))
                    i_end = min(max(i_end, 0), len(x) - 1)
                    st.pyplot(_plot_temperature_profile(y, T[i_end, :], "y (mm)", f"T across x = {x[i_end]*1e3:.1f} mm", material))
                with prof3:
                    if "t" in result and "T_probe" in result:
                        st.pyplot(_plot_probe_history(result["t"], result["T_probe"], material))
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
            st.info("Click **Run full 2D thermal simulation** on the **Setup** tab to see results.")


def _page_keyhole_cfd():
    st.header("3D Keyhole Laser-Welding CFD")
    st.write(
        "Full 3D VOF keyhole model with enthalpy-porosity melting, vapor recoil, "
        "and preCICE↔CalculiX distortion coupling."
    )

    root = Path(__file__).resolve().parent.parent
    keyhole_root = root / "keyhole-cfd"

    if not keyhole_root.exists():
        st.error("`keyhole-cfd/` directory not found.")
        return

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("OpenFOAM case")
        if st.button("Generate workpiece STL", width='stretch'):
            with st.spinner("Generating STL..."):
                rc, out, err = _run_python_script(
                    "keyhole-cfd/scripts/generate_workpiece_stl.py"
                )
            if rc == 0:
                st.success("STL generated")
                st.code(out, language="text")
            else:
                st.error(f"Failed:\n{err}")
    with c2:
        st.subheader("Case config")
        if st.button("Configure OpenFOAM case", width='stretch'):
            with st.spinner("Configuring case..."):
                rc, out, err = _run_python_script(
                    "keyhole-cfd/scripts/configure_case.py"
                )
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
                st.image(png, width='stretch')
        except Exception as e:
            st.warning(f"Could not render STL: {e}")
    else:
        st.info("Click **Generate workpiece STL** to create the 3D workpiece.")

    st.subheader("Build & run on WSL2 / Ubuntu")
    st.code(
        """cd ~/WeldSYM/keyhole-cfd
python3 scripts/configure_case.py
blockMesh
laserKeyholeVoF
# or parallel:
# decomposePar; mpirun -np 4 laserKeyholeVoF -parallel; reconstructPar
""",
        language="bash",
    )


def _page_docs():
    st.header("Documentation")
    st.markdown(
        """
        - **2D thermal solver**: explicit finite differences with a moving Gaussian surface source.
        - **Wobble patterns**: circle, line/sine, figure-8, infinity. Amplitude in µm, frequency in Hz.
        - **Heat signature**: time-integrated surface Gaussian divided by effective thickness.
        - **3D keyhole CFD**: OpenFOAM VOF solver. Build with `wmake` in WSL2.
        - **Coupled distortion**: preCICE + CalculiX (stub config in `keyhole-cfd/scripts/preCICE_config.xml`).
        """
    )


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
        ["2D Thermal + Wobble", "3D Keyhole CFD", "Docs"],
    )

    if page == "2D Thermal + Wobble":
        _page_thermal_and_wobble()
    elif page == "3D Keyhole CFD":
        _page_keyhole_cfd()
    elif page == "Docs":
        _page_docs()


if __name__ == "__main__":
    main()
