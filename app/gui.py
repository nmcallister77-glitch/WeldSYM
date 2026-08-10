"""Streamlit GUI for Weld Sim — 2D thermal + 3D keyhole CFD preview."""

from __future__ import annotations

import csv
import io
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from weldsim.simulation import (
    WeldParams,
    MaterialParams,
    ThermalSimulationConfig,
    run_thermal_simulation,
)


def plot_temperature_2d(x: np.ndarray, y: np.ndarray, T: np.ndarray) -> plt.Figure:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots()
    cmap = ax.pcolormesh(X * 1e3, Y * 1e3, T, shading="auto", cmap="inferno")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperature field (K)")
    fig.colorbar(cmap, ax=ax, label="Temperature (K)")
    return fig


def plot_temperature_3d(x: np.ndarray, y: np.ndarray, T: np.ndarray) -> plt.Figure:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        X * 1e3, Y * 1e3, T, cmap=cm.inferno, linewidth=0, antialiased=True
    )
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("T (K)")
    ax.set_title("3D temperature field")
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label="Temperature (K)")
    return fig


def run_python_script(script: str, *args: str) -> tuple[int, str, str]:
    """Run a project Python script and return (returncode, stdout, stderr)."""
    root = Path(__file__).resolve().parent.parent
    cmd = [sys.executable, str(root / script)]
    cmd.extend(args)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=root)
    return proc.returncode, proc.stdout, proc.stderr


def page_thermal_2d():
    st.header("2D Welding Thermal Simulation")
    st.write(
        "Finite-difference transient conduction with a moving Gaussian surface source."
    )

    col1, col2 = st.columns(2)
    with col1:
        power = st.slider("Power (W)", 500.0, 5000.0, 1500.0, 50.0)
        efficiency = st.slider("Efficiency", 0.3, 1.0, 0.8, 0.05)
        speed = st.slider("Travel speed (m/s)", 0.001, 0.02, 0.005, 0.001)
    with col2:
        Lx = st.number_input("Plate length Lx (m)", 0.05, 0.5, 0.1, 0.01)
        Ly = st.number_input("Plate width Ly (m)", 0.02, 0.2, 0.05, 0.01)
        T1 = st.number_input("Effective thickness (m)", 0.001, 0.05, 0.005, 0.001)

    nx = st.slider("Grid points in X", 21, 101, 41, 2)
    ny = st.slider("Grid points in Y", 11, 51, 21, 2)
    t_end = st.number_input("Simulation time (s)", 1.0, 100.0, 5.0, 1.0)
    dt = st.number_input("Time step (s)", 0.001, 1.0, 0.05, 0.005)

    if st.button("Run simulation"):
        weld = WeldParams(
            power=power,
            efficiency=efficiency,
            speed=speed,
            start_pos=(0.01, Ly / 2),
            direction="x",
        )
        mat = MaterialParams()

        config = ThermalSimulationConfig(
            nx=nx,
            ny=ny,
            Lx=Lx,
            Ly=Ly,
            t_end=t_end,
            dt=dt,
            weld=weld,
            material=mat,
            output_file=None,
            T1=T1,
        )

        with st.spinner("Running 2D simulation..."):
            result = run_thermal_simulation(config)

        x, y, T = result["x"], result["y"], result["T"]
        T_min, T_max = float(T.min()), float(T.max())

        st.write(f"**Temperature range:** {T_min:.1f} K – {T_max:.1f} K")

        tab1, tab2 = st.tabs(["2D contour", "3D surface"])
        with tab1:
            st.pyplot(plot_temperature_2d(x, y, T))
        with tab2:
            st.pyplot(plot_temperature_3d(x, y, T))

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


def page_keyhole_cfd():
    st.header("3D Keyhole Laser-Welding CFD")
    st.write(
        "OpenFOAM / preCICE / CalculiX workflow for 3D keyhole, melt pool, and distortion."
    )

    root = Path(__file__).resolve().parent.parent
    keyhole_root = root / "keyhole-cfd"

    if not keyhole_root.exists():
        st.error("`keyhole-cfd/` directory not found.")
        return

    with st.expander("OpenFOAM case (openfoam/)"):
        st.write(
            "- **Solver:** `laserKeyholeVoF` (VOF + enthalpy porosity + recoil pressure)\n"
            "- **Mesh:** `blockMesh` + optional `snappyHexMesh` with AMR\n"
            "- **Physics:** 3D moving Gaussian heat source, boiling, vapor recoil, Marangoni\n"
            "- **Coupling:** preCICE → CalculiX FEA for thermomechanical distortion"
        )

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Generate workpiece STL"):
            with st.spinner("Generating STL..."):
                rc, out, err = run_python_script(
                    "keyhole-cfd/scripts/generate_workpiece_stl.py"
                )
            if rc == 0:
                st.success("STL generated")
                st.code(out, language="text")
            else:
                st.error(f"Failed:\n{err}")

    with col2:
        if st.button("Configure OpenFOAM case"):
            with st.spinner("Configuring case..."):
                rc, out, err = run_python_script(
                    "keyhole-cfd/scripts/configure_case.py"
                )
            if rc == 0:
                st.success("Case configured")
                st.code(out, language="text")
            else:
                st.error(f"Failed:\n{err}")

    stl_path = keyhole_root / "openfoam" / "triSurface" / "workpiece.stl"
    if stl_path.exists():
        st.write(f"Workpiece STL: `{stl_path}`")
        try:
            import pyvista as pv

            mesh = pv.read(stl_path)
            st.write(f"Mesh has {mesh.n_points} points and {mesh.n_cells} cells.")

            plotter = pv.Plotter(off_screen=True, window_size=(800, 600))
            plotter.add_mesh(mesh, color="silver", show_edges=True)
            plotter.add_axes()
            png = plotter.screenshot()
            st.image(png, caption="Workpiece STL preview")
        except Exception as e:
            st.warning(f"Could not render STL: {e}")
    else:
        st.info("Click **Generate workpiece STL** to create the 3D workpiece.")


def main():
    st.set_page_config(
        page_title="Weld Sim",
        page_icon="",
        layout="wide",
    )

    st.title(" Weld Sim — 2D Thermal + 3D Keyhole CFD")

    page = st.sidebar.radio(
        "Select view",
        ["2D Thermal Sim", "3D Keyhole CFD"],
    )

    if page == "2D Thermal Sim":
        page_thermal_2d()
    elif page == "3D Keyhole CFD":
        page_keyhole_cfd()


if __name__ == "__main__":
    main()
