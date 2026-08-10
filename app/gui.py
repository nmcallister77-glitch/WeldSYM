"""Streamlit GUI for Weld Sim."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from weldsim.simulation import (
    WeldParams,
    MaterialParams,
    ThermalSimulationConfig,
    run_thermal_simulation,
)


def plot_temperature(x: np.ndarray, y: np.ndarray, T: np.ndarray) -> plt.Figure:
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots()
    cmap = ax.pcolormesh(X * 1e3, Y * 1e3, T, shading="auto", cmap="inferno")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("Temperature field (K)")
    fig.colorbar(cmap, ax=ax, label="Temperature (K)")
    return fig


def main():
    st.set_page_config(page_title="Weld Sim", page_icon="🔥")

    st.title("🔥 Weld Sim – 2D Welding Thermal Simulation")
    st.write(
        "Set welding parameters, run a 2D transient thermal simulation, and view the temperature field."
    )

    st.sidebar.header("Welding parameters")

    power = st.sidebar.slider("Power (W)", 500.0, 5000.0, 1500.0, 50.0)
    efficiency = st.sidebar.slider("Efficiency", 0.3, 1.0, 0.8, 0.05)
    speed = st.sidebar.slider("Travel speed (m/s)", 0.001, 0.02, 0.005, 0.001)

    st.sidebar.header("Geometry & mesh")
    Lx = st.sidebar.number_input("Plate length Lx (m)", 0.05, 0.5, 0.1, 0.01)
    Ly = st.sidebar.number_input("Plate width Ly (m)", 0.02, 0.2, 0.05, 0.01)
    nx = st.sidebar.slider("Grid points in X", 21, 101, 41, 2)
    ny = st.sidebar.slider("Grid points in Y", 11, 51, 21, 2)

    st.sidebar.header("Time settings")
    t_end = st.sidebar.number_input("Simulation time (s)", 1.0, 100.0, 5.0, 1.0)
    dt = st.sidebar.number_input("Time step (s)", 0.001, 1.0, 0.05, 0.005)

    if st.sidebar.button("Run simulation"):
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
            output_file=None,  # we don't save CSV here
        )

        with st.spinner("Running simulation..."):
            result = run_thermal_simulation(config)

        x, y, T = result["x"], result["y"], result["T"]
        T_min, T_max = float(T.min()), float(T.max())

        st.write(f"**Temperature range:** {T_min:.1f} K – {T_max:.1f} K")

        fig = plot_temperature(x, y, T)
        st.pyplot(fig)

        # Optional: offer CSV download
        import csv
        import io

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


if __name__ == "__main__":
    main()
