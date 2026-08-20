"""Interactive Plotly charts for the Weld Sim Streamlit GUI.

All functions return ``plotly.graph_objects.Figure`` objects that
``st.plotly_chart`` will render with pan, box-zoom and reset-axes tools
already in the mode bar.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from weldsim.calibration import Comparison
from weldsim.materials import Material
from weldsim.thermal.solver3d import Solution3D
from weldsim.report import WeldReport
from weldsim.weld_path import WeldPath, WobbleParams, beam_at_time, beam_trajectory


def _axis_mm(title: str) -> dict[str, Any]:
    return dict(title=title, gridcolor="rgba(0,0,0,0.1)", zeroline=False)


def _default_layout(title: str | None = None) -> dict[str, Any]:
    layout: dict[str, Any] = dict(
        margin=dict(l=60, r=20, t=50, b=50),
        paper_bgcolor="white",
        plot_bgcolor="white",
        hovermode="closest",
        dragmode="zoom",
    )
    if title:
        layout["title"] = dict(text=title, x=0.5, xanchor="center")
    return layout


def _log_colorbar(z: np.ndarray, colorbar_title: str) -> dict[str, Any]:
    """Build a colorbar with labelled log10 ticks for a log-scaled field."""
    finite = z[np.isfinite(z) & (z > 0)]
    if finite.size == 0:
        return dict(title=colorbar_title)
    lo, hi = np.log10(finite.min()), np.log10(finite.max())
    # Pick 5-6 round log ticks
    ticks = np.linspace(lo, hi, 6)
    ticktext = [f"{10 ** t:.2e}" for t in ticks]
    return dict(
        title=colorbar_title,
        tickvals=ticks,
        ticktext=ticktext,
    )


def _zone_levels_and_labels(
    T_peak: np.ndarray, material: Material
) -> tuple[list[float], list[str], list[str]]:
    """Categorical temperature levels and labels for the zone map."""
    levels: list[float] = [float(T_peak.min()), float(material.haz_outer_temperature)]
    labels: list[str] = ["Unaffected"]
    colors: list[str] = ["#301f1f"]
    for zone in sorted(material.haz_zones, key=lambda z: z.t_min):
        upper = min(zone.t_max, material.solidus)
        if upper > levels[-1]:
            levels.append(float(upper))
            labels.append(zone.name)
            colors.append("#8B4513")  # generic HAZ brown
    if material.solidus > levels[-1]:
        levels.append(float(material.solidus))
        labels.append("HAZ" if not material.haz_zones else "Near fusion boundary")
        colors.append("#CD853F")
    levels.append(max(float(material.liquidus), float(T_peak.max()) + 1.0))
    labels.append("Fusion zone")
    colors.append("#FF4500")
    return levels, labels, colors


def _zone_index_map(
    T_peak: np.ndarray, material: Material
) -> tuple[np.ndarray, list[str], list[str]]:
    """Map each cell to a zone index and build the matching colorscale."""
    levels, labels, colors = _zone_levels_and_labels(T_peak, material)
    # levels are upper bounds; digitize gives the zone index for each T.
    zone = np.digitize(T_peak, np.asarray(levels[1:]), right=False)
    zone = np.clip(zone, 0, len(labels) - 1)

    n = len(labels)
    # Build a piecewise colorscale centered on each integer zone index.
    colorscale: list[list[float | str]] = []
    for i, color in enumerate(colors):
        lo = (i + 0.5) / n
        hi = (i + 1.5) / n if i < n - 1 else 1.0
        colorscale.append([lo, color])
        if hi < 1.0:
            colorscale.append([hi, color])
    return zone, labels, colorscale


def plot_temperature_2d(x: np.ndarray, y: np.ndarray, T: np.ndarray) -> go.Figure:
    """Interactive 2D temperature heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            x=x * 1e3,
            y=y * 1e3,
            z=T.T,
            colorscale="Inferno",
            colorbar=dict(title="T (K)"),
            hovertemplate="x %{x:.2f} mm<br>y %{y:.2f} mm<br>T %{z:.0f} K<extra></extra>",
        )
    )
    fig.update_layout(
        **_default_layout("Temperature field (K)"),
        xaxis=_axis_mm("x (mm)"),
        yaxis=_axis_mm("y (mm)"),
    )
    return fig


def plot_temperature_surface(x: np.ndarray, y: np.ndarray, T: np.ndarray) -> go.Figure:
    """3D surface of the final 2D temperature field (spinnable)."""
    X, Y = np.meshgrid(x * 1e3, y * 1e3, indexing="ij")
    fig = go.Figure(
        data=go.Surface(
            x=X,
            y=Y,
            z=T,
            colorscale="Inferno",
            colorbar=dict(title="T (K)"),
            hovertemplate="x %{x:.2f} mm<br>y %{y:.2f} mm<br>T %{z:.0f} K<extra></extra>",
        )
    )
    fig.update_layout(
        **_default_layout("3D temperature surface"),
        scene=dict(
            xaxis=dict(title="x (mm)"),
            yaxis=dict(title="y (mm)"),
            zaxis=dict(title="T (K)"),
            aspectmode="data",
            dragmode="orbit",
        ),
    )
    return fig


def _plate_traces(
    x: np.ndarray,
    y: np.ndarray,
    thickness: float,
    top_thickness: float | None = None,
) -> list[Any]:
    """Wireframe + mesh that make a 2t lap joint look like two stacked plates."""
    x0, x1 = float(x[0]) * 1e3, float(x[-1]) * 1e3
    y0, y1 = float(y[0]) * 1e3, float(y[-1]) * 1e3
    z_top = 0.0
    z_bot = float(thickness) * 1e3

    def _box_vertices(z0: float, z1: float) -> dict[str, list[float]]:
        return {
            "x": [x0, x1, x1, x0, x0, x1, x1, x0],
            "y": [y0, y0, y1, y1, y0, y0, y1, y1],
            "z": [z0, z0, z0, z0, z1, z1, z1, z1],
        }

    def _box_faces() -> tuple[list[int], list[int], list[int]]:
        # 12 triangles for a box
        faces = [
            (0, 1, 2),
            (0, 2, 3),  # bottom
            (4, 6, 5),
            (4, 7, 6),  # top
            (0, 4, 5),
            (0, 5, 1),  # side y0
            (1, 5, 6),
            (1, 6, 2),  # side x1
            (2, 6, 7),
            (2, 7, 3),  # side y1
            (3, 7, 4),
            (3, 4, 0),  # side x0
        ]
        i = [f[0] for f in faces]
        j = [f[1] for f in faces]
        k = [f[2] for f in faces]
        return i, j, k

    def _box_wireframe(z0: float, z1: float) -> tuple[list[float], list[float], list[float]]:
        xs, ys, zs = [], [], []
        for z in (z0, z1):
            xs.extend([x0, x1, x1, x0, x0, None])
            ys.extend([y0, y0, y1, y1, y0, None])
            zs.extend([z, z, z, z, z, None])
        for xi in (x0, x1):
            for yi in (y0, y1):
                xs.extend([xi, xi, None])
                ys.extend([yi, yi, None])
                zs.extend([z0, z1, None])
        return xs, ys, zs

    i, j, k = _box_faces()
    traces: list[Any] = []

    if top_thickness is not None and 0.0 < top_thickness < thickness:
        z_if = float(top_thickness) * 1e3
        Xs, Ys = np.meshgrid(x * 1e3, y * 1e3, indexing="ij")

        # Translucent volumes so the two plates are visually distinct.
        top_box = _box_vertices(z_top, z_if)
        bot_box = _box_vertices(z_if, z_bot)
        traces.append(
            go.Mesh3d(
                x=top_box["x"],
                y=top_box["y"],
                z=top_box["z"],
                i=i,
                j=j,
                k=k,
                color="deepskyblue",
                opacity=0.08,
                name="Top sheet",
                hoverinfo="skip",
                flatshading=True,
            )
        )
        traces.append(
            go.Mesh3d(
                x=bot_box["x"],
                y=bot_box["y"],
                z=bot_box["z"],
                i=i,
                j=j,
                k=k,
                color="orangered",
                opacity=0.08,
                name="Bottom sheet",
                hoverinfo="skip",
                flatshading=True,
            )
        )

        traces.append(
            go.Surface(
                x=Xs,
                y=Ys,
                z=np.full(Xs.shape, z_if),
                colorscale=[[0, "gold"], [1, "gold"]],
                showscale=False,
                opacity=0.25,
                name="Lap joint interface",
                hoverinfo="skip",
            )
        )

        # Wireframes on top of the volumes.
        for label, z0, z1, color in (
            ("Top sheet", z_top, z_if, "deepskyblue"),
            ("Bottom sheet", z_if, z_bot, "orangered"),
        ):
            xs, ys, zs = _box_wireframe(z0, z1)
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    line=dict(color=color, width=2),
                    name=label,
                    hoverinfo="skip",
                )
            )

        # Interface loop in a contrasting, thicker line.
        traces.append(
            go.Scatter3d(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                z=[z_if, z_if, z_if, z_if, z_if],
                mode="lines",
                line=dict(color="gold", width=3),
                name="Interface loop",
                hoverinfo="skip",
            )
        )
    else:
        box = _box_vertices(z_top, z_bot)
        traces.append(
            go.Mesh3d(
                x=box["x"],
                y=box["y"],
                z=box["z"],
                i=i,
                j=j,
                k=k,
                color="lightgrey",
                opacity=0.06,
                name="Plate volume",
                hoverinfo="skip",
                flatshading=True,
            )
        )
        xs, ys, zs = _box_wireframe(z_top, z_bot)
        traces.append(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="lines",
                line=dict(color="lightslategrey", width=1),
                name="Plate edges",
                hoverinfo="skip",
            )
        )
    return traces


def plot_weld_3d(
    solution: Solution3D,
    material: Material,
    path: WeldPath | None = None,
    wobble: WobbleParams | None = None,
    top_thickness: float | None = None,
) -> go.Figure:
    """Spinnable 3D viewport showing the full weld (fusion + HAZ isosurfaces)."""
    X, Y, Z = np.meshgrid(solution.x, solution.y, solution.z, indexing="ij")
    x = X.ravel() * 1e3
    y = Y.ravel() * 1e3
    z = Z.ravel() * 1e3
    values = solution.T_peak.ravel()

    traces: list[Any] = []

    # Fusion-zone isosurface
    traces.append(
        go.Isosurface(
            x=x,
            y=y,
            z=z,
            value=values,
            isomin=float(solution.solidus),
            isomax=float(solution.solidus),
            surface_count=1,
            colorscale=[[0, "#ff3333"], [1, "#ff3333"]],
            showscale=False,
            name="Fusion zone",
            opacity=0.85,
            hovertemplate=(
                "x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<br>fusion boundary<extra></extra>"
            ),
        )
    )

    # HAZ isosurface
    traces.append(
        go.Isosurface(
            x=x,
            y=y,
            z=z,
            value=values,
            isomin=float(solution.haz_limit),
            isomax=float(solution.haz_limit),
            surface_count=1,
            colorscale=[[0, "#ffaa00"], [1, "#ffaa00"]],
            showscale=False,
            name="HAZ limit",
            opacity=0.35,
            hovertemplate="x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<br>HAZ limit<extra></extra>",
        )
    )

    # Top/bottom sheet interface plane for 2t lap joints
    traces.extend(_plate_traces(solution.x, solution.y, solution.thickness, top_thickness))

    # Beam path (top surface, z=0)
    if path is not None:
        t_end = path.duration
        if t_end > 0:
            try:
                x_traj, y_traj = beam_trajectory(
                    path,
                    wobble or WobbleParams(amplitude=0.0, frequency=0.0),
                    t_end=t_end,
                    dt=min(0.001, t_end / 500),
                )
                traces.append(
                    go.Scatter3d(
                        x=x_traj * 1e3,
                        y=y_traj * 1e3,
                        z=np.zeros_like(x_traj),
                        mode="lines",
                        line=dict(color="cyan", width=3),
                        name="Beam track",
                        hoverinfo="skip",
                    )
                )
            except Exception:
                pass

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_default_layout("3D weld viewport"),
        scene=dict(
            xaxis=dict(title="x (mm)"),
            yaxis=dict(title="y (mm)"),
            zaxis=dict(title="z (mm)", autorange="reversed"),
            aspectmode="data",
            dragmode="orbit",
        ),
    )
    return fig


def plot_weld_3d_animation(
    solution: Solution3D,
    material: Material,
    path: WeldPath | None = None,
    wobble: WobbleParams | None = None,
    top_thickness: float | None = None,
    time_scale: float = 1.0,
    max_frames: int = 120,
) -> go.Figure:
    """Animated 3D weld pool with fusion and HAZ isosurfaces and time scaling."""
    if not solution.T_history:
        fig = go.Figure()
        fig.add_annotation(
            text="No 3D animation frames were stored for this run.",
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    # Time scaling: drop frames for fast forward, stretch frames for slow motion.
    n_frames = len(solution.T_history)
    if n_frames > max_frames:
        stride = max(1, n_frames // max_frames)
        if n_frames // stride > max_frames:
            stride += 1
        frame_times = solution.frame_times[::stride]
        T_history = solution.T_history[::stride]
    else:
        frame_times = list(solution.frame_times)
        T_history = list(solution.T_history)

    if len(frame_times) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Not enough animation frames for a 3D playback.",
            showarrow=False,
            font=dict(size=14),
        )
        return fig

    frame_dt = float(frame_times[1] - frame_times[0])
    if time_scale >= 1.0:
        play_stride = max(1, int(round(time_scale)))
        frame_ms = max(20, int(play_stride * frame_dt * 1000.0 / time_scale))
    else:
        play_stride = 1
        frame_ms = max(20, int(frame_dt * 1000.0 / time_scale))

    if play_stride > 1:
        frame_times = frame_times[::play_stride]
        T_history = T_history[::play_stride]

    total_time = float(frame_times[-1])
    playback_time = total_time / time_scale

    x = np.asarray(solution.x) * 1e3
    y = np.asarray(solution.y) * 1e3
    z = np.asarray(solution.z) * 1e3
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    x_all = X.ravel()
    y_all = Y.ravel()
    z_all = Z.ravel()

    solidus = float(solution.solidus)
    haz_limit = float(solution.haz_limit)

    fusion_kwargs = dict(
        x=x_all,
        y=y_all,
        z=z_all,
        isomin=solidus,
        isomax=solidus,
        surface_count=1,
        colorscale=[[0, "#ff3333"], [1, "#ff3333"]],
        showscale=False,
        name="Fusion zone",
        opacity=0.9,
        caps=dict(x_show=False, y_show=False, z_show=True),
        hovertemplate="x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<br>fusion<extra></extra>",
    )
    haz_kwargs = dict(
        x=x_all,
        y=y_all,
        z=z_all,
        isomin=haz_limit,
        isomax=haz_limit,
        surface_count=1,
        colorscale=[[0, "#ffaa00"], [1, "#ffaa00"]],
        showscale=False,
        name="HAZ",
        opacity=0.18,
        caps=dict(x_show=False, y_show=False, z_show=False),
        hovertemplate="x %{x:.2f}<br>y %{y:.2f}<br>z %{z:.2f}<br>HAZ<extra></extra>",
    )
    beam_kwargs = dict(
        mode="markers",
        marker=dict(size=5, color="cyan", symbol="diamond"),
        name="beam",
        hoverinfo="skip",
    )

    base_traces: list[Any] = []
    base_traces.extend(
        _plate_traces(
            np.asarray(solution.x),
            np.asarray(solution.y),
            solution.thickness,
            top_thickness,
        )
    )
    n_plate = len(base_traces)
    base_traces.append(go.Isosurface(value=T_history[0].ravel(), **fusion_kwargs))
    base_traces.append(go.Isosurface(value=T_history[0].ravel(), **haz_kwargs))
    base_traces.append(go.Scatter3d(x=[], y=[], z=[], **beam_kwargs))

    dynamic_indices = [n_plate, n_plate + 1, n_plate + 2]

    frames: list[go.Frame] = []
    for t, T in zip(frame_times, T_history):
        T_r = T.ravel()
        if path is not None:
            t_clip = float(min(t, path.duration))
            xb, yb = beam_at_time(path, wobble or WobbleParams(0.0, 0.0, "circle"), t_clip)
            xb_mm, yb_mm = float(xb * 1e3), float(yb * 1e3)
        else:
            xb_mm, yb_mm = float("nan"), float("nan")
        frames.append(
            go.Frame(
                name=f"t={t:.3f}",
                traces=dynamic_indices,
                data=[
                    go.Isosurface(value=T_r, **fusion_kwargs),
                    go.Isosurface(value=T_r, **haz_kwargs),
                    go.Scatter3d(x=[xb_mm], y=[yb_mm], z=[0.0], **beam_kwargs),
                ],
            )
        )

    slider_steps = [
        {
            "args": [
                [f"t={t:.3f}"],
                {
                    "frame": {"duration": 0, "redraw": True},
                    "mode": "immediate",
                    "transition": {"duration": 0},
                },
            ],
            "label": f"{t:.2f}",
            "method": "animate",
        }
        for t in frame_times
    ]

    fig = go.Figure(data=base_traces, frames=frames)
    fig.update_layout(
        **_default_layout(f"3D weld animation — {playback_time:.2f}s playback ({time_scale:.1f}x)"),
        scene=dict(
            xaxis=dict(title="x (mm)"),
            yaxis=dict(title="y (mm)"),
            zaxis=dict(title="z (mm)", autorange="reversed"),
            aspectmode="data",
            dragmode="orbit",
        ),
        updatemenus=[
            {
                "type": "buttons",
                "direction": "left",
                "x": 0.1,
                "y": 1.15,
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Go",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": frame_ms,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": frame_ms,
                                    "easing": "linear",
                                },
                                "fromcurrent": True,
                                "mode": "immediate",
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "yanchor": "top",
                "xanchor": "left",
                "x": 0.15,
                "y": -0.05,
                "steps": slider_steps,
            }
        ],
    )
    return fig


def plot_heat_signature(
    x: np.ndarray,
    y: np.ndarray,
    Q: np.ndarray,
    x_traj: np.ndarray | None = None,
    y_traj: np.ndarray | None = None,
) -> go.Figure:
    """Log-scaled heat signature with optional beam path overlay."""
    Qplot = Q.copy()
    pos = Qplot[Qplot > 0]
    if pos.size:
        Qplot[Qplot == 0] = pos.min() * 1e-6
    logQ = np.log10(Qplot)

    traces: list[Any] = [
        go.Heatmap(
            x=x * 1e3,
            y=y * 1e3,
            z=logQ.T,
            colorscale="Hot",
            colorbar=_log_colorbar(Qplot, "Q (J/m³)"),
            hovertemplate=(
                "x %{x:.2f} mm<br>y %{y:.2f} mm<br>Q %{customdata:.2e} J/m³<extra></extra>"
            ),
            customdata=Qplot.T,
        )
    ]

    if x_traj is not None and y_traj is not None:
        traces.append(
            go.Scatter(
                x=x_traj * 1e3,
                y=y_traj * 1e3,
                mode="lines",
                line=dict(color="cyan", width=1),
                name="beam path",
                opacity=0.5,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        **_default_layout("Heat signature (J/m³) — log scale"),
        xaxis=_axis_mm("x (mm)"),
        yaxis=_axis_mm("y (mm)"),
    )
    return fig


def plot_temperature_profile(
    x: np.ndarray,
    T: np.ndarray,
    xlabel: str,
    title: str,
    material: Material,
) -> go.Figure:
    """Line profile with solidus/liquidus reference lines."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x * 1e3,
            y=T,
            mode="lines",
            line=dict(color="black", width=1.5),
            name="temperature",
        )
    )
    for value, name, color, dash in [
        (material.solidus, "solidus", "orange", "dash"),
        (material.liquidus, "liquidus", "red", "dash"),
    ]:
        fig.add_hline(y=value, line_dash=dash, line_color=color, annotation_text=name)
    fig.update_layout(
        **_default_layout(title),
        xaxis=_axis_mm(xlabel),
        yaxis=dict(title="T (K)", gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def plot_probe_history(t: np.ndarray, T_probe: np.ndarray, material: Material) -> go.Figure:
    """Probe time-temperature history."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=t,
            y=T_probe,
            mode="lines",
            line=dict(color="blue", width=1.5),
            name="probe T",
        )
    )
    fig.add_hline(
        y=material.solidus, line_dash="dash", line_color="orange", annotation_text="solidus"
    )
    fig.add_hline(
        y=material.liquidus, line_dash="dash", line_color="red", annotation_text="liquidus"
    )
    fig.update_layout(
        **_default_layout("Time-temperature history at probe"),
        xaxis=dict(title="Time (s)", gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(title="T (K)", gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def plot_beam_path(
    x_traj: np.ndarray,
    y_traj: np.ndarray,
    Lx: float,
    Ly: float,
) -> go.Figure:
    """Plan-view wobbled beam path with start/end markers."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_traj * 1e3,
            y=y_traj * 1e3,
            mode="lines",
            line=dict(color="red", width=1),
            opacity=0.7,
            name="path",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_traj[0] * 1e3],
            y=[y_traj[0] * 1e3],
            mode="markers",
            marker=dict(color="green", size=8),
            name="start",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[x_traj[-1] * 1e3],
            y=[y_traj[-1] * 1e3],
            mode="markers",
            marker=dict(color="blue", size=8),
            name="end",
        )
    )
    fig.update_layout(
        **_default_layout("Wobbled beam path"),
        xaxis=dict(title="x (mm)", range=[0, Lx * 1e3], scaleanchor="y"),
        yaxis=dict(title="y (mm)", range=[0, Ly * 1e3], scaleratio=1),
    )
    return fig


def plot_zone_map(
    x: np.ndarray,
    y: np.ndarray,
    T_peak: np.ndarray,
    material: Material,
) -> go.Figure:
    """Categorical plan-view zone map with fusion and HAZ boundary lines."""
    zone, labels, colorscale = _zone_index_map(T_peak, material)
    n = len(labels)

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=x * 1e3,
            y=y * 1e3,
            z=zone.T,
            zmin=-0.5,
            zmax=n - 0.5,
            colorscale=colorscale,
            colorbar=dict(
                title="Zone",
                tickvals=list(range(n)),
                ticktext=labels,
                ticks="",
            ),
            hovertemplate="x %{x:.2f} mm<br>y %{y:.2f} mm<br>T %{customdata:.0f} K<extra></extra>",
            customdata=T_peak.T,
            showscale=True,
        )
    )
    # Fusion boundary
    fig.add_trace(
        go.Contour(
            x=x * 1e3,
            y=y * 1e3,
            z=T_peak.T,
            contours=dict(start=material.solidus, end=material.solidus, size=1, coloring="lines"),
            line=dict(color="cyan", width=2),
            showscale=False,
            hoverinfo="skip",
            name="fusion boundary",
        )
    )
    # HAZ limit
    fig.add_trace(
        go.Contour(
            x=x * 1e3,
            y=y * 1e3,
            z=T_peak.T,
            contours=dict(
                start=material.haz_outer_temperature,
                end=material.haz_outer_temperature,
                size=1,
                coloring="lines",
            ),
            line=dict(color="lime", width=1.5),
            showscale=False,
            hoverinfo="skip",
            name="HAZ limit",
        )
    )
    fig.update_layout(
        **_default_layout("Weld zones from peak temperature"),
        xaxis=_axis_mm("x (mm)"),
        yaxis=_axis_mm("y (mm)"),
    )
    return fig


def _make_contour_figure(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    xlabel: str,
    ylabel: str,
    title: str,
    solidus: float,
    haz_limit: float,
    colorbar_title: str = "Peak T (K)",
) -> go.Figure:
    """Shared 2D filled-contour layout for cross/longitudinal sections."""
    T_min, T_max = float(z.min()), float(z.max())
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=z,
            colorscale="Inferno",
            colorbar=dict(title=colorbar_title),
            contours=dict(
                start=T_min,
                end=T_max,
                size=max((T_max - T_min) / 24, 1.0),
                coloring="fill",
            ),
            hovertemplate="%{x:.2f} mm<br>%{y:.2f} mm<br>T %{z:.0f} K<extra></extra>",
            showscale=True,
        )
    )
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=z,
            contours=dict(start=solidus, end=solidus, size=1, coloring="lines"),
            line=dict(color="cyan", width=2),
            showscale=False,
            hoverinfo="skip",
            name="fusion boundary",
        )
    )
    fig.add_trace(
        go.Contour(
            x=x,
            y=y,
            z=z,
            contours=dict(start=haz_limit, end=haz_limit, size=1, coloring="lines"),
            line=dict(color="lime", width=1.5),
            showscale=False,
            hoverinfo="skip",
            name="HAZ limit",
        )
    )
    fig.update_layout(
        **_default_layout(title),
        xaxis=_axis_mm(xlabel),
        yaxis=dict(
            title=ylabel, autorange="reversed", gridcolor="rgba(0,0,0,0.1)", scaleanchor="x"
        ),
    )
    return fig


def plot_cross_section(solution: Solution3D) -> go.Figure:
    """Transverse macro-section with fusion and HAZ boundaries."""
    index = solution.section_index()
    T_peak = solution.section(index)
    return _make_contour_figure(
        x=solution.y * 1e3,
        y=solution.z * 1e3,
        z=T_peak.T,
        xlabel="y across the weld (mm)",
        ylabel="depth below the surface (mm)",
        title=f"Cross-section at x = {solution.x[index] * 1e3:.1f} mm",
        solidus=solution.solidus,
        haz_limit=solution.haz_limit,
    )


def plot_longitudinal_section(solution: Solution3D) -> go.Figure:
    """Longitudinal section through the weld centreline."""
    centre = int(np.argmax(solution.T_peak.max(axis=(0, 2))))
    T_peak = solution.T_peak[:, centre, :]
    return _make_contour_figure(
        x=solution.x * 1e3,
        y=solution.z * 1e3,
        z=T_peak.T,
        xlabel="x along the weld (mm)",
        ylabel="depth (mm)",
        title=f"Longitudinal section at y = {solution.y[centre] * 1e3:.1f} mm",
        solidus=solution.solidus,
        haz_limit=solution.haz_limit,
    )


def plot_macro_section(report: WeldReport, material: Material) -> go.Figure:
    """Transverse peak-temperature profile with solidus/liquidus/HAZ limits and fusion fill."""
    profile = report.metrics.profile
    y_mm = profile.y * 1e3
    T = profile.T_peak

    fig = go.Figure()
    # Base line at solidus for the fill
    fig.add_trace(
        go.Scatter(
            x=y_mm,
            y=np.full_like(y_mm, material.solidus),
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    # Fill above solidus
    y_fill = np.where(T >= material.solidus, T, material.solidus)
    fig.add_trace(
        go.Scatter(
            x=y_mm,
            y=y_fill,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 0, 0, 0.2)",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_mm,
            y=T,
            mode="lines",
            line=dict(color="black", width=1.5),
            name="peak temperature",
        )
    )
    fig.add_hline(
        y=material.liquidus, line_dash="dash", line_color="red", annotation_text="liquidus"
    )
    fig.add_hline(
        y=material.solidus, line_dash="dash", line_color="orange", annotation_text="solidus"
    )
    fig.add_hline(
        y=material.haz_outer_temperature,
        line_dash="dash",
        line_color="green",
        annotation_text="HAZ limit",
    )
    fig.update_layout(
        **_default_layout(f"Macro-section at x = {profile.x * 1e3:.1f} mm"),
        xaxis=_axis_mm("y across the weld (mm)"),
        yaxis=dict(title="Peak T (K)", gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def plot_phases(phases: dict) -> go.Figure:
    """Horizontal bar chart of predicted HAZ phase fractions."""
    names = list(phases)
    values = [phases[n] * 100 for n in names]
    fig = go.Figure(
        go.Bar(
            x=values,
            y=names,
            orientation="h",
            marker_color="steelblue",
            text=[f"{v:.0f}%" for v in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        **_default_layout("Predicted HAZ constitution"),
        xaxis=dict(title="Volume fraction (%)", range=[0, 100], gridcolor="rgba(0,0,0,0.1)"),
        yaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def plot_energy_density(x: np.ndarray, y: np.ndarray, E: np.ndarray) -> go.Figure:
    """Absorbed energy density heatmap."""
    fig = go.Figure(
        data=go.Heatmap(
            x=x * 1e3,
            y=y * 1e3,
            z=(E * 1e-6).T,
            colorscale="Hot",
            colorbar=dict(title="MJ/m²"),
            hovertemplate="x %{x:.2f} mm<br>y %{y:.2f} mm<br>E %{z:.2f} MJ/m²<extra></extra>",
        )
    )
    fig.update_layout(
        **_default_layout("Absorbed energy density (MJ/m²)"),
        xaxis=_axis_mm("x (mm)"),
        yaxis=_axis_mm("y (mm)"),
    )
    return fig


def plot_wobble_concentration(y: np.ndarray, E: np.ndarray) -> go.Figure:
    """Energy density across the wobble track."""
    station = int(np.argmax(E.max(axis=1)))
    y_mm = y * 1e3
    E_line = E[station, :] * 1e-6
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_mm,
            y=E_line,
            mode="lines",
            line=dict(color="red", width=1.5),
            fill="tozeroy",
            fillcolor="rgba(255, 0, 0, 0.15)",
            name="energy density",
        )
    )
    fig.update_layout(
        **_default_layout("Heat concentration across the wobble track"),
        xaxis=_axis_mm("y across the weld (mm)"),
        yaxis=dict(title="Energy density (MJ/m²)", gridcolor="rgba(0,0,0,0.1)"),
    )
    return fig


def plot_measured_vs_predicted(comparison: Comparison) -> go.Figure:
    """Parity plot for measured versus predicted penetration."""
    measured = [r.measured_penetration * 1e3 for r in comparison.residuals]
    predicted = [r.predicted_penetration * 1e3 for r in comparison.residuals]
    limit = max(measured + predicted + [0.1]) * 1.15

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, limit],
            mode="lines",
            line=dict(color="black", dash="dash", width=1),
            name="perfect agreement",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, 1.2 * limit],
            mode="lines",
            line=dict(color="gray", width=0.8),
            name="+20%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[0, limit],
            y=[0, 0.8 * limit],
            mode="lines",
            line=dict(color="gray", width=0.8),
            name="−20%",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=measured,
            y=predicted,
            mode="markers+text",
            marker=dict(color="tab:red", size=10),
            text=[r.label for r in comparison.residuals],
            textposition="top right",
            textfont=dict(size=8),
            name="coupons",
        )
    )
    fig.update_layout(
        **_default_layout("Macro-section parity"),
        xaxis=dict(
            title="Measured penetration (mm)",
            range=[0, limit],
            gridcolor="rgba(0,0,0,0.1)",
            scaleanchor="y",
        ),
        yaxis=dict(
            title="Predicted penetration (mm)",
            range=[0, limit],
            gridcolor="rgba(0,0,0,0.1)",
            scaleratio=1,
        ),
    )
    return fig
