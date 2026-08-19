"""Tests for the built-in 3D solver and its weld measurements."""

import numpy as np
import pytest

from weldsim.errors import StabilityError, ValidationError
from weldsim.materials import load_material
from weldsim.report import build_report
from weldsim.simulation import ThermalSimulationConfig, run_thermal_simulation
from weldsim.thermal.fd_solver import PhaseModel
from weldsim.thermal.solver3d import stable_dt, keyhole_power_fraction, run_3d_thermal
from weldsim.types import WeldParams
from weldsim.weld_path import WeldPath, WobbleParams

STEEL = load_material("S355_structural_steel")


def phase_of(material=STEEL) -> PhaseModel:
    return PhaseModel(
        solidus=material.solidus,
        liquidus=material.liquidus,
        latent_heat=material.latent_heat_fusion,
        boiling=material.boiling,
    )


def weld(power=3000.0, sigma=2e-4, speed=0.02) -> WeldParams:
    return WeldParams(
        power=power,
        efficiency=0.7,
        speed=speed,
        start_pos=(0.004, 0.006),
        direction="x",
        sigma=sigma,
    )


def path(speed=0.02) -> WeldPath:
    return WeldPath(start=(0.004, 0.006), end=(0.026, 0.006), speed=speed)


def solve(**overrides):
    kwargs = dict(
        nx=41,
        ny=31,
        nz=11,
        Lx=0.03,
        Ly=0.012,
        thickness=0.004,
        t_end=0.4,
        weld=weld(),
        material=STEEL,
        path=path(),
        phase=phase_of(),
    )
    kwargs.update(overrides)
    return run_3d_thermal(**kwargs)


def test_grid_and_field_shapes_follow_ix_iy_iz():
    sol = solve(t_end=0.05)
    assert sol.T.shape == (41, 31, 11)
    assert sol.T_peak.shape == sol.T.shape
    assert sol.x[0] == 0.0 and sol.x[-1] == pytest.approx(0.03)
    assert sol.z[0] == 0.0 and sol.z[-1] == pytest.approx(0.004)
    assert sol.thickness == pytest.approx(0.004)
    assert sol.section(0).shape == (31, 11)


def test_automatic_time_step_is_stable_and_bounded():
    sol = solve(t_end=0.05)
    alpha = STEEL.thermal_diffusivity
    dx, dy, dz = 0.03 / 40, 0.012 / 30, 0.004 / 10
    assert sol.dt < stable_dt(alpha, dx, dy, dz)
    assert sol.steps == int(np.ceil(0.05 / sol.dt))
    assert np.isfinite(sol.T).all()


def test_time_step_above_the_stability_limit_is_rejected():
    with pytest.raises(StabilityError, match="Unstable time step"):
        solve(dt=1.0)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"nz": 2}, "at least 3 points"),
        ({"thickness": 0.0}, "thickness must be positive"),
        ({"t_end": -1.0}, "time must be positive"),
    ],
)
def test_invalid_geometry_is_rejected(kwargs, message):
    with pytest.raises(ValidationError, match=message):
        solve(**kwargs)


def test_a_run_that_would_take_minutes_is_refused_up_front():
    with pytest.raises(ValidationError, match="billion cell updates"):
        solve(nx=201, ny=201, nz=61, t_end=30.0)


def test_energy_is_conserved_without_surface_losses():
    """All absorbed power must end up in the plate, source terms included."""
    sol = solve(
        t_end=0.05,
        phase=None,
        emissivity=0.0,
        convection_coefficient=0.0,
        T0=300.0,
    )
    dx = sol.x[1] - sol.x[0]
    dy = sol.y[1] - sol.y[0]
    dz = sol.z[1] - sol.z[0]
    layer = np.ones(sol.z.size)
    layer[0] = layer[-1] = 0.5  # face nodes own half a cell
    stored = (
        STEEL.density * STEEL.specific_heat * (sol.T - 300.0) * layer[None, None, :]
    ).sum() * (dx * dy * dz)
    supplied = weld().absorbed_power * sol.steps * sol.dt
    assert stored == pytest.approx(supplied, rel=1e-3)


def test_keyhole_power_fraction_spans_the_transition_band():
    assert keyhole_power_fraction(1e8) == 0.0
    assert keyhole_power_fraction(1e12) == 1.0
    mid = keyhole_power_fraction(6e9)
    assert 0.0 < mid < 1.0


def test_keyhole_beam_penetrates_deeper_than_a_defocused_one():
    """A focused beam should key-hole; spreading the same power should not."""
    focused = solve(weld=weld(sigma=1.5e-4))
    defocused = solve(weld=weld(sigma=1.2e-3))

    assert focused.keyhole_fraction > 0.5
    assert defocused.keyhole_fraction == 0.0
    deep = focused.section_metrics()
    shallow = defocused.section_metrics()
    assert deep.penetration > shallow.penetration
    assert deep.aspect_ratio > shallow.aspect_ratio


def test_wobble_trades_penetration_for_width():
    """Spreading the spot drops the intensity out of the keyhole regime."""
    straight = solve(weld=weld(sigma=1.5e-4))
    wobbled = solve(
        weld=weld(sigma=1.5e-4),
        wobble=WobbleParams(amplitude=6e-4, frequency=150.0, pattern="circle"),
    )
    assert straight.keyhole_fraction > wobbled.keyhole_fraction

    narrow_deep = straight.section_metrics()
    wide_shallow = wobbled.section_metrics()
    assert wide_shallow.width_top > narrow_deep.width_top
    assert wide_shallow.penetration < narrow_deep.penetration


def test_wobble_faster_than_the_time_step_warns_but_still_runs():
    sol = solve(wobble=WobbleParams(amplitude=4e-4, frequency=5000.0, pattern="circle"))
    assert any("faster than the time step" in w for w in sol.warnings)
    assert np.isfinite(sol.T_peak).all()


def test_boiling_cap_holds_and_is_reported():
    sol = solve(weld=weld(power=6000.0, sigma=1.5e-4))
    assert sol.T_peak.max() <= STEEL.boiling + 1e-6
    assert any("boiling point" in w for w in sol.warnings)


def test_section_is_taken_mid_weld_not_at_the_stop_crater():
    sol = solve()
    index = sol.section_index()
    fused = np.flatnonzero((sol.T_peak >= sol.solidus).any(axis=(1, 2)))
    assert fused[0] < index < fused[-1]
    assert sol.section_metrics().station == pytest.approx(sol.x[index])


def test_measured_penetration_never_exceeds_the_plate():
    sol = solve(weld=weld(power=8000.0, sigma=1.5e-4), thickness=0.002)
    metrics = sol.section_metrics()
    assert metrics.full_penetration
    assert metrics.penetration == pytest.approx(0.002)
    assert metrics.haz_depth <= 0.002
    assert metrics.root_width > 0.0


def test_unmelted_section_reports_no_weld():
    sol = solve(weld=weld(power=100.0, sigma=1.2e-3), t_end=0.2)
    metrics = sol.section_metrics()
    assert not metrics.melted
    assert metrics.penetration == 0.0
    assert metrics.fusion_area == 0.0
    assert not metrics.full_penetration


def test_progress_callback_reaches_completion():
    seen: list[float] = []
    solve(t_end=0.05, on_progress=seen.append)
    assert seen[0] > 0.0
    assert seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)


def test_projected_history_matches_the_3d_fields():
    sol = solve(t_end=0.2)
    history = sol.to_history()
    assert history.T_peak.shape == (41, 31)
    assert history.T_peak.max() == pytest.approx(sol.T_peak.max())
    assert np.array_equal(history.cooling_rate, sol.cooling_rate[:, :, 0], equal_nan=True)
    assert history.dwell_temp == pytest.approx(STEEL.solidus)


def test_3d_run_through_the_high_level_api_reports_measured_penetration():
    config = ThermalSimulationConfig(
        nx=41,
        ny=31,
        nz=11,
        Lx=0.03,
        Ly=0.012,
        t_end=0.4,
        dt=0.001,
        solver="3d",
        T1=0.004,
        plate_thickness=0.004,
        material=STEEL,
        output_file=None,
        weld=weld(),
        path=path(),
    )
    result = run_thermal_simulation(config)
    assert set(result) >= {"x", "y", "z", "T", "history", "solution3d"}
    assert result["T"].shape == (41, 31)

    report = build_report(config, result)
    assert report.solver == "3d"
    assert report.section is not None
    assert report.penetration == pytest.approx(report.section.penetration)
    assert "3D fusion boundary" in report.penetration_basis
    assert "Cross-section" in "\n".join(report.summary_lines())
    assert report.as_dict()["section"]["penetration"] == pytest.approx(report.penetration)


def test_unknown_solver_is_rejected():
    config = ThermalSimulationConfig(solver="quantum", output_file=None)
    with pytest.raises(ValidationError, match="Unknown solver"):
        run_thermal_simulation(config)


def test_3d_grid_size_limit_counts_the_thickness():
    config = ThermalSimulationConfig(
        nx=500, ny=500, nz=50, solver="3d", output_file=None, material=STEEL
    )
    with pytest.raises(ValidationError, match="exceeds the limit"):
        run_thermal_simulation(config)
