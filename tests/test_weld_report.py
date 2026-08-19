"""Tests for the engineering outputs derived from a thermal run."""

from __future__ import annotations

import numpy as np
import pytest

from weldsim.distortion import estimate_distortion, inherent_strain
from weldsim.keyhole import estimate_keyhole, peak_intensity
from weldsim.materials import load_material
from weldsim.microstructure import carbon_equivalent_iiw, predict_microstructure
from weldsim.report import build_report
from weldsim.simulation import ThermalSimulationConfig, run_thermal_simulation
from weldsim.thermal.fd_solver import T85_LOWER, T85_UPPER, ThermalHistory
from weldsim.types import WeldParams
from weldsim.weld_metrics import compute_weld_metrics, level_extent
from weldsim.weld_path import WeldPath, WobbleParams
from weldsim.wobble_analysis import analyse_wobble


@pytest.fixture(scope="module")
def steel():
    return load_material("S355_structural_steel")


@pytest.fixture(scope="module")
def titanium():
    return load_material("Ti6Al4V")


@pytest.fixture(scope="module")
def steel_run(steel):
    """A short bead-on-plate run that melts, plus its config."""
    weld = WeldParams(
        power=2000.0,
        efficiency=0.8,
        speed=0.01,
        start_pos=(0.01, 0.015),
        direction="x",
        sigma=0.001,
    )
    config = ThermalSimulationConfig(
        nx=61,
        ny=41,
        Lx=0.05,
        Ly=0.03,
        t_end=6.0,
        dt=0.004,
        weld=weld,
        material=steel,
        output_file=None,
        T1=0.003,
        plate_thickness=0.003,
        path=WeldPath(start=(0.01, 0.015), end=(0.04, 0.015), speed=0.01),
        probe=(0.025, 0.015),
    )
    return config, run_thermal_simulation(config)


def test_level_extent_interpolates_between_nodes():
    coord = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = np.array([0.0, 0.0, 10.0, 0.0, 0.0])
    # The level-5 contour sits halfway between the peak and each zero neighbour.
    assert level_extent(coord, values, 5.0) == pytest.approx(1.0)
    assert level_extent(coord, values, 20.0) == 0.0


def test_thermal_history_is_consistent_with_the_field(steel_run):
    _, result = steel_run
    history = result["history"]
    assert isinstance(history, ThermalHistory)
    assert history.T_peak.shape == result["T"].shape
    # Nothing can be hotter at the end than it ever was.
    assert np.all(history.T_peak >= result["T"] - 1e-9)
    # Cells that cooled through 800 C have a positive t8/5 and cooling rate.
    cooled = np.isfinite(history.t_8_5)
    assert cooled.any()
    assert np.all(history.t_8_5[cooled] > 0)
    assert np.all(history.cooling_rate[np.isfinite(history.cooling_rate)] > 0)


def test_evaporation_cap_holds_the_peak_at_boiling(steel):
    """Without a cap, a concentrated beam runs to physically impossible temperatures."""
    weld = WeldParams(
        power=4000.0,
        efficiency=0.9,
        speed=0.02,
        start_pos=(0.005, 0.01),
        direction="x",
        sigma=0.0005,
    )
    base = dict(
        nx=41,
        ny=31,
        Lx=0.03,
        Ly=0.02,
        t_end=1.0,
        dt=0.002,
        weld=weld,
        material=steel,
        output_file=None,
        T1=0.002,
    )
    capped = run_thermal_simulation(ThermalSimulationConfig(**base, phase_change=True))
    uncapped = run_thermal_simulation(ThermalSimulationConfig(**base, phase_change=False))
    assert capped["history"].T_peak.max() <= steel.boiling + 1e-6
    assert uncapped["history"].T_peak.max() > steel.boiling


def test_weld_metrics_measure_the_fusion_zone(steel_run, steel):
    config, result = steel_run
    metrics = compute_weld_metrics(result["x"], result["y"], result["history"], steel, config.weld)
    assert metrics.melted
    assert metrics.fusion_width > 0
    assert metrics.fusion_area > 0
    assert metrics.haz_width > 0
    # The HAZ lies outside the fusion zone, so the reported section must be wider
    # at the HAZ limit than at the solidus.
    assert metrics.haz_limit < metrics.solidus
    assert metrics.t_8_5 is None or metrics.t_8_5 > 0
    assert metrics.profile.y.shape == result["y"].shape


def test_weld_metrics_report_no_weld_when_nothing_melts(steel):
    weld = WeldParams(
        power=50.0,
        efficiency=0.5,
        speed=0.02,
        start_pos=(0.005, 0.01),
        direction="x",
        sigma=0.002,
    )
    config = ThermalSimulationConfig(
        nx=31,
        ny=21,
        Lx=0.03,
        Ly=0.02,
        t_end=1.0,
        dt=0.002,
        weld=weld,
        material=steel,
        output_file=None,
        T1=0.005,
    )
    result = run_thermal_simulation(config)
    metrics = compute_weld_metrics(result["x"], result["y"], result["history"], steel, weld)
    assert not metrics.melted
    assert metrics.fusion_width == 0.0
    assert any("never reached the solidus" in w for w in metrics.warnings)


def test_keyhole_mode_follows_intensity(steel):
    focused = WeldParams(power=3000.0, efficiency=0.8, speed=0.05, start_pos=(0, 0), sigma=0.0001)
    defocused = WeldParams(power=3000.0, efficiency=0.8, speed=0.05, start_pos=(0, 0), sigma=0.005)
    assert peak_intensity(focused) > peak_intensity(defocused)
    assert estimate_keyhole(focused, steel, 0.005).mode == "keyhole"
    assert estimate_keyhole(defocused, steel, 0.005).mode == "conduction"


def test_keyhole_depth_is_capped_by_the_plate(steel):
    weld = WeldParams(power=6000.0, efficiency=0.9, speed=0.005, start_pos=(0, 0), sigma=0.0002)
    estimate = estimate_keyhole(weld, steel, plate_thickness=0.002)
    assert estimate.full_penetration
    assert estimate.depth == pytest.approx(0.002)
    assert any("full penetration" in note for note in estimate.notes)


def test_wobble_lowers_intensity_and_widens_the_channel(steel):
    weld = WeldParams(power=2000.0, efficiency=0.8, speed=0.05, start_pos=(0, 0), sigma=0.0001)
    plain = estimate_keyhole(weld, steel, 0.01)
    wobbled = estimate_keyhole(weld, steel, 0.01, wobble_amplitude=0.0005)
    assert plain.mode == "keyhole"
    assert wobbled.peak_intensity < plain.peak_intensity
    assert wobbled.width > plain.width
    # Same energy spread over a wider channel is a shallower weld.
    assert wobbled.depth < plain.depth


def test_steel_microstructure_responds_to_cooling_time(steel_run, steel):
    _, result = steel_run
    x, y, history = result["x"], result["y"], result["history"]
    fast = predict_microstructure(x, y, history, steel, t_8_5=1.0, cooling_rate=300.0)
    slow = predict_microstructure(x, y, history, steel, t_8_5=100.0, cooling_rate=3.0)
    assert fast.phases["Martensite"] == pytest.approx(1.0)
    assert slow.phases["Martensite"] == pytest.approx(0.0)
    assert slow.phases["Ferrite + pearlite"] > 0.9
    assert fast.hardness_hv is not None and slow.hardness_hv is not None
    assert fast.hardness_hv > slow.hardness_hv
    assert fast.bands  # the material YAML defines HAZ bands
    assert any("cracking" in w for w in fast.warnings)


def test_titanium_microstructure_uses_the_cooling_rate(steel_run, titanium):
    _, result = steel_run
    x, y, history = result["x"], result["y"], result["history"]
    quenched = predict_microstructure(x, y, history, titanium, t_8_5=0.1, cooling_rate=1000.0)
    slow = predict_microstructure(x, y, history, titanium, t_8_5=50.0, cooling_rate=5.0)
    assert quenched.model == "martensitic_rate"
    assert quenched.phases["Martensitic alpha-prime"] == pytest.approx(1.0)
    assert slow.phases["Martensitic alpha-prime"] == pytest.approx(0.0)
    assert quenched.hardness_hv is not None and slow.hardness_hv is not None
    assert quenched.hardness_hv > slow.hardness_hv


def test_microstructure_without_cooling_data_warns(steel_run, steel):
    _, result = steel_run
    micro = predict_microstructure(
        result["x"], result["y"], result["history"], steel, t_8_5=None, cooling_rate=None
    )
    assert micro.phases == {}
    assert micro.hardness_hv is None
    assert any("t8/5" in w for w in micro.warnings)


def test_carbon_equivalent_matches_the_iiw_formula():
    assert carbon_equivalent_iiw({"C": 0.18, "Mn": 1.5}) == pytest.approx(0.18 + 1.5 / 6.0)
    assert carbon_equivalent_iiw({"Al": 0.03}) is None


def test_inherent_strain_saturates_at_the_yield_strain(steel):
    mech = steel.mechanical
    delta_T_yield = mech.yield_strain / mech.thermal_expansion
    T_peak = np.array([[steel.T0, steel.T0 + delta_T_yield * 0.5, steel.T0 + 10 * delta_T_yield]])
    strain, dT = inherent_strain(T_peak, steel)
    assert dT == pytest.approx(delta_T_yield)
    assert strain[0, 0] == 0.0
    assert strain[0, 1] == 0.0  # below the yield temperature rise
    assert strain[0, 2] == pytest.approx(mech.thermal_expansion * delta_T_yield)


def test_distortion_scales_with_heat_input(steel_run, steel):
    config, result = steel_run
    x, y, history = result["x"], result["y"], result["history"]
    estimate = estimate_distortion(
        x,
        y,
        history,
        steel,
        plate_thickness=0.003,
        plate_width=config.Ly,
        plate_length=config.Lx,
        penetration=0.001,
    )
    assert estimate.shrinkage_force > 0
    assert estimate.transverse_shrinkage > 0
    assert estimate.angular_distortion > 0
    assert estimate.peak_tensile_stress == pytest.approx(steel.mechanical.yield_stress)

    # A through-thickness plastic zone is symmetric about the mid-plane, so it
    # shortens the plate without bending it.
    full = estimate_distortion(
        x,
        y,
        history,
        steel,
        plate_thickness=0.003,
        plate_width=config.Ly,
        plate_length=config.Lx,
        penetration=0.003,
    )
    assert full.angular_distortion == pytest.approx(0.0)
    assert full.transverse_shrinkage > estimate.transverse_shrinkage


def test_wobble_analysis_quantifies_spreading(steel_run):
    config, result = steel_run
    x, y = result["x"], result["y"]
    plain = analyse_wobble(
        config.path,
        WobbleParams(amplitude=0.0, frequency=0.0),
        config.weld,
        config.T1,
        x,
        y,
        dt=0.01,
    )
    wobbled = analyse_wobble(
        config.path,
        WobbleParams(amplitude=0.001, frequency=100.0, pattern="circle"),
        config.weld,
        config.T1,
        x,
        y,
        dt=0.01,
    )
    assert wobbled.swept_width > plain.swept_width
    assert wobbled.peak_energy_density < plain.peak_energy_density
    assert wobbled.peak_reduction > 0
    assert wobbled.peak_beam_speed > plain.peak_beam_speed
    assert wobbled.pitch == pytest.approx(config.path.speed / 100.0)


def test_build_report_produces_a_full_assessment(steel_run):
    config, result = steel_run
    report = build_report(config, result)
    assert report.material_name == "S355"
    assert report.metrics.melted
    assert report.keyhole.depth > 0
    assert report.microstructure is not None
    assert report.distortion is not None
    assert report.wobble is not None

    summary = "\n".join(report.summary_lines())
    assert "Penetration" in summary
    assert "Fusion zone" in summary

    data = report.as_dict()
    assert data["metrics"]["fusion_width"] > 0
    assert "profile" not in data["metrics"]
    import json

    json.dumps(data)  # must be serialisable for the JSON download


def test_report_works_without_material_library_data():
    """A bare thermal-properties run still reports, with generic metallurgy."""
    config = ThermalSimulationConfig(
        nx=31,
        ny=21,
        Lx=0.03,
        Ly=0.02,
        t_end=1.0,
        dt=0.002,
        output_file=None,
        T1=0.003,
    )
    result = run_thermal_simulation(config)
    report = build_report(config, result)
    assert "Custom" in report.material_name
    assert report.microstructure is not None
    assert report.microstructure.bands == []


def test_cooling_interval_constants_are_800_and_500_celsius():
    assert T85_UPPER - 273.15 == pytest.approx(800.0)
    assert T85_LOWER - 273.15 == pytest.approx(500.0)
