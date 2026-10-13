"""Tests for Fresnel optics, ray tracing, and recoil-based keyhole estimates."""

from __future__ import annotations

import pytest

from weldsim.keyhole import (
    KEYHOLE_THRESHOLD_INTENSITY,
    estimate_keyhole,
    keyhole_threshold_intensity,
    recoil_onset_temperature,
    recoil_pressure,
)
from weldsim.materials import load_material
from weldsim.optics import (
    drude_refractive_index,
    fresnel_absorptivity,
    keyhole_absorptance,
)
from weldsim.types import WeldParams


def _ti64_drude_index() -> tuple[float, float]:
    return drude_refractive_index(1.07e-6, 1.11e30, 1.9e-6)


def test_drude_fresnel_absorptivity_matches_flat_plate_run():
    n, k = _ti64_drude_index()
    # LaserbeamFoam cunninghamValidation run, log 'Total Q deposited' 59.0 W
    # of 156 W on the flat plate.
    assert fresnel_absorptivity(1.0, n, k) == pytest.approx(0.378, abs=0.005)


def test_fresnel_absorptivity_increases_toward_grazing_incidence():
    n, k = _ti64_drude_index()
    assert fresnel_absorptivity(0.0, n, k) == pytest.approx(0.0, abs=1e-12)
    assert fresnel_absorptivity(0.2, n, k) > fresnel_absorptivity(1.0, n, k)


def test_conical_keyhole_absorptance_increases_with_depth_and_is_bounded():
    n, k = _ti64_drude_index()
    depths = [0.0, 50e-6, 117e-6, 300e-6]
    traces = [keyhole_absorptance(n, k, depth, 70e-6) for depth in depths]
    assert traces[0].absorptance == pytest.approx(traces[0].flat_absorptivity)
    assert all(
        shallower.absorptance <= deeper.absorptance for shallower, deeper in zip(traces, traces[1:])
    )
    assert 0.5 <= traces[2].absorptance <= 0.75
    assert all(trace.absorptance <= 1.0 for trace in traces)


def test_material_optics_and_recoil_properties_are_optional_by_yaml():
    titanium = load_material("Ti6Al4V")
    assert titanium.optical_n == pytest.approx(2.8)
    assert titanium.optical_k == pytest.approx(3.5)
    assert titanium.vapor_molar_mass == pytest.approx(0.0479)
    assert titanium.recoil_prefactor == pytest.approx(0.54)
    assert titanium.ambient_pressure == pytest.approx(101325.0)

    aluminum = load_material("Aluminum")
    assert aluminum.optical_n is None
    assert aluminum.optical_k is None
    assert aluminum.vapor_molar_mass is None
    assert aluminum.recoil_prefactor == pytest.approx(0.54)
    assert aluminum.ambient_pressure == pytest.approx(101325.0)


def test_ti64_recoil_onset_and_keyhole_estimate():
    titanium = load_material("Ti6Al4V")
    assert recoil_pressure(titanium.boiling, titanium) == pytest.approx(
        titanium.recoil_prefactor * titanium.ambient_pressure
    )
    onset = recoil_onset_temperature(titanium, 70e-6)
    assert onset is not None
    assert onset < titanium.boiling

    weld = WeldParams(
        power=156.0,
        efficiency=0.378,
        speed=1e-4,
        start_pos=(0.0, 0.0),
        sigma=70e-6,
    )
    estimate = estimate_keyhole(weld, titanium, 0.005)
    assert estimate.mode == "keyhole"
    assert estimate.recoil_onset_temperature is not None


def test_missing_vapor_molar_mass_uses_fallback_threshold():
    aluminum = load_material("Aluminum")
    weld = WeldParams(
        power=1000.0,
        efficiency=0.5,
        speed=0.01,
        start_pos=(0.0, 0.0),
        sigma=100e-6,
    )
    assert recoil_onset_temperature(aluminum, weld.sigma) is None
    assert keyhole_threshold_intensity(aluminum, weld) == KEYHOLE_THRESHOLD_INTENSITY
    with pytest.raises(ValueError, match="no vapour molar mass"):
        recoil_pressure(aluminum.boiling, aluminum)
