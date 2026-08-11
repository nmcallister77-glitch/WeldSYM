"""Tests for weldsim.types dataclasses."""

from __future__ import annotations

from dataclasses import fields

from weldsim.types import MaterialParams, WeldParams


def test_weld_params_defaults():
    weld = WeldParams(power=3000.0, efficiency=0.8, speed=0.005, start_pos=(0.01, 0.025))
    assert weld.direction == "x"
    assert weld.heat_source_type == "goldak"
    assert weld.goldak_params is None
    assert weld.sigma == 0.002


def test_weld_params_overrides():
    weld = WeldParams(
        power=1500.0,
        efficiency=0.5,
        speed=0.01,
        start_pos=(0.0, 0.0),
        direction="y",
        heat_source_type="gaussian",
        goldak_params={"a_f": 0.001},
        sigma=0.004,
    )
    assert weld.direction == "y"
    assert weld.heat_source_type == "gaussian"
    assert weld.goldak_params == {"a_f": 0.001}
    assert weld.sigma == 0.004


def test_weld_params_requires_core_arguments():
    required = {f.name for f in fields(WeldParams) if f.default is not None}
    assert {"power", "efficiency", "speed", "start_pos"} <= required


def test_material_params_defaults_and_diffusivity():
    mat = MaterialParams()
    assert (mat.k, mat.rho, mat.cp, mat.T0) == (50.0, 7850.0, 500.0, 300.0)
    alpha = mat.k / (mat.rho * mat.cp)
    assert 1e-6 < alpha < 1e-4


def test_material_params_equality():
    assert MaterialParams() == MaterialParams()
    assert MaterialParams(k=25.0) != MaterialParams()
