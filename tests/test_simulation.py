"""Tests for the high-level simulation API."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from weldsim.simulation import (
    MaterialParams,
    ThermalSimulationConfig,
    WeldParams,
    run_thermal_simulation,
    save_temperature_csv,
)


def _config(**kwargs) -> ThermalSimulationConfig:
    params = dict(
        nx=11,
        ny=6,
        Lx=0.02,
        Ly=0.01,
        t_end=0.02,
        dt=0.005,
        output_file=None,
    )
    params.update(kwargs)
    return ThermalSimulationConfig(**params)


def test_config_defaults():
    config = ThermalSimulationConfig()
    assert (config.nx, config.ny) == (51, 26)
    assert (config.Lx, config.Ly) == (0.1, 0.05)
    assert (config.t_end, config.dt) == (10.0, 0.05)
    assert config.weld is None
    assert config.material == MaterialParams()
    assert config.output_file == "results/temperature.csv"


def test_run_returns_grid_and_field():
    config = _config()
    result = run_thermal_simulation(config)
    assert set(result) == {"x", "y", "T"}
    assert result["x"].shape == (11,)
    assert result["y"].shape == (6,)
    assert result["T"].shape == (11, 6)
    assert result["T"].max() > config.material.T0


def test_run_fills_in_default_weld_params():
    config = _config()
    run_thermal_simulation(config)
    assert isinstance(config.weld, WeldParams)
    assert config.weld.direction == "x"
    assert config.weld.start_pos == (0.01, config.Ly / 2)


def test_run_respects_supplied_weld_params():
    weld = WeldParams(power=0.0, efficiency=0.8, speed=0.005, start_pos=(0.005, 0.005))
    config = _config(weld=weld)
    result = run_thermal_simulation(config)
    assert config.weld is weld
    np.testing.assert_allclose(result["T"], config.material.T0)


def test_run_uses_material_initial_temperature():
    config = _config(
        weld=WeldParams(power=0.0, efficiency=0.8, speed=0.005, start_pos=(0.01, 0.005)),
        material=MaterialParams(T0=400.0),
    )
    result = run_thermal_simulation(config)
    np.testing.assert_allclose(result["T"], 400.0)


def test_run_propagates_solver_errors():
    with pytest.raises(ValueError, match="Unstable"):
        run_thermal_simulation(_config(dt=1.0))


def test_run_writes_csv_and_creates_directories(tmp_path):
    out = tmp_path / "nested" / "temperature.csv"
    result = run_thermal_simulation(_config(output_file=str(out)))
    assert out.exists()

    with out.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == result["T"].size
    assert float(rows[0]["x_m"]) == pytest.approx(result["x"][0])
    assert float(rows[0]["T_K"]) == pytest.approx(result["T"][0, 0], abs=1e-3)


def test_run_writes_csv_into_current_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run_thermal_simulation(_config(output_file="temperature.csv"))
    assert (tmp_path / "temperature.csv").exists()


def test_save_temperature_csv_row_ordering(tmp_path):
    x = np.array([0.0, 1.0])
    y = np.array([0.0, 2.0, 4.0])
    T = np.array([[300.0, 301.0, 302.0], [303.0, 304.0, 305.0]])
    out = tmp_path / "t.csv"

    save_temperature_csv(str(out), x, y, T)

    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines[0] == "x_m,y_m,T_K"
    assert len(lines) == 1 + T.size
    parsed = [tuple(float(v) for v in line.split(",")) for line in lines[1:]]
    assert parsed[0] == (0.0, 0.0, 300.0)
    assert parsed[1] == (0.0, 2.0, 301.0)
    assert parsed[3] == (1.0, 0.0, 303.0)
    assert parsed[-1] == (1.0, 4.0, 305.0)
