"""Tests for input validation and error propagation."""

from __future__ import annotations

import numpy as np
import pytest

from weldsim.cli import main as cli_main
from weldsim.exceptions import ConfigurationError, OutputError
from weldsim.simulation import (
    ThermalSimulationConfig,
    run_thermal_simulation,
    save_temperature_csv,
)
from weldsim.thermal.fd_solver import run_2d_fd_thermal
from weldsim.types import MaterialParams, WeldParams


def _weld(**kwargs) -> WeldParams:
    params = dict(
        power=3000.0,
        efficiency=0.8,
        speed=0.005,
        start_pos=(0.01, 0.025),
        direction="x",
    )
    params.update(kwargs)
    return WeldParams(**params)


def _solver_kwargs(**overrides):
    kwargs = dict(
        nx=11,
        ny=11,
        Lx=0.1,
        Ly=0.05,
        t_end=0.1,
        dt=0.05,
        weld=_weld(),
        material=MaterialParams(),
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "overrides",
    [
        {"nx": 1},
        {"ny": 2},
        {"Lx": 0.0},
        {"Ly": -0.05},
        {"dt": 0.0},
        {"t_end": -1.0},
        {"dt": 10.0, "t_end": 1.0},
        {"material": MaterialParams(rho=0.0)},
        {"weld": _weld(direction="z")},
        {"weld": _weld(sigma=0.0)},
        {"weld": _weld(power=-1.0)},
        {"weld": _weld(efficiency=1.5)},
    ],
)
def test_solver_rejects_invalid_inputs(overrides):
    with pytest.raises(ConfigurationError):
        run_2d_fd_thermal(**_solver_kwargs(**overrides))


def test_unstable_time_step_raises_configuration_error():
    with pytest.raises(ConfigurationError, match="Unstable"):
        run_2d_fd_thermal(**_solver_kwargs(nx=201, ny=201, dt=1.0, t_end=2.0))


def test_save_temperature_csv_rejects_mismatched_shapes(tmp_path):
    T = np.zeros((3, 4))
    with pytest.raises(ConfigurationError):
        save_temperature_csv(str(tmp_path / "out.csv"), np.zeros(2), np.zeros(4), T)


def test_save_temperature_csv_leaves_no_partial_file_on_failure(tmp_path):
    target = tmp_path / "subdir" / "out.csv"  # parent does not exist
    with pytest.raises(OutputError):
        save_temperature_csv(str(target), np.zeros(2), np.zeros(2), np.zeros((2, 2)))
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_run_thermal_simulation_rejects_empty_output_path():
    config = ThermalSimulationConfig(output_file="   ")
    with pytest.raises(ConfigurationError):
        run_thermal_simulation(config)


def test_run_thermal_simulation_writes_csv(tmp_path):
    out = tmp_path / "results" / "temperature.csv"
    config = ThermalSimulationConfig(
        nx=11, ny=11, t_end=0.1, dt=0.05, weld=_weld(), output_file=str(out)
    )
    result = run_thermal_simulation(config)
    assert out.exists()
    assert result["T"].shape == (11, 11)
    assert np.all(np.isfinite(result["T"]))


def test_cli_returns_usage_exit_code_on_invalid_input(tmp_path, capsys):
    code = cli_main(["--nx", "1", "--output", str(tmp_path / "temperature.csv")])
    assert code == 2
    assert "error:" in capsys.readouterr().err


def test_cli_success_exit_code(tmp_path):
    code = cli_main(
        [
            "--nx",
            "11",
            "--ny",
            "11",
            "--t-end",
            "0.1",
            "--dt",
            "0.05",
            "--output",
            str(tmp_path / "temperature.csv"),
        ]
    )
    assert code == 0
