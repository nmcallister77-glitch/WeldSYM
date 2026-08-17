"""Input validation and stability guards for the 2D thermal simulation."""

from __future__ import annotations

import numpy as np
import pytest

from weldsim.errors import StabilityError, ValidationError
from weldsim.simulation import (
    MAX_CELLS,
    MAX_STEPS,
    ThermalSimulationConfig,
    run_thermal_simulation,
    validate_config,
)
from weldsim.types import MaterialParams, WeldParams


def _weld(**overrides) -> WeldParams:
    params = dict(
        power=1500.0,
        efficiency=0.8,
        speed=0.01,
        start_pos=(0.01, 0.025),
        direction="x",
        sigma=0.002,
    )
    params.update(overrides)
    return WeldParams(**params)


def _config(**overrides) -> ThermalSimulationConfig:
    params = dict(
        nx=21,
        ny=11,
        Lx=0.08,
        Ly=0.05,
        t_end=0.5,
        dt=0.005,
        weld=_weld(),
        material=MaterialParams(),
        output_file=None,
        T1=0.005,
    )
    params.update(overrides)
    return ThermalSimulationConfig(**params)


def test_valid_config_runs_and_stays_finite(tmp_path):
    result = run_thermal_simulation(_config(output_file=str(tmp_path / "T.csv")))

    assert np.all(np.isfinite(result["T"]))
    assert result["T"].max() > MaterialParams().T0
    assert (tmp_path / "T.csv").exists()


@pytest.mark.parametrize(
    "overrides",
    [
        {"T1": 0.0},
        {"Lx": 0.0},
        {"Ly": -0.05},
        {"dt": 0.0},
        {"t_end": 0.0},
        {"nx": 2},
        {"ny": 1},
        {"material": MaterialParams(k=0.0)},
        {"material": MaterialParams(rho=0.0)},
        {"material": MaterialParams(cp=0.0)},
    ],
)
def test_rejects_non_physical_geometry_and_material(overrides):
    with pytest.raises(ValidationError):
        validate_config(_config(**overrides))


@pytest.mark.parametrize(
    "weld_overrides",
    [
        {"power": 0.0},
        {"power": -100.0},
        {"speed": 0.0},
        {"speed": -0.01},
        {"efficiency": 0.0},
        {"efficiency": 1.5},
        {"sigma": 0.0},
    ],
)
def test_rejects_non_physical_process_parameters(weld_overrides):
    with pytest.raises(ValidationError):
        validate_config(_config(weld=_weld(**weld_overrides)))


def test_rejects_oversized_problems():
    side = int(MAX_CELLS**0.5) + 10
    with pytest.raises(ValidationError, match="exceeds the limit"):
        validate_config(_config(nx=side, ny=side))

    with pytest.raises(ValidationError, match="time steps"):
        validate_config(_config(t_end=1.0, dt=1.0 / (MAX_STEPS + 10)))


def test_unstable_time_step_reports_the_largest_stable_dt():
    with pytest.raises(StabilityError, match=r"Use dt <= "):
        run_thermal_simulation(_config(dt=0.5))


def test_zero_thickness_is_rejected_before_producing_nan():
    """Previously ``--thickness 0`` divided by zero and reported a NaN field."""
    with pytest.raises(ValidationError, match="thickness"):
        run_thermal_simulation(_config(T1=0.0))
