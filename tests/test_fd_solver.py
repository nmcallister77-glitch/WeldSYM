"""Tests for the 2D finite-difference thermal solver."""

from __future__ import annotations

import numpy as np
import pytest

from weldsim.thermal.fd_solver import heat_source_at_point, run_2d_fd_thermal
from weldsim.types import MaterialParams, WeldParams


@pytest.fixture
def weld() -> WeldParams:
    return WeldParams(
        power=3000.0,
        efficiency=0.8,
        speed=0.005,
        start_pos=(0.01, 0.005),
        direction="x",
        sigma=0.002,
    )


@pytest.fixture
def material() -> MaterialParams:
    return MaterialParams()


def _solve(weld, material, **kwargs):
    params = dict(nx=11, ny=6, Lx=0.02, Ly=0.01, t_end=0.02, dt=0.005)
    params.update(kwargs)
    return run_2d_fd_thermal(weld=weld, material=material, **params)


def test_grid_coordinates_and_shape(weld, material):
    x, y, T = _solve(weld, material)
    assert x.shape == (11,)
    assert y.shape == (6,)
    assert T.shape == (11, 6)
    assert x[0] == 0.0 and x[-1] == pytest.approx(0.02)
    assert y[0] == 0.0 and y[-1] == pytest.approx(0.01)
    np.testing.assert_allclose(np.diff(x), 0.02 / 10)


def test_boundaries_held_at_initial_temperature(weld, material):
    _, _, T = _solve(weld, material, T0=350.0)
    np.testing.assert_allclose(T[0, :], 350.0)
    np.testing.assert_allclose(T[-1, :], 350.0)
    np.testing.assert_allclose(T[:, 0], 350.0)
    np.testing.assert_allclose(T[:, -1], 350.0)


def test_heat_source_raises_interior_temperature(weld, material):
    _, _, T = _solve(weld, material)
    interior = T[1:-1, 1:-1]
    assert interior.max() > 300.0
    assert np.all(interior >= 300.0)


def test_zero_power_keeps_uniform_field(material):
    weld = WeldParams(power=0.0, efficiency=0.8, speed=0.005, start_pos=(0.01, 0.005))
    _, _, T = _solve(weld, material)
    np.testing.assert_allclose(T, 300.0)


def test_unstable_time_step_raises(weld, material):
    with pytest.raises(ValueError, match="Unstable"):
        _solve(weld, material, dt=1.0)


def test_invalid_direction_raises(material):
    weld = WeldParams(
        power=3000.0,
        efficiency=0.8,
        speed=0.005,
        start_pos=(0.01, 0.005),
        direction="z",
    )
    with pytest.raises(ValueError, match="direction must be"):
        _solve(weld, material)


def test_zero_time_returns_initial_field(weld, material):
    _, _, T = _solve(weld, material, t_end=0.0)
    np.testing.assert_allclose(T, 300.0)


def test_hot_spot_follows_source_along_x(material):
    weld = WeldParams(
        power=5000.0,
        efficiency=1.0,
        speed=0.2,
        start_pos=(0.002, 0.005),
        direction="x",
        sigma=0.001,
    )
    _, _, T_early = _solve(weld, material, nx=21, ny=6, t_end=0.005, dt=0.001)
    _, _, T_late = _solve(weld, material, nx=21, ny=6, t_end=0.02, dt=0.001)
    i_early = int(np.unravel_index(np.argmax(T_early), T_early.shape)[0])
    i_late = int(np.unravel_index(np.argmax(T_late), T_late.shape)[0])
    assert i_late > i_early


def test_solver_supports_y_direction(material):
    weld = WeldParams(
        power=5000.0,
        efficiency=1.0,
        speed=0.2,
        start_pos=(0.01, 0.002),
        direction="y",
        sigma=0.001,
    )
    _, _, T_early = _solve(weld, material, nx=11, ny=21, t_end=0.005, dt=0.001)
    _, _, T_late = _solve(weld, material, nx=11, ny=21, t_end=0.02, dt=0.001)
    j_early = int(np.unravel_index(np.argmax(T_early), T_early.shape)[1])
    j_late = int(np.unravel_index(np.argmax(T_late), T_late.shape)[1])
    assert j_late > j_early


def test_heat_source_peaks_at_source_center(weld):
    q_center = heat_source_at_point(weld.start_pos[0], weld.start_pos[1], 0.0, weld)
    q_offset = heat_source_at_point(
        weld.start_pos[0] + 5 * weld.sigma, weld.start_pos[1], 0.0, weld
    )
    assert q_center > q_offset
    assert q_offset >= 0.0


def test_heat_source_matches_gaussian_formula(weld):
    h = 0.001
    expected = (weld.power * weld.efficiency) / (2 * np.pi * weld.sigma**2 * h)
    assert heat_source_at_point(*weld.start_pos, 0.0, weld) == pytest.approx(expected)

    r = 0.003
    decay = np.exp(-(r**2) / (2 * weld.sigma**2))
    got = heat_source_at_point(weld.start_pos[0] + r, weld.start_pos[1], 0.0, weld)
    assert got == pytest.approx(expected * decay)


def test_heat_source_center_moves_with_time(weld):
    t = 1.0
    moved = weld.start_pos[0] + weld.speed * t
    q_at_moved = heat_source_at_point(moved, weld.start_pos[1], t, weld)
    q_at_start = heat_source_at_point(weld.start_pos[0], weld.start_pos[1], t, weld)
    assert q_at_moved > q_at_start


def test_heat_source_moves_along_y_direction():
    weld = WeldParams(
        power=3000.0,
        efficiency=0.8,
        speed=0.01,
        start_pos=(0.01, 0.005),
        direction="y",
    )
    t = 0.5
    y_src = weld.start_pos[1] + weld.speed * t
    q_moved = heat_source_at_point(weld.start_pos[0], y_src, t, weld)
    q_start = heat_source_at_point(weld.start_pos[0], weld.start_pos[1], t, weld)
    assert q_moved > q_start


def test_heat_source_invalid_direction_raises(weld):
    weld.direction = "diagonal"
    with pytest.raises(ValueError, match="direction must be"):
        heat_source_at_point(0.0, 0.0, 0.0, weld)


def test_heat_source_scales_linearly_with_power(weld):
    q1 = heat_source_at_point(*weld.start_pos, 0.0, weld)
    weld.power *= 2
    q2 = heat_source_at_point(*weld.start_pos, 0.0, weld)
    assert q2 == pytest.approx(2 * q1)
