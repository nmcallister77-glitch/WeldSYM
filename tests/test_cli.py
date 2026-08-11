"""Tests for the weldsim command-line interface."""

from __future__ import annotations

import numpy as np
import pytest

from weldsim import cli


@pytest.fixture
def captured_config(monkeypatch):
    """Capture the config passed to the solver without running a simulation."""
    seen: dict[str, object] = {}

    def fake_run(config):
        seen["config"] = config
        return {
            "x": np.zeros(2),
            "y": np.zeros(2),
            "T": np.array([[300.0, 400.0], [500.0, 600.0]]),
        }

    monkeypatch.setattr(cli, "run_thermal_simulation", fake_run)
    return seen


def test_main_uses_defaults(monkeypatch, captured_config, capsys):
    monkeypatch.setattr("sys.argv", ["weldsim"])
    cli.main()

    config = captured_config["config"]
    assert (config.nx, config.ny) == (51, 26)
    assert (config.Lx, config.Ly) == (0.1, 0.05)
    assert (config.t_end, config.dt) == (10.0, 0.05)
    assert config.output_file == "results/temperature.csv"
    assert config.weld.power == 3000.0
    assert config.weld.efficiency == 0.8
    assert config.weld.speed == 0.005
    assert config.weld.direction == "x"
    assert config.weld.sigma == 0.002
    assert config.weld.start_pos == (0.01, 0.025)
    assert config.material == cli.MaterialParams()

    out = capsys.readouterr().out
    assert "results/temperature.csv" in out
    assert "300.0 K" in out and "600.0 K" in out


def test_main_parses_arguments(monkeypatch, captured_config, tmp_path):
    out_file = tmp_path / "custom.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "weldsim",
            "--power",
            "1234.5",
            "--efficiency",
            "0.6",
            "--speed",
            "0.02",
            "--t-end",
            "1.5",
            "--dt",
            "0.01",
            "--nx",
            "21",
            "--ny",
            "11",
            "--Lx",
            "0.2",
            "--Ly",
            "0.08",
            "--output",
            str(out_file),
        ],
    )
    cli.main()

    config = captured_config["config"]
    assert (config.nx, config.ny) == (21, 11)
    assert (config.Lx, config.Ly) == (0.2, 0.08)
    assert (config.t_end, config.dt) == (1.5, 0.01)
    assert config.output_file == str(out_file)
    assert config.weld.power == 1234.5
    assert config.weld.efficiency == 0.6
    assert config.weld.speed == 0.02
    assert config.weld.start_pos == (0.01, 0.04)


def test_main_rejects_invalid_numeric_argument(monkeypatch, captured_config):
    monkeypatch.setattr("sys.argv", ["weldsim", "--power", "not-a-number"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 2
    assert "config" not in captured_config


def test_help_exits_cleanly(monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["weldsim", "--help"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert exc.value.code == 0
    assert "--power" in capsys.readouterr().out


def test_end_to_end_writes_output(monkeypatch, tmp_path):
    out_file = tmp_path / "results" / "temperature.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "weldsim",
            "--nx",
            "11",
            "--ny",
            "6",
            "--Lx",
            "0.02",
            "--Ly",
            "0.01",
            "--t-end",
            "0.02",
            "--dt",
            "0.005",
            "--output",
            str(out_file),
        ],
    )
    cli.main()
    assert out_file.exists()
    assert len(out_file.read_text(encoding="utf-8").splitlines()) == 1 + 11 * 6
