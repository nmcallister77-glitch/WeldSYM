"""CLI exit codes and error reporting."""

from __future__ import annotations

import pytest

from weldsim.cli import main


def _run(monkeypatch, capsys, *args: str) -> tuple[int, str]:
    monkeypatch.setattr("sys.argv", ["weldsim", *args])
    code = main()
    return code, capsys.readouterr().err


def test_successful_run_exits_zero(monkeypatch, capsys, tmp_path):
    code, err = _run(
        monkeypatch,
        capsys,
        "--t-end",
        "0.5",
        "--dt",
        "0.005",
        "--nx",
        "21",
        "--ny",
        "11",
        "--output",
        str(tmp_path / "T.csv"),
    )

    assert code == 0
    assert err == ""
    assert (tmp_path / "T.csv").exists()


@pytest.mark.parametrize(
    "args",
    [
        ("--thickness", "0"),
        ("--speed", "0"),
        ("--speed", "-0.01"),
        ("--power", "0"),
        ("--dt", "0.5"),
    ],
)
def test_invalid_input_exits_nonzero_without_writing_output(monkeypatch, capsys, tmp_path, args):
    output = tmp_path / "T.csv"
    code, err = _run(monkeypatch, capsys, "--t-end", "0.5", "--output", str(output), *args)

    assert code == 2
    assert err.startswith("error: ")
    assert not output.exists()
