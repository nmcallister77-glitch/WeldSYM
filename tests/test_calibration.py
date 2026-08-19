"""Tests for the measured-vs-predicted comparison and the calibration fit."""

from __future__ import annotations

import pytest

from weldsim.calibration import (
    Comparison,
    Coupon,
    CouponResidual,
    Mesh,
    calibrate,
    calibration_yaml,
    compare,
    load_calibration,
    save_calibration,
)
from weldsim.errors import ValidationError
from weldsim.materials import load_material

# Coarse enough that a fit runs in seconds; the fusion boundary is what matters
# here, and it settles long before the temperature field does.
MESH = Mesh(nx=41, ny=21, nz=7, weld_length=0.008, width=0.008)


@pytest.fixture(scope="module")
def steel():
    return load_material("S355_structural_steel")


def coupon(label="A", power=2000.0, speed=0.02, penetration=0.0015, **kwargs):
    return Coupon(
        label=label,
        power=power,
        speed=speed,
        thickness=0.003,
        penetration=penetration,
        sigma=0.0002,
        **kwargs,
    )


def test_compare_reports_residuals_against_the_measurements(steel):
    comparison = compare([coupon(fusion_width=0.0012)], steel, efficiency=0.8, mesh=MESH)
    residual = comparison.residuals[0]
    assert residual.label == "A"
    assert residual.predicted_penetration > 0
    assert residual.predicted_fusion_width > 0
    assert residual.penetration_error == pytest.approx(
        residual.predicted_penetration - residual.measured_penetration
    )
    assert residual.penetration_error_percent is not None
    assert comparison.penetration_rms >= abs(comparison.penetration_bias)
    assert comparison.width_rms is not None
    assert comparison.cost() > 0


def test_more_power_predicts_deeper_penetration(steel):
    shallow, deep = (
        compare([coupon(power=power)], steel, efficiency=0.8, mesh=MESH).residuals[0]
        for power in (1500.0, 3000.0)
    )
    assert deep.predicted_penetration > shallow.predicted_penetration


def test_absorption_efficiency_drives_predicted_depth(steel):
    low, high = (
        compare([coupon()], steel, efficiency=efficiency, mesh=MESH).residuals[0]
        for efficiency in (0.3, 0.9)
    )
    assert high.predicted_penetration > low.predicted_penetration


def test_calibration_finds_the_efficiency_that_reproduces_a_measurement(steel):
    """Fit against a prediction: the search must recover the setting that made it."""
    truth = 0.5
    synthetic = compare([coupon()], steel, efficiency=truth, mesh=MESH).residuals[0]
    measured = coupon(penetration=synthetic.predicted_penetration)

    calibration = calibrate(
        [measured],
        steel,
        efficiencies=(0.25, 0.4, 0.55, 0.9),
        tapers=(0.4,),
        baseline_efficiency=0.9,
        mesh=MESH,
    )
    assert calibration.efficiency == pytest.approx(0.55, abs=0.16)
    assert calibration.cost < calibration.baseline_cost
    assert calibration.improvement > 0
    assert calibration.material == steel.name
    assert calibration.created


def test_calibration_round_trips_through_yaml(steel, tmp_path):
    calibration = calibrate(
        [coupon(fusion_width=0.0012)],
        steel,
        efficiencies=(0.4, 0.8),
        tapers=(0.4,),
        mesh=MESH,
    )
    path = tmp_path / "calibration.yaml"
    save_calibration(str(path), calibration)
    efficiency, taper, data = load_calibration(str(path))
    assert efficiency == pytest.approx(calibration.efficiency)
    assert taper == pytest.approx(calibration.keyhole_taper)
    assert [c["label"] for c in data["coupons"]] == ["A"]
    assert "efficiency" in calibration_yaml(calibration)


def test_load_calibration_rejects_an_unrelated_yaml_file(tmp_path):
    path = tmp_path / "not_a_calibration.yaml"
    path.write_text("hello: world\n", encoding="utf-8")
    with pytest.raises(ValidationError, match="not a WeldSYM calibration"):
        load_calibration(str(path))


def test_impossible_measurements_are_rejected(steel):
    with pytest.raises(ValidationError, match="exceeds"):
        compare([coupon(penetration=0.01)], steel, efficiency=0.8, mesh=MESH)
    with pytest.raises(ValidationError, match="speed must be greater than 0"):
        compare([coupon(speed=0.0)], steel, efficiency=0.8, mesh=MESH)
    with pytest.raises(ValidationError, match="at least one measured coupon"):
        compare([], steel, efficiency=0.8, mesh=MESH)


def test_cost_weights_error_relative_to_the_measured_size():
    """A 0.2 mm miss on a 1 mm weld must not read the same as on a 5 mm weld."""

    def cost(measured, predicted):
        residual = CouponResidual(
            label="x",
            measured_penetration=measured,
            predicted_penetration=predicted,
            measured_fusion_width=None,
            predicted_fusion_width=0.0,
            full_penetration=False,
        )
        return Comparison(efficiency=0.8, keyhole_taper=0.4, residuals=[residual]).cost()

    assert cost(0.001, 0.0012) > cost(0.005, 0.0052)
    assert cost(0.002, 0.002) == pytest.approx(0.0)
