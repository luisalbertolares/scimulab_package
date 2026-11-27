import numpy as np
import pytest

from scimulib.bader_calc import (
    calculate_bader_net_charges,
    sum_bader_magnetic_moments,
    sum_bader_net_charges,
)
from scimulib.exceptions import (
    InvalidValenceValueError,
    MagneticMomentMismatchError,
)


def test_calculate_bader_net_charges():
    np.testing.assert_allclose(
        calculate_bader_net_charges(
            ["A", "B"],
            [2.0, 3.0],
            {"A": 1.0, "B": 4.0},
        ),
        [-1.0, 1.0],
    )


@pytest.mark.parametrize("value", [-1.0, True])
def test_calculate_bader_net_charges_rejects_invalid_valence(value):
    with pytest.raises(InvalidValenceValueError):
        calculate_bader_net_charges(["A"], [1.0], {"A": value})


def test_sum_bader_net_charges_sums_the_validated_numeric_values():
    assert sum_bader_net_charges(["A", "B"], ["1", "-1"]) == {
        "A": 1.0,
        "B": -1.0,
    }


def test_sum_bader_magnetic_moments_uses_absolute_tolerance():
    with pytest.raises(MagneticMomentMismatchError):
        sum_bader_magnetic_moments(
            ["Fe"],
            [1_000_000.5],
            total_magnetic_moment=1_000_000.0,
            atol=0.1,
        )


@pytest.mark.parametrize("atol", [-1.0, float("nan"), float("inf"), True])
def test_bader_sums_reject_invalid_tolerance(atol):
    with pytest.raises(ValueError, match="Absolute tolerance"):
        sum_bader_net_charges(["A"], [0.0], atol=atol)

    with pytest.raises(ValueError, match="Absolute tolerance"):
        sum_bader_magnetic_moments(["Fe"], [0.0], 0.0, atol=atol)
