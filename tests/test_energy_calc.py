import pytest

from scimulib.energy_calc import (
    _calculate_energy_difference,
    calculate_adsorption_energy,
    calculate_cohesive_energy,
    calculate_mixing_cohesive_energy,
    calculate_strain_mixing_energy,
    calculate_vacancy_formation_energy,
)
from scimulib.exceptions import EnergyCalculationError, StoichiometryError


def test_cohesive_energy_validates_atomic_counts_and_calculates_per_atom():
    assert calculate_cohesive_energy(
        total_energy=-17,
        atomic_energies={"X": -5, "Y": -4},
        atomic_counts={"X": 2, "Y": 1},
    ) == pytest.approx(1)

    for counts in ({}, {"X": 0}, {"X": -1}, {"X": 1.5}, {"X": True}):
        with pytest.raises(StoichiometryError):
            calculate_cohesive_energy(-5, {"X": -3}, counts)


def test_adsorption_energy_accepts_mapping_of_adsorbate_energies():
    assert calculate_adsorption_energy(
        -20, -10, {"H2": -3, "O": -2}
    ) == pytest.approx(-5)


def test_strain_mixing_energy_uses_frozen_minus_relaxed_energy():
    assert calculate_strain_mixing_energy(
        frozen_energies={"A": -8, "B": -4},
        relaxed_energies={"A": -10, "B": -5},
        atomic_fractions={"A": 0.5, "B": 0.5},
    ) == pytest.approx(1.5)


@pytest.mark.parametrize(
    ("defective_energy", "references", "coefficients"),
    [
        (0.0, {"reference": 1e308, "second": 1e308}, {"reference": 1, "second": 1}),
        (0.0, {"reference": 1e308}, {"reference": 2}),
        (1e308, {"reference": -1e308}, {"reference": 1}),
    ],
)
def test_energy_difference_rejects_overflow(defective_energy, references, coefficients):
    with pytest.raises(EnergyCalculationError):
        _calculate_energy_difference(
            defective_energy,
            references,
            coefficients,
        )


@pytest.mark.parametrize("count", [0, -1, 1.5, True])
def test_vacancy_energy_requires_positive_integer_count(count):
    with pytest.raises(StoichiometryError):
        calculate_vacancy_formation_energy(-10, -12, -1, count)


def test_mixing_cohesive_energy_validates_inputs():
    assert calculate_mixing_cohesive_energy(-6, 3) == pytest.approx(2)

    for count in (0, -1, 1.5, True):
        with pytest.raises(StoichiometryError):
            calculate_mixing_cohesive_energy(-6, count)

    with pytest.raises(EnergyCalculationError):
        calculate_mixing_cohesive_energy(float("nan"), 3)
