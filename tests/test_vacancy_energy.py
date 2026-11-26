import pytest
from pymatgen.core import Lattice, Molecule, Structure

from scimulib.exceptions import EnergyCalculationError, StoichiometryError
from scimulib.scimulib import ChemicalEntity, MetalOxide, _SYSTEM_POLICY


def oxide(name, species, energy):
    structure = Structure(
        Lattice.cubic(10),
        species,
        [[index / 10, 0, 0] for index in range(len(species))],
    )
    return MetalOxide(name, energy, 0, structure)


def molecule(name, species, energy):
    geometry = Molecule(species, [[index, 0, 0] for index in range(len(species))])
    return ChemicalEntity(name, energy, 0, geometry)


def test_vacancy_formation_with_fractional_oxygen():
    pristine = oxide("pristine", ["Ti", "O", "O", "O"], -20)
    defective = oxide("defective", ["Ti", "O", "O"], -15)
    oxygen = molecule("oxygen", ["O", "O"], -4)

    assert defective.vacancy_formation_energy(pristine, oxygen) == pytest.approx(3)

    two_vacancies = oxide("two vacancies", ["Ti", "O"], -11)
    assert two_vacancies.vacancy_formation_energy(
        pristine, oxygen, num_vacancies=2
    ) == pytest.approx(5)


def test_mass_balance_supports_weighted_references_without_changing_energies():
    pristine = oxide("pristine", ["Ti", "O", "O"], -20)
    defective = oxide("defective", ["Ti", "O"], -15)
    oxygen = molecule("oxygen", ["O", "O"], -4)
    references = pristine._generate_calculation_references(
        _SYSTEM_POLICY,
        coefficients={"defective": 1, "oxygen": 0.5},
        defective=defective,
        oxygen=oxygen,
    )
    assert references == {"defective": -15, "oxygen": -4}
    with pytest.raises(StoichiometryError, match="Mass balance mismatch"):
        pristine._generate_calculation_references(
            _SYSTEM_POLICY, defective=defective, oxygen=oxygen
        )


@pytest.mark.parametrize("count", [0, -1, 0.5, True])
def test_vacancy_rejects_invalid_count(count):
    pristine = oxide("pristine", ["Ti", "O", "O"], -20)
    defective = oxide("defective", ["Ti", "O"], -15)
    oxygen = molecule("oxygen", ["O", "O"], -4)
    with pytest.raises(StoichiometryError):
        defective.vacancy_formation_energy(pristine, oxygen, num_vacancies=count)


def test_vacancy_rejects_wrong_composition_and_nonfinite_energy():
    pristine = oxide("pristine", ["Ti", "O", "O"], -20)
    defective = oxide("defective", ["Ti", "O"], -15)
    oxygen = molecule("oxygen", ["O", "O"], -4)
    with pytest.raises(StoichiometryError):
        defective.vacancy_formation_energy(
            pristine, molecule("water", ["O", "H", "H"], -4)
        )
    with pytest.raises(StoichiometryError, match="Mass balance mismatch"):
        defective.vacancy_formation_energy(pristine, oxygen, num_vacancies=2)
    oxygen.total_energy = float("nan")
    with pytest.raises(EnergyCalculationError):
        defective.vacancy_formation_energy(pristine, oxygen)


def test_vacancy_rejects_changes_to_other_species():
    pristine = oxide("pristine", ["Ti", "Ti", "O", "O"], -20)
    defective = oxide("defective", ["Ti", "O"], -15)
    oxygen = molecule("oxygen", ["O", "O"], -4)
    with pytest.raises(StoichiometryError, match="Mass balance mismatch"):
        defective.vacancy_formation_energy(pristine, oxygen)


def test_weighted_positional_references():
    pristine = oxide("pristine", ["Ti", "O"], -20)
    titanium = molecule("titanium", ["Ti"], -10)
    oxygen = molecule("oxygen", ["O", "O"], -4)
    assert pristine._generate_calculation_references(
        _SYSTEM_POLICY,
        titanium,
        oxygen,
        coefficients={"system_0": 1, "system_1": 0.5},
    ) == {"system_0": -10, "system_1": -4}


def test_existing_atomic_reference_call():
    pristine = oxide("pristine", ["Ti", "O", "O"], -20)
    titanium = molecule("titanium", ["Ti"], -3)
    assert pristine.cohesive_energy(titanium, O=-2) == pytest.approx(13 / 3)


@pytest.mark.parametrize("coefficients", [
    {"defective": 1, "oxygen": 0},
    {"defective": 1, "oxygen": float("nan")},
    {"defective": 1, "oxygen": 0.5, "unknown": 1},
    {"defective": 1},
])
def test_invalid_mass_balance_coefficients(coefficients):
    pristine = oxide("pristine", ["Ti", "O", "O"], -20)
    defective = oxide("defective", ["Ti", "O"], -15)
    oxygen = molecule("oxygen", ["O", "O"], -4)
    with pytest.raises((StoichiometryError, ValueError)):
        pristine._generate_calculation_references(
            _SYSTEM_POLICY, coefficients=coefficients, defective=defective, oxygen=oxygen
        )
