import pytest
from pymatgen.core import Lattice, Molecule, Structure

from scimulib.exceptions import MissingReferenceError, StoichiometryError
from scimulib.scimulib import ChemicalEntity, MetalOxide, Surface


def oxide(species):
    return Structure(
        Lattice.cubic(10),
        species,
        [[index * 0.1, 0, 0] for index in range(len(species))],
    )


def oxygen_reference(energy=-4):
    return ChemicalEntity(
        "O2", energy, geometry=Molecule(["O", "O"], [[0, 0, 0], [1, 0, 0]])
    )


def test_metal_oxide_vacancy_energy_uses_stored_references_and_call_overrides():
    pristine = MetalOxide("pristine", -12, geometry=oxide(["Au", "O", "O", "O"]))
    oxygen = oxygen_reference()
    defective = MetalOxide(
        "defective",
        -9,
        geometry=oxide(["Au", "O", "O"]),
        pristine_reference=pristine,
        oxygen_reference=oxygen,
    )

    assert defective.pristine_reference is pristine
    assert defective.oxygen_reference is oxygen
    assert defective.vacancy_formation_energy() == pytest.approx(1)

    alternate_oxygen = oxygen_reference(energy=-6)
    assert defective.vacancy_formation_energy(oxygen=alternate_oxygen) == pytest.approx(0)
    assert defective.oxygen_reference is oxygen


def test_metal_oxide_vacancy_energy_requires_references_and_valid_count():
    defective = MetalOxide("defective", -9, geometry=oxide(["Au", "O", "O"]))
    with pytest.raises(MissingReferenceError):
        defective.vacancy_formation_energy()

    pristine = MetalOxide("pristine", -12, geometry=oxide(["Au", "O", "O", "O"]))
    oxygen = oxygen_reference()
    for count in (0, -1, 0.5, True):
        with pytest.raises(StoichiometryError, match="positive integer"):
            defective.vacancy_formation_energy(
                pristine=pristine, oxygen=oxygen, num_vacancies=count
            )


def test_metal_oxide_constructor_validates_vacancy_references():
    pristine = MetalOxide("pristine", -12, geometry=oxide(["Au", "O", "O", "O"]))
    defective_geometry = oxide(["Au", "O", "O"])
    with pytest.raises(TypeError, match="MetalOxide"):
        MetalOxide(
            "defective",
            -9,
            geometry=defective_geometry,
            pristine_reference=Surface("surface", -12, geometry=pristine.geometry),
        )

    non_molecular_oxygen = ChemicalEntity(
        "O", -2, geometry=Molecule(["O"], [[0, 0, 0]])
    )
    with pytest.raises(StoichiometryError, match="O2"):
        MetalOxide(
            "defective",
            -9,
            geometry=defective_geometry,
            oxygen_reference=non_molecular_oxygen,
        )

    unbalanced_pristine = MetalOxide(
        "unbalanced pristine", -12, geometry=oxide(["Au", "Au", "O", "O", "O"])
    )
    with pytest.raises(StoichiometryError, match="Mass balance"):
        MetalOxide(
            "defective",
            -9,
            geometry=defective_geometry,
            pristine_reference=unbalanced_pristine,
            oxygen_reference=oxygen_reference(),
        )
