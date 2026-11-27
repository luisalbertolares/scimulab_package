from pathlib import Path

from math import isclose
import numpy as np
import pytest

from pymatgen.core import Lattice, Molecule, Structure

from scimulib.exceptions import (
    EnergyCalculationError,
    MagneticMomentMismatchError,
    MissingDataError,
    MissingReferenceError,
    MismatchedGeometryError,
    StoichiometryError,
    VaspExtractionError,
)
from scimulib.scimulib import ChemicalEntity, Nanocluster, Surface


# Path of the test output files from VASP (Au7Co6 cluster)
TEST_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = TEST_DIRECTORY / "test_data"

# Reference energy, magnetic moment and geometry
with open (DATA_DIRECTORY / 'energy.txt', 'r') as f:
    energy_txt = float(f.read().strip())

with open (DATA_DIRECTORY / 'magmom.txt', 'r') as f:
    magmom_txt = float(f.read().strip())

bader_charges = np.array([
    11.155531,
    11.188453,
    11.144737,
    11.110426,
    11.172038,
    11.312841,
    11.312844,
    8.226238,
    8.905799,
    8.921268,
    8.846117,
    8.867026,
    8.836679
])

bader_magnetic_moments = np.array([
    0.024969,
    0.078398,
    0.078382,
    0.025017,
    0.053099,
    0.159327,
    0.159328,
    1.900855,
    2.104810,
    2.099021,
    2.109633,
    2.102573,
    2.104588
])

# Reference structure for the tests
lattice = Lattice.cubic(21)
species = ['Au'] * 7 + ['Co'] * 6
coords = [
    [0.5012406651094983,  0.4073275107542340,  0.4104206126129512],
    [0.5022071923580069,  0.4168117137338751,  0.5939111034924919],
    [0.5939111034924919,  0.4168117137338751,  0.5022071923580069],
    [0.4104206126129512,  0.4073275107542340,  0.5012406651094983],
    [0.4139187530975761,  0.5061287488693690,  0.4139187530975761],
    [0.4063587383186778,  0.5088486782956219,  0.5839164616914249],
    [0.5839164616914249,  0.5088486782956219,  0.4063587383186778],
    [0.5000000000000000,  0.5000000000000000,  0.5000000000000000],
    [0.4997179022681417,  0.5926953036560114,  0.4278038193796933],
    [0.5035821153555964,  0.5797114286814797,  0.5775458069905356],
    [0.4278038193796933,  0.5926953036560114,  0.4997179022681417],
    [0.5775458069905356,  0.5797114286814797,  0.5035821153555964],
    [0.5829062494204733,  0.5082438485277284,  0.5829062494204733]
]
test_structure = Structure(lattice, species, coords)
test_structure.add_site_property("bader_charge", bader_charges)
test_structure.add_site_property("bader_magmom", bader_magnetic_moments)

test_entity = ChemicalEntity(
    name="Au7Co6",
    total_energy=energy_txt,
    net_magmom=magmom_txt,
    geometry=test_structure
)

def test_chemical_entity_properties():
    assert test_entity.name == "Au7Co6"
    assert isclose(test_entity.total_energy, energy_txt, rel_tol=1e-6)
    assert isclose(test_entity.net_magmom, magmom_txt, rel_tol=1e-6)
    assert np.allclose(test_entity.bader_charges, bader_charges)
    assert np.allclose(test_entity.bader_magnetic_moments, bader_magnetic_moments)
    assert np.allclose(test_entity.fractional_coords, test_structure.frac_coords)
    assert np.allclose(test_entity.cartesian_coords, test_structure.cart_coords)
    assert test_entity.composition == test_structure.composition

def test_chemical_entity_from_vasp():
    vasp_entity = ChemicalEntity.from_vasp(
        name="Au7Co6",
        directory=DATA_DIRECTORY
    )
    assert vasp_entity.name == "Au7Co6"
    assert isclose(vasp_entity.total_energy, energy_txt, rel_tol=1e-6)
    assert isclose(vasp_entity.net_magmom, magmom_txt, rel_tol=1e-6)
    assert np.allclose(vasp_entity.bader_charges, bader_charges)
    assert np.allclose(vasp_entity.bader_magnetic_moments, bader_magnetic_moments)
    assert np.allclose(vasp_entity.fractional_coords, test_structure.frac_coords)
    assert np.allclose(vasp_entity.cartesian_coords, test_structure.cart_coords)
    assert vasp_entity.composition == test_structure.composition


def test_optional_data_and_from_vasp_string_directory():
    entity = ChemicalEntity("H", geometry=Molecule(["H"], [[0, 0, 0]]))
    with pytest.raises(MissingDataError):
        _ = entity.total_energy
    with pytest.raises(MissingDataError):
        _ = entity.net_magmom
    assert ChemicalEntity("energy only", total_energy=-1).total_energy == -1
    with pytest.raises(ValueError):
        ChemicalEntity("empty")
    assert ChemicalEntity.from_vasp(str(DATA_DIRECTORY)).name == DATA_DIRECTORY.name


def test_from_vasp_handles_missing_optional_properties(tmp_path, monkeypatch):
    class Parser:
        def __init__(self, directory):
            pass

        def get_total_energy(self):
            raise VaspExtractionError("missing energy")

        def get_net_magmom(self):
            raise VaspExtractionError("missing magnetization")

        def get_structure(self):
            return Structure(Lattice.cubic(5), ["H"], [[0, 0, 0]])

        def get_valence_electrons(self):
            raise VaspExtractionError("missing valence")

    monkeypatch.setattr("scimulib.scimulib.VaspParser", Parser)
    entity = ChemicalEntity.from_vasp(tmp_path)
    assert entity.composition == {"H": 1.0}
    with pytest.raises(MissingDataError):
        _ = entity.total_energy


def test_geometry_and_valence_changes_refresh_derived_properties():
    geometry = Molecule(["H"], [[0, 0, 0]], site_properties={"bader_charge": [0.5]})
    valences = {"H": 1}
    entity = ChemicalEntity("H", geometry=geometry, valence_electrons=valences)
    assert entity.composition == {"H": 1.0}
    np.testing.assert_allclose(entity.bader_net_charges, [0.5])
    valences["H"] = 20
    geometry[0] = "O"
    assert entity.composition == {"H": 1.0}
    entity.valence_electrons = {"H": 2}
    np.testing.assert_allclose(entity.bader_net_charges, [1.5])
    entity.geometry = Molecule(["O", "O"], [[0, 0, 0], [1, 0, 0]])
    assert entity.composition == {"O": 2.0}
    assert entity.atomic_symbols == ["O", "O"]
    assert entity.atomic_fractions == {"O": 1.0}


def test_periodic_geometry_comparison_ignores_all_species_without_mutation():
    lattice = Lattice.cubic(10)
    coords = [[0, 0, 0], [0.1, 0, 0], [0.3, 0.1, 0]]
    alloy = ChemicalEntity("alloy", geometry=Structure(lattice, ["Au", "Cu", "O"], coords))
    reference = ChemicalEntity("reference", geometry=Structure(lattice, ["Au", "Au", "O"], coords))
    different = ChemicalEntity(
        "different", geometry=Structure(
            lattice, ["Au", "Au", "O"], [[0, 0, 0], [0.1, 0, 0], [0.5, 0.3, 0]]
        )
    )

    assert not alloy.has_same_geometry(reference)
    assert alloy.has_same_geometry(reference, ignore_species=True)
    assert not alloy.has_same_geometry(different, ignore_species=True)
    assert alloy.atomic_symbols == ["Au", "Cu", "O"]
    assert reference.atomic_symbols == ["Au", "Au", "O"]


def test_summed_magnetic_moments_follow_updated_total():
    entity = ChemicalEntity(
        "H",
        net_magmom=1,
        geometry=Molecule(["H"], [[0, 0, 0]], site_properties={"bader_magmom": [1]}),
    )
    assert entity.sum_bader_magmoms == {"H": 1.0}
    entity.net_magmom = 2
    with pytest.raises(MagneticMomentMismatchError):
        _ = entity.sum_bader_magmoms


def test_cohesive_energy_mixing_for_numeric_and_entity_references():
    alloy = Nanocluster("alloy", -6, 0, Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]]))
    refs = {"Au": (-4, -1), "Cu": (-6, -2)}
    assert alloy.ideal_cohesive_energy(**refs) == pytest.approx(1.0)
    assert alloy.mix_cohesive_energy(**refs) == pytest.approx(0.5)
    au_cluster = Nanocluster("Au2", -4, 0, Molecule(["Au", "Au"], [[0, 0, 0], [1, 0, 0]]))
    cu_atom = ChemicalEntity("Cu", -2, 0, Molecule(["Cu"], [[0, 0, 0]]))
    assert alloy.ideal_cohesive_energy(Au=(au_cluster, -1), Cu=(-6, cu_atom)) == pytest.approx(1.0)
    assert alloy.mix_cohesive_energy(Au=(au_cluster, -1), Cu=(-6, cu_atom)) == pytest.approx(0.5)


def test_named_single_species_references_must_match_their_keys():
    alloy = Nanocluster("alloy", -5, 0, Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]]))
    au = ChemicalEntity("Au", -1, 0, Molecule(["Au"], [[0, 0, 0]]))
    cu = ChemicalEntity("Cu", -2, 0, Molecule(["Cu"], [[0, 0, 0]]))
    with pytest.raises(StoichiometryError):
        alloy.cohesive_energy(Au=cu, Cu=au)


def test_stored_atomic_references_and_temporary_overrides():
    geometry = Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]])
    au = ChemicalEntity("Au", -1, geometry=Molecule(["Au"], [[0, 0, 0]]))
    references = {"Au": au, "Cu": -2}
    alloy = ChemicalEntity("alloy", -6, geometry=geometry, atomic_references=references)
    expected = alloy.cohesive_energy(Au=au, Cu=-2)
    assert alloy.cohesive_energy(**{"Au": au, "Cu": -2}) == pytest.approx(expected)
    references["Cu"] = -100
    alloy.atomic_references["Cu"] = -100
    assert alloy.cohesive_energy() == pytest.approx(expected)
    assert alloy.cohesive_energy(Cu=-3) == pytest.approx(alloy.cohesive_energy(Au=au, Cu=-3))
    assert alloy.cohesive_energy() == pytest.approx(expected)
    au.total_energy = -2
    assert alloy.cohesive_energy() == pytest.approx(alloy.cohesive_energy(Au=-2, Cu=-2))
    au.geometry = Molecule(["Cu"], [[0, 0, 0]])
    with pytest.raises(StoichiometryError):
        alloy.cohesive_energy()
    assert alloy.cohesive_energy(Au=-2) == pytest.approx(
        alloy.cohesive_energy(Au=-2, Cu=-2)
    )


def test_partial_and_invalid_stored_atomic_references():
    geometry = Structure(Lattice.cubic(5), ["Au", "Cu"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    surface = Surface("surface", -6, geometry=geometry, atomic_references={"Au": -1})
    with pytest.raises(MissingReferenceError):
        surface.cohesive_energy()
    assert surface.cohesive_energy(Cu=-2) == pytest.approx(
        Surface("other", -6, geometry=geometry).cohesive_energy(Au=-1, Cu=-2)
    )
    wrong_atom = ChemicalEntity("Au2", -2, geometry=Molecule(["Au", "Au"], [[0, 0, 0], [1, 0, 0]]))
    with pytest.raises(StoichiometryError):
        Surface("surface", -6, geometry=geometry, atomic_references={"Au": wrong_atom})
    with pytest.raises(StoichiometryError):
        Surface("surface", -6, geometry=geometry, atomic_references={"O": -1})
    with pytest.raises(EnergyCalculationError):
        Surface("surface", -6, geometry=geometry, atomic_references={"Au": float("nan")})
    surface.geometry = Structure(Lattice.cubic(5), ["Cu"], [[0, 0, 0]])
    with pytest.raises(StoichiometryError, match="not present"):
        surface.cohesive_energy(Cu=-2)


def test_stored_cluster_references_across_energy_methods():
    geometry = Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]])
    bare = Nanocluster("bare", -6, geometry=geometry)
    alloy = Nanocluster(
        "alloy", -6, geometry=geometry,
        atomic_references={"Au": -1, "Cu": -2},
        relaxed_references={"Au": -4, "Cu": -6},
        frozen_references={"Au": -3, "Cu": -4},
    )
    assert alloy.ideal_cohesive_energy() == pytest.approx(
        bare.ideal_cohesive_energy(Au=(-4, -1), Cu=(-6, -2))
    )
    assert alloy.mix_cohesive_energy() == pytest.approx(
        bare.mix_cohesive_energy(Au=(-4, -1), Cu=(-6, -2))
    )
    assert alloy.mixing_energy(Au=-5) == pytest.approx(bare.mixing_energy(Au=-5, Cu=-6))
    au_cluster = Nanocluster(
        "Au2", -5, geometry=Molecule(["Au", "Au"], [[0, 0, 0], [1, 0, 0]])
    )
    assert alloy.mixing_energy(Au=au_cluster) == pytest.approx(
        bare.mixing_energy(Au=-5, Cu=-6)
    )
    assert alloy.interaction_mixing_energy() == pytest.approx(
        bare.interaction_mixing_energy(Au=-3, Cu=-4)
    )
    assert alloy.strain_mixing_energy(Au=(-2, -5)) == pytest.approx(
        bare.strain_mixing_energy(Au=(-2, -5), Cu=(-4, -6))
    )
    assert alloy.ideal_cohesive_energy(Au=(-5, -2)) == pytest.approx(
        bare.ideal_cohesive_energy(Au=(-5, -2), Cu=(-6, -2))
    )
    assert alloy.relaxed_references == {"Au": -4, "Cu": -6}
    assert alloy.frozen_references == {"Au": -3, "Cu": -4}
    partial = Nanocluster("partial", -6, geometry=geometry, relaxed_references={"Au": -4})
    assert partial.mixing_energy(Cu=-6) == pytest.approx(bare.mixing_energy(Au=-4, Cu=-6))
    with pytest.raises(MissingReferenceError):
        partial.mixing_energy()


def test_stored_cluster_reference_validation(monkeypatch):
    geometry = Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]])
    mono_geometry = Molecule(["Au", "Au"], [[0, 0, 0], [2, 0, 0]])
    mono = Nanocluster("Au2", -4, geometry=mono_geometry)
    with pytest.raises(StoichiometryError, match="multimetallic"):
        Nanocluster("Au2", -4, geometry=mono_geometry, relaxed_references={"Au": -4})
    with pytest.raises(StoichiometryError, match="multimetallic"):
        Nanocluster(
            "AuH", -4, geometry=Molecule(["Au", "H"], [[0, 0, 0], [1, 0, 0]]),
            relaxed_references={"Au": -4},
        )
    with pytest.raises(StoichiometryError):
        Nanocluster("alloy", -6, geometry=geometry, relaxed_references={"Cu": mono})
    monkeypatch.setattr(
        ChemicalEntity, "has_same_geometry",
        lambda self, other, **kwargs: np.allclose(self.cartesian_coords, other.cartesian_coords),
    )
    with pytest.raises(MismatchedGeometryError):
        Nanocluster("alloy", -6, geometry=geometry, frozen_references={"Au": mono})
    mono.geometry = Molecule(["Au", "Au"], [[0, 0, 0], [1, 0, 0]])
    alloy = Nanocluster(
        "alloy", -6, geometry=geometry, frozen_references={"Au": mono, "Cu": -4}
    )
    assert alloy.interaction_mixing_energy() == pytest.approx(-2)
    mono.geometry = mono_geometry
    with pytest.raises(MismatchedGeometryError):
        alloy.interaction_mixing_energy()


@pytest.mark.parametrize("stored", [None, {"Au": -1, "Cu": -2}])
@pytest.mark.parametrize("method", [
    "cohesive_energy", "ideal_cohesive_energy", "mix_cohesive_energy",
    "mixing_energy", "strain_mixing_energy", "interaction_mixing_energy",
])
def test_energy_methods_reject_positional_references(stored, method):
    alloy = Nanocluster(
        "alloy", -6, geometry=Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]]),
        atomic_references=stored,
    )
    au = ChemicalEntity("Au", -1, geometry=Molecule(["Au"], [[0, 0, 0]]))
    with pytest.raises(TypeError):
        getattr(alloy, method)(au)
    with pytest.raises(TypeError):
        getattr(alloy, method)({"Au": -1, "Cu": -2})


@pytest.mark.parametrize("stored", [None, {"Au": -3, "Cu": -4}])
def test_named_frozen_reference_geometry_checked_once(monkeypatch, stored):
    geometry = Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]])
    alloy = Nanocluster("alloy", -6, geometry=geometry, frozen_references=stored)
    au = Nanocluster("Au2", -3, geometry=Molecule(["Au", "Au"], [[0, 0, 0], [1, 0, 0]]))
    matches = []

    def same_geometry(target, reference, **kwargs):
        matches.append((target, reference))
        return True

    monkeypatch.setattr(ChemicalEntity, "has_same_geometry", same_geometry)
    assert alloy.interaction_mixing_energy(Au=au, Cu=-4) == pytest.approx(-2.5)
    assert matches == [(alloy, au)]


def test_from_vasp_accepts_constructor_references():
    alloy = Nanocluster.from_vasp(
        DATA_DIRECTORY, atomic_references={"Au": -1},
        relaxed_references={"Au": -4},
    )
    assert alloy.atomic_references == {"Au": -1}
    assert alloy.relaxed_references == {"Au": -4}
