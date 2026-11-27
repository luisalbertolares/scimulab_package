from pathlib import Path

import pytest
from pymatgen.core import Lattice, Molecule, Structure

from scimulib.exceptions import (
    EnergyCalculationError,
    MismatchedGeometryError,
    MissingGeometryError,
    MissingReferenceError,
    StoichiometryError,
)
from scimulib.scimulib import (
    ChemicalEntity, MetalOxide, Nanocluster, Surface, SupportedSystem, _SUPPORTED_FROZEN
)


def structure(species, coords=None):
    if coords is None:
        coords = [[index * 0.1, 0, 0] for index in range(len(species))]
    return Structure(Lattice.cubic(10), species, coords)


def components():
    cluster = Nanocluster(
        "alloy", -5, geometry=Molecule(["Au", "Cu"], [[0, 0, 0], [1, 0, 0]])
    )
    surface = MetalOxide("surface", -10, geometry=structure(["Au", "O", "O"]))
    return cluster, surface


def supported(**kwargs):
    cluster, surface = components()
    return SupportedSystem(
        "supported", -20, geometry=structure(["Au", "Cu", "Au", "O", "O"]),
        nanocluster=cluster, surface=surface, **kwargs
    )


def reference(symbol, energy, coords=None):
    return ChemicalEntity(
        symbol, energy, geometry=structure([symbol, symbol, "Au", "O", "O"], coords)
    )


def test_components_and_adsorption_revalidated():
    system = supported()
    assert isinstance(system.geometry, Structure)
    assert system.adsorption_energy() == pytest.approx(-5)
    system.surface.total_energy = -11
    assert system.adsorption_energy() == pytest.approx(-4)
    system.nanocluster.geometry = Molecule(["Au"], [[0, 0, 0]])
    with pytest.raises(StoichiometryError, match="Mass balance"):
        system.adsorption_energy()


def test_supported_system_is_a_chemical_entity_not_a_nanocluster():
    system = supported(relaxed_references={"Au": -19, "Cu": -18},
                       frozen_references={"Au": -18, "Cu": -16})
    assert SupportedSystem.__bases__ == (ChemicalEntity,)
    assert isinstance(system, ChemicalEntity)
    assert not isinstance(system, Nanocluster)
    assert system.relaxed_references == {"Au": -19, "Cu": -18}
    assert system.frozen_references == {"Au": -18, "Cu": -16}
    system.relaxed_references["Au"] = 0
    system.frozen_references["Cu"] = 0
    assert system.mixing_energy() == pytest.approx(-1.5)
    assert system.interaction_mixing_energy() == pytest.approx(-3)
    with pytest.raises(TypeError, match="Nanocluster"):
        SupportedSystem("bad", -20, geometry=system.geometry,
                        nanocluster=system, surface=system.surface)


def test_component_composition_and_types_are_required():
    cluster, surface = components()
    with pytest.raises(StoichiometryError, match="Mass balance"):
        SupportedSystem("bad", -20, geometry=structure(["Au", "Cu", "O", "O", "O"]),
                        nanocluster=cluster, surface=surface)
    with pytest.raises(MissingGeometryError):
        SupportedSystem("bad", -20, nanocluster=cluster, surface=surface)
    with pytest.raises(TypeError, match="Nanocluster"):
        SupportedSystem("bad", -20, geometry=structure(["Au"]), nanocluster=surface,
                        surface=surface)
    with pytest.raises(TypeError, match="Surface"):
        SupportedSystem("bad", -20, geometry=structure(["Au"]), nanocluster=cluster,
                        surface=cluster)


def test_energy_methods_use_cluster_fractions_and_complete_reference_cohesion():
    system = supported(
        atomic_references={"Au": -1, "Cu": -2, "O": -3},
        relaxed_references={"Au": reference("Au", -19), "Cu": reference("Cu", -18)},
        frozen_references={"Au": reference("Au", -18), "Cu": reference("Cu", -16)},
    )
    assert system.cohesive_energy() == pytest.approx(2)
    assert system.ideal_cohesive_energy() == pytest.approx(1.7)
    assert system.mix_cohesive_energy() == pytest.approx(0.3)
    assert system.mixing_energy() == pytest.approx(-1.5)
    assert system.interaction_mixing_energy() == pytest.approx(-3)
    assert system.strain_mixing_energy() == pytest.approx(-1.5)
    assert system.mixing_energy(Au=-17) == pytest.approx(-2.5)
    assert system.strain_mixing_energy(Au=(-17, -19)) == pytest.approx(-2)
    assert system.interaction_mixing_energy(Cu=-15) == pytest.approx(-3.5)
    assert system.mixing_energy() == pytest.approx(-1.5)
    assert system.ideal_cohesive_energy(atomic_references={"O": -4}) == pytest.approx(1.3)


def test_supported_references_validate_composition_geometry_and_changes():
    system = supported()
    wrong = ChemicalEntity("bad", -19, geometry=structure(["Au", "Au", "O", "O", "O"]))
    with pytest.raises(StoichiometryError, match="composition"):
        system.mixing_energy(Au=wrong, Cu=-18)
    with pytest.raises(MissingReferenceError):
        system.mixing_energy(Au=-19)
    with pytest.raises(StoichiometryError, match="Unknown"):
        system.mixing_energy(Au=-19, Cu=-18, O=-20)
    shifted = reference("Au", -18, [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0],
                                    [0.3, 0, 0], [0.6, 0.2, 0]])
    with pytest.raises(MismatchedGeometryError):
        system.interaction_mixing_energy(Au=shifted, Cu=-16)
    au_ref = reference("Au", -19)
    system = supported(relaxed_references={"Au": au_ref, "Cu": -18})
    au_ref.geometry = structure(["Au", "Au", "Au", "O", "O", "O"])
    assert system.mixing_energy(Au=-19) == pytest.approx(-1.5)
    with pytest.raises(StoichiometryError, match="composition"):
        system.mixing_energy()
    with pytest.raises(EnergyCalculationError):
        supported(relaxed_references={"Au": float("nan")})


def test_supported_reference_coverage_with_partial_and_mixed_references():
    system = supported(relaxed_references={"Au": reference("Au", -19)})
    assert system.mixing_energy(Cu=-18) == pytest.approx(-1.5)
    with pytest.raises(MissingReferenceError) as error:
        system.mixing_energy()
    assert error.value.missing_species == ["Cu"]
    with pytest.raises(StoichiometryError, match="Unknown"):
        system.mixing_energy(Cu=-18, O=reference("Au", -19))
    with pytest.raises(StoichiometryError, match="Unknown"):
        supported(relaxed_references={"O": -19})
    with pytest.raises(EnergyCalculationError):
        system.mixing_energy(Cu=float("inf"))
    assert system.relaxed_references.keys() == {"Au"}


def test_supported_frozen_references_check_composition_before_geometry():
    system = supported(frozen_references={"Au": reference("Au", -18)})
    wrong = ChemicalEntity("wrong", -16, geometry=structure(["Cu", "Cu", "O", "O", "O"]))
    with pytest.raises(StoichiometryError, match="composition"):
        system.interaction_mixing_energy(Cu=wrong)
    assert system.interaction_mixing_energy(Cu=reference("Cu", -16)) == pytest.approx(-3)


def test_supported_frozen_spec_checks_geometry_after_composition():
    system = supported()
    shifted = reference("Au", -18, [[0, 0, 0], [0.1, 0, 0], [0.2, 0, 0],
                                    [0.3, 0, 0], [0.6, 0.2, 0]])
    with pytest.raises(MismatchedGeometryError):
        _SUPPORTED_FROZEN.resolve(system, {"Au": shifted, "Cu": -16})
    assert _SUPPORTED_FROZEN.resolve(
        system, {"Au": reference("Au", -18), "Cu": -16}
    ) == {"Au": -18, "Cu": -16}


@pytest.mark.parametrize("method", [
    "ideal_cohesive_energy", "mixing_energy", "strain_mixing_energy",
    "interaction_mixing_energy",
])
def test_mixing_validates_live_components_once_per_call(method, monkeypatch):
    system = supported(
        atomic_references={"Au": -1, "Cu": -2, "O": -3},
        relaxed_references={"Au": -19, "Cu": -18},
        frozen_references={"Au": -18, "Cu": -16},
    )
    original = SupportedSystem._validate_components
    calls = []

    def validate_components(instance):
        calls.append(instance)
        original(instance)

    monkeypatch.setattr(SupportedSystem, "_validate_components", validate_components)
    getattr(system, method)()
    assert calls == [system]
    system.nanocluster.geometry = Molecule(["Au"], [[0, 0, 0]])
    with pytest.raises(StoichiometryError, match="Mass balance"):
        getattr(system, method)()


def test_stored_reference_keys_follow_live_nanocluster_composition():
    system = supported(relaxed_references={"Au": -19, "Cu": -18})
    system.nanocluster.geometry = Molecule(["Au", "Ag"], [[0, 0, 0], [1, 0, 0]])
    system.geometry = structure(["Au", "Ag", "Au", "O", "O"])
    with pytest.raises(StoichiometryError, match="Unknown"):
        system.mixing_energy(Ag=-18)


def test_stored_references_still_require_a_multimetallic_cluster():
    cluster = Nanocluster("Au2", -5, geometry=Molecule(["Au", "Au"], [[0, 0, 0], [1, 0, 0]]))
    surface = Surface("surface", -10, geometry=structure(["O"]))
    with pytest.raises(StoichiometryError, match="multimetallic"):
        SupportedSystem(
            "supported", -20, geometry=structure(["Au", "Au", "O"]),
            nanocluster=cluster, surface=surface, relaxed_references={"Au": -19},
        )


def test_supported_system_from_vasp_retains_periodic_geometry():
    directory = Path(__file__).parent / "test_data"
    geometry = ChemicalEntity.from_vasp(directory).geometry
    cluster = Nanocluster(
        "cluster", -5, geometry=Molecule(geometry.species[:2], geometry.cart_coords[:2])
    )
    surface_geometry = geometry.copy()
    surface_geometry.remove_sites([0, 1])
    surface = Surface("surface", -10, geometry=surface_geometry)
    system = SupportedSystem.from_vasp(directory, nanocluster=cluster, surface=surface)
    assert isinstance(system, SupportedSystem)
    assert isinstance(system.geometry, Structure)
    assert system.composition == geometry.composition.get_el_amt_dict()
    assert system.nanocluster is cluster
    assert system.surface is surface


def test_supported_oxygen_vacancy_and_invalid_support():
    cluster, defective_surface = components()
    oxygen = ChemicalEntity("O2", -4, geometry=Molecule(["O", "O"], [[0, 0, 0], [1, 0, 0]]))
    defective_surface = MetalOxide(
        "defective", -10, geometry=defective_surface.geometry, oxygen_reference=oxygen
    )
    pristine_surface = MetalOxide("pristine", -12, geometry=structure(["Au", "O", "O", "O"]))
    defective = SupportedSystem(
        "defective", -20, geometry=structure(["Au", "Cu", "Au", "O", "O"]),
        nanocluster=cluster, surface=defective_surface
    )
    pristine = SupportedSystem(
        "pristine", -25, geometry=structure(["Au", "Cu", "Au", "O", "O", "O"]),
        nanocluster=cluster, surface=pristine_surface
    )
    assert defective.vacancy_formation_energy(pristine=pristine) == pytest.approx(3)
    with pytest.raises(StoichiometryError, match="Mass balance"):
        defective.vacancy_formation_energy(pristine=pristine, num_vacancies=2)
    oxygen_cluster = Nanocluster(
        "oxygenated", -7, geometry=Molecule(["Au", "Cu", "O"], [[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    )
    wrong_pristine = SupportedSystem(
        "wrong pristine", -25, geometry=pristine.geometry,
        nanocluster=oxygen_cluster, surface=defective_surface
    )
    with pytest.raises(StoichiometryError, match="nanoclusters"):
        defective.vacancy_formation_energy(pristine=wrong_pristine)
    oxygen.geometry = Molecule(["O"], [[0, 0, 0]])
    with pytest.raises(StoichiometryError, match="O2"):
        defective.vacancy_formation_energy(pristine=pristine)
    nonoxide = Surface("surface", -10, geometry=defective_surface.geometry)
    system = SupportedSystem(
        "nonoxide", -20, geometry=defective.geometry, nanocluster=cluster, surface=nonoxide
    )
    with pytest.raises(TypeError, match="MetalOxide"):
        system.vacancy_formation_energy(pristine=pristine)


@pytest.mark.parametrize("count", [0, -1, 0.5, True])
def test_supported_vacancy_rejects_invalid_count(count):
    cluster, surface = components()
    pristine_surface = MetalOxide("pristine", -12, geometry=structure(["Au", "O", "O", "O"]))
    defective = SupportedSystem(
        "defective", -20, geometry=structure(["Au", "Cu", "Au", "O", "O"]),
        nanocluster=cluster, surface=surface,
    )
    pristine = SupportedSystem(
        "pristine", -25, geometry=structure(["Au", "Cu", "Au", "O", "O", "O"]),
        nanocluster=cluster, surface=pristine_surface,
    )
    oxygen = ChemicalEntity("O2", -4, geometry=Molecule(["O", "O"], [[0, 0, 0], [1, 0, 0]]))
    with pytest.raises(StoichiometryError, match="positive integer"):
        defective.vacancy_formation_energy(
            pristine=pristine, oxygen=oxygen, num_vacancies=count
        )


def test_stored_vacancy_references_and_overrides():
    cluster, surface = components()
    pristine_surface = MetalOxide("pristine surface", -12, geometry=structure(["Au", "O", "O", "O"]))
    pristine = SupportedSystem(
        "pristine", -25, geometry=structure(["Au", "Cu", "Au", "O", "O", "O"]),
        nanocluster=cluster, surface=pristine_surface
    )
    oxygen = ChemicalEntity("O2", -4, geometry=Molecule(["O", "O"], [[0, 0, 0], [1, 0, 0]]))
    defective = SupportedSystem(
        "defective", -20, geometry=structure(["Au", "Cu", "Au", "O", "O"]),
        nanocluster=cluster, surface=surface, pristine_reference=pristine,
        oxygen_reference=oxygen
    )
    assert defective.pristine_reference is pristine
    assert defective.oxygen_reference is oxygen
    assert defective.vacancy_formation_energy() == pytest.approx(3)
    other_oxygen = ChemicalEntity("other O2", -6, geometry=oxygen.geometry)
    assert defective.vacancy_formation_energy(oxygen=other_oxygen) == pytest.approx(2)
    assert defective.vacancy_formation_energy() == pytest.approx(3)
    with pytest.raises(MissingReferenceError):
        supported().vacancy_formation_energy()
    with pytest.raises(StoichiometryError, match="Mass balance"):
        SupportedSystem(
            "bad", -20, geometry=structure(["Au", "Cu", "Au", "O", "O"]),
            nanocluster=cluster, surface=surface, pristine_reference=pristine,
            oxygen_reference=oxygen
        ).vacancy_formation_energy(num_vacancies=2)
