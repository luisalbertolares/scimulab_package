from types import SimpleNamespace

import pytest
from pymatgen.core import Molecule

from scimulib.exceptions import MissingDataError, StoichiometryError
from scimulib.references import Entry, ReferenceSpec, same_atom_count
from scimulib.scimulib import ChemicalEntity, Nanocluster


def line(*symbols):
    return Molecule(list(symbols), [[index, 0, 0] for index in range(len(symbols))])


def test_rule_runs_on_plain_objects():
    target = SimpleNamespace(name="alloy", composition={"Au": 1.0, "Cu": 1.0})
    same_atom_count(Entry(target, "Au", SimpleNamespace(composition={"Au": 2.0}), "cluster"))
    with pytest.raises(StoichiometryError, match=r"\(2\), got: 3"):
        same_atom_count(Entry(target, "Au", SimpleNamespace(composition={"Au": 3.0}), "cluster"))


def test_rules_run_by_phase_and_energies_are_read_after_entity_rules():
    calls = []
    spec = ReferenceSpec(
        "probe",
        key_rules=(lambda refs: calls.append("keys"),),
        entity_rules=(lambda entry: calls.append(f"entity {entry.key}"),),
        set_rules=(lambda refs: calls.append("set"),),
    )
    alloy = Nanocluster("alloy", -6, geometry=line("Au", "Cu"))
    with pytest.raises(MissingDataError):
        spec.resolve(alloy, {"Au": ChemicalEntity("Au", geometry=line("Au")), "Cu": -2})
    assert calls == ["keys", "entity Au"]
    calls.clear()
    spec.resolve(alloy, {"Au": ChemicalEntity("Au", -1, geometry=line("Au")), "Cu": -2})
    assert calls == ["keys", "entity Au", "set"]


def test_error_precedence_does_not_depend_on_keyword_order():
    alloy = Nanocluster("alloy", -6, geometry=line("Au", "Cu"))
    no_energy = ChemicalEntity("Au", geometry=line("Au"))
    dimer = ChemicalEntity("Cu2", -2, geometry=line("Cu", "Cu"))
    with pytest.raises(StoichiometryError):
        alloy.cohesive_energy(Au=no_energy, Cu=dimer)
    with pytest.raises(StoichiometryError):
        alloy.cohesive_energy(Cu=dimer, Au=no_energy)
    with pytest.raises(MissingDataError):
        alloy.cohesive_energy(Au=no_energy, Cu=-2)
