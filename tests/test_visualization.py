import json
import re

import numpy as np
import pytest
from pymatgen.core import Lattice, Molecule, Structure

from scimulib.exceptions import MissingSitePropertyError
from scimulib.scimulib import ChemicalEntity
from scimulib.visualization import build_py3dmol_view

pytest.importorskip("py3Dmol")


def _overlay_axes(view):
    match = re.search(r"const axes = (.*);", view._make_html())
    assert match is not None
    return json.loads(match.group(1))


def test_cartesian_overlay_is_independent_of_molecule_coordinates():
    molecule = Molecule(["C", "O"], [[10, 20, 30], [11, 20, 30]])
    view = build_py3dmol_view(molecule, axes_length=4, hover_labels=False)
    axes = _overlay_axes(view)

    assert [axis["label"] for axis in axes] == ["x", "y", "z"]
    np.testing.assert_allclose([axis["vector"] for axis in axes], np.eye(3) * 4)
    html = view._make_html()
    assert "setViewChangeCallback(updateAxes)" in html
    assert "right:0;bottom:0;pointer-events:none" in html
    assert ".addArrow(" not in html


def test_cell_overlay_uses_normalized_lattice_directions_with_colorbar():
    lattice = Lattice([[2, 0, 0], [1, 2, 0], [0, 1, 3]])
    structure = Structure(lattice, ["C"], [[0, 0, 0]])
    view = build_py3dmol_view(
        structure, axes="cell", axes_length=3, property_values=[1]
    )
    axes = _overlay_axes(view.view)

    assert [axis["label"] for axis in axes] == ["a", "b", "c"]
    expected = lattice.matrix / np.linalg.norm(lattice.matrix, axis=1)[:, None] * 3
    np.testing.assert_allclose([axis["vector"] for axis in axes], expected)
    assert "setViewChangeCallback(updateAxes)" in view._repr_html_()


def test_axes_can_be_disabled_and_cell_requires_structure():
    molecule = Molecule(["C"], [[0, 0, 0]])
    view = build_py3dmol_view(molecule, axes=None, hover_labels=False)
    assert "Coordinate axes" not in view._make_html()
    with pytest.raises(ValueError, match="only be drawn for periodic Structure"):
        build_py3dmol_view(molecule, axes="cell")


@pytest.fixture
def entity():
    structure = Structure(
        Lattice.cubic(4), ["C", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]]
    )
    structure.add_site_property("charge", [-0.5, 0.5])
    return ChemicalEntity("test", total_energy=0, net_magmom=0, geometry=structure)


def test_visualize_shows_geometry_and_accepts_custom_element_colors(entity):
    view = entity.visualize(element_colors={"C": "#123456"}, axes=None)
    html = view._make_html()
    assert "#123456" in html
    assert "linear-gradient" not in html
    with pytest.raises(TypeError, match="atom_property"):
        entity.visualize(atom_property="charge")


def test_visualize_property_maps_site_property_and_repeated_values(entity):
    view = entity.visualize_property(
        atom_property="charge", repeat=(2, 1, 1), value_format="+.1f"
    )
    html = view._repr_html_()
    assert "charge" in html
    expected_values = json.dumps(["-0.5", "-0.5", "+0.5", "+0.5"])
    assert json.dumps(expected_values)[1:-1] in html
    assert "linear-gradient" in html


def test_visualize_property_accepts_values_and_checks_invalid_inputs(entity):
    view = entity.visualize_property(
        atom_property=np.array([1.0, 2.0]), property_name="Energy", cmap="plasma"
    )
    assert "Energy" in view._repr_html_()
    with pytest.raises(MissingSitePropertyError, match="missing"):
        entity.visualize_property(atom_property="missing")
    with pytest.raises(ValueError, match="1D array of length 2"):
        entity.visualize_property(atom_property=[1.0])
    with pytest.raises(ValueError, match="required"):
        entity.visualize_property(atom_property=None)
