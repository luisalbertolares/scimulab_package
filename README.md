# scimulib

`scimulib` is a Python library for representing and post-processing ab initio
calculations of chemical entities, nanoclusters, surfaces, metal oxides, and
supported nanoclusters. This guide focuses on the classes in
[`src/scimulib/scimulib.py`](src/scimulib/scimulib.py).

## Installation

Install the package and its declared dependencies from the repository:

```bash
python -m pip install .
```

For interactive 3D visualization and property color maps, install the optional
visualization dependencies:

```bash
python -m pip install ".[visualization]"
```

To run the test suite, install its optional dependency with
`python -m pip install ".[test]"`.

The classes described below are defined in `scimulib.scimulib`:

```python
from scimulib.scimulib import (
    ChemicalEntity,
    MetalOxide,
    Nanocluster,
    Surface,
    SupportedSystem,
)
```

Geometries are represented with `pymatgen` `Molecule` or `Structure` objects.
Nanoclusters are non-periodic; surfaces and metal oxides use periodic
`Structure` geometries.

## Choosing a class

| Class | Use it for |
| --- | --- |
| `ChemicalEntity` | A general entity with energy, magnetic moment, and/or geometry data. |
| `Nanocluster` | A non-periodic cluster or nanoparticle model, including alloy clusters. |
| `Surface` | A periodic surface model. |
| `MetalOxide` | A periodic metal-oxide surface with oxygen-vacancy calculations. |
| `SupportedSystem` | A nanocluster supported on a surface, represented together with its components. |

`Nanocluster`, `Surface`, and `MetalOxide` build on `ChemicalEntity`.
`SupportedSystem` also inherits its general entity properties, and requires a
`Nanocluster` and a `Surface` as components.

## Common entity data

`ChemicalEntity` is the base class. Its constructor accepts a name, optional
total energy and net magnetic moment, an optional geometry, and optional
valence-electron counts. At least one of energy, geometry, or magnetic moment
must be supplied.

```python
from pymatgen.core import Molecule
from scimulib.scimulib import ChemicalEntity

atom = ChemicalEntity(
    name="Au atom",
    total_energy=-3.0,
    geometry=Molecule(["Au"], [[0, 0, 0]]),
)

print(atom.composition)       # {"Au": 1.0}
print(atom.cartesian_coords)  # Cartesian coordinates as a NumPy array
```

The geometry and composition properties expose data derived from the stored
geometry. Assigning a geometry stores a copy; reading `geometry` returns a
clean copy. Properties that need unavailable data raise a package-specific
exception rather than returning a fabricated value.

### Reading VASP output

Use `from_vasp` to create an entity from a calculation directory. The class
method parses the available VASP energy, magnetic moment, structure, and
valence-electron data. It also attempts to read Bader charges and magnetic
moments when those files are available; unavailable values are logged, and
construction fails if no usable energy, geometry, or magnetic moment is found.

```python
entity = ChemicalEntity.from_vasp(
    directory="path/to/vasp/calculation",
    name="Au cluster",  # Optional; defaults to the directory name
)
```

The factory is inherited by the specialized classes, so it can also be called
as `Nanocluster.from_vasp(...)`, `Surface.from_vasp(...)`, or
`MetalOxide.from_vasp(...)` when the parsed geometry is appropriate for that
class.

### Geometry, Bader data, and visualization

For entities with geometry, useful common properties and methods include:

- `composition`, `atomic_fractions`, `symbols`, `atomic_symbols`,
  `cartesian_coords`, and, for periodic structures, `fractional_coords`.
- `has_same_geometry(other, ...)` to compare compatible molecular or periodic
  geometries.
- `bader_charges`, `bader_net_charges`, and `bader_magnetic_moments` for
  per-site values. `sum_bader_charges` and `sum_bader_magmoms` aggregate values
  by element. Net Bader charges require valence-electron data.
- `visualize(...)` for an interactive structure viewer and
  `visualize_property(atom_property=...)` for coloring sites by a property.

These features require the corresponding geometry or site properties to be
present. For example, Bader properties are only available if they were parsed
or added to the geometry.

## Energy references

Energy methods use reference values supplied either when an object is created
or for an individual method call. References may be `ChemicalEntity` objects
or numeric total energies. Atomic references are keyed by element symbol, for
example `{"Au": atom}`.

References passed to a method override stored references for that call without
changing the stored mapping. Stored mappings are copied, while entity
references remain live and are validated again when used. Entity references
can be checked against geometry and composition rules; a numeric energy has no
geometry to validate. Use consistent energy units for the target and all
references (typically eV).

For example, the following computes cohesive energy per atom using an isolated
Au atom as the reference:

```python
from pymatgen.core import Molecule
from scimulib.scimulib import ChemicalEntity, Nanocluster

au_atom = ChemicalEntity(
    name="Au atom",
    total_energy=-3.0,
    geometry=Molecule(["Au"], [[0, 0, 0]]),
)
au2 = Nanocluster(
    name="Au2",
    total_energy=-6.5,
    geometry=Molecule(["Au", "Au"], [[0, 0, 0], [2.5, 0, 0]]),
    atomic_references={"Au": au_atom},
)

cohesive_energy = au2.cohesive_energy()
```

The example energies are illustrative only; provide values from calculations
performed with a consistent setup.

## Class reference

### `ChemicalEntity`

Use this class for general molecules or periodic structures, and as the common
base for more specialized models. In addition to the properties above, it
provides:

- `cohesive_energy(**named_atoms)`: cohesive energy per atom from isolated
  atomic references.
- Bader analysis properties and visualization methods described above.
- `from_vasp(directory, name=None, **references)`: the VASP factory method.

### `Nanocluster`

Use `Nanocluster` for a non-periodic atomic aggregate. A supplied periodic
`Structure` is converted to a `Molecule` using its Cartesian coordinates.
Alongside atomic references, it can store relaxed and frozen monometallic
cluster references, keyed by element.

Its energy methods are:

- `cohesive_energy(**named_atoms)` for cohesive energy relative to isolated
  atoms.
- `ideal_cohesive_energy(Au=(cluster_ref, atom_ref), ...)` for the
  composition-weighted ideal cohesive energy.
- `mix_cohesive_energy(...)` for cohesive energy relative to that ideal value.
- `mixing_energy(**named_clusters)` for mixing relative to relaxed
  monometallic clusters.
- `strain_mixing_energy(Au=(frozen_ref, relaxed_ref), ...)` for the
  composition-weighted frozen-minus-relaxed reference contribution.
- `interaction_mixing_energy(**named_clusters)` for the alloy energy relative
  to frozen monometallic cluster references.

Cluster references supplied as entities are checked for their species and atom
count. Frozen entity references are also checked against the target geometry.
Numeric energies cannot be checked for geometry or composition.

### `Surface`

Use `Surface` for a periodic surface model. It accepts a `Structure` geometry
and atomic references, and otherwise provides the common `ChemicalEntity`
properties and methods.

### `MetalOxide`

`MetalOxide` is a `Surface` with optional stored references for oxygen-vacancy
calculations:

- `pristine_reference`: the oxide before oxygen removal.
- `oxygen_reference`: a `ChemicalEntity` representing an O₂ molecule.

Call `vacancy_formation_energy(pristine=..., oxygen=..., num_vacancies=1)` on
the defective oxide. References can be stored on the object or supplied by
keyword to the method. The method checks the O₂ reference and composition
balance, and returns the total formation energy for the requested number of
vacancies (not the value per vacancy).

### `SupportedSystem`

Use `SupportedSystem` when both the combined supported geometry and its
isolated components are available. Construction requires:

- `nanocluster`: a `Nanocluster` component.
- `surface`: a `Surface` component.
- `geometry`: the combined geometry of the supported system.

The combined composition must match the sum of the component compositions.
Relaxed and frozen supported references can be stored by nanocluster element;
they represent complete supported systems, not isolated clusters.

Its principal methods are:

- `adsorption_energy()`: supported-system energy minus the isolated surface
  and nanocluster energies.
- `cohesive_energy(...)`, `ideal_cohesive_energy(...)`, and
  `mix_cohesive_energy(...)`.
- `mixing_energy(...)`, `strain_mixing_energy(...)`, and
  `interaction_mixing_energy(...)`, using the corresponding supported-system
  references.
- `vacancy_formation_energy(...)` when the surface component is a
  `MetalOxide`; this uses a pristine supported system and an O₂ reference.

The supported-system mixing energies are weighted by the nanocluster
composition. Component and reference compositions are validated when the
relevant methods are called.

## Notes

- Geometry-dependent properties and calculations require geometry. Energy
  calculations also require the target and reference energies they use.
- Stored entity references are revalidated when used; keep their geometries
  and energies consistent with the calculation represented by the target.
- The package-specific exceptions in `scimulib.exceptions` identify missing
  data, missing references, incompatible geometry, and invalid stoichiometry.
