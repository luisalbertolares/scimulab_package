from typing import Callable, Self


def check_pymatgen_imports():
    """
    Checks if the required pymatgen classes are available. Raises ImportError if not.
    """
    try:
        from pymatgen.io.vasp.outputs import Vasprun, Oszicar
        from pymatgen.core import Structure, Lattice
    except ImportError as e:
        raise ImportError(
            "The 'pymatgen' library is required for VASP parsing. "
            "Please install it via 'pip install pymatgen'."
        ) from e


class ReferencePolicy:
    def __init__(
        self,
        label: str,
        single_species: bool = True,
        expected_atoms: int | None = None,
        same_atom_count: bool = False,
        same_geometry: bool = False,
        extra_validator: Callable[[Self, Self], None] | None = None,
    ):
        self.label = label
        self.single_species = single_species
        self.expected_atoms = expected_atoms
        self.same_atom_count = same_atom_count
        self.same_geometry = same_geometry
        self.extra_validator = extra_validator