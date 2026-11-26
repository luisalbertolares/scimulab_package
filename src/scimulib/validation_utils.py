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
    """
    Class representing the policy for validating reference data ( e.g., 
    geometry, composition, and energies) in calculation methods inside classes.
    """
    def __init__(
        self,
        label: str,
        single_species: bool = True,
        expected_atoms: int | None = None,
        same_atom_count: bool = False,
        same_geometry: bool = False,
        check_mass_balance: bool = False,
        extra_validator: Callable[[Self, Self], None] | None = None,
    ):
        """
        Initializes the reference policy with the specified validation rules.

        Attributes:
            label: A descriptive label for the reference type, used in error
                messages and default keys (e.g., 'atom', 'cluster', 'adsorption').
            single_species: Whether each reference must consist of a single
                chemical species.
            expected_atoms: Fixed number of atoms required in each reference,
                or None if no fixed count is enforced.
            same_atom_count: Whether each reference must have the same total
                atom count as the target structure.
            same_geometry: Whether each reference must share the same geometry
                as the target structure.
            check_mass_balance: Whether the combined composition of all
                provided references, weighted by any supplied stoichiometric
                coefficients, must match the target structure (applies only
                when all references are entity instances).
            extra_validator: Optional callable with signature
                ``(target, reference) -> None`` that raises an exception if
                custom validation fails.
        """
        self.label = label
        self.single_species = single_species
        self.expected_atoms = expected_atoms
        self.same_atom_count = same_atom_count
        self.same_geometry = same_geometry
        self.check_mass_balance = check_mass_balance
        self.extra_validator = extra_validator