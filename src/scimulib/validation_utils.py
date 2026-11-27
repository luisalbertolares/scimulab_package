from scimulib.exceptions import StoichiometryError


def validate_vacancy_count(num_vacancies: int) -> None:
    """Require a positive integer number of vacancies."""
    if (
        isinstance(num_vacancies, bool)
        or not isinstance(num_vacancies, int)
        or num_vacancies <= 0
    ):
        raise StoichiometryError("The number of vacancies must be a positive integer.")


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
