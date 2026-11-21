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