import warnings
from pathlib import Path

from pymatgen.core import Structure

from scimulib.abinitio_parser import VaspParser, BaderParser


class ChemicalEntity:

    def __init__(self, name: str, total_energy: float, net_magmom: float, structure: Structure):
        self._name = name
        self._total_energy = total_energy
        self._net_magmom = net_magmom
        self._structure = structure

    @classmethod
    def from_vasp(cls, directory: str | Path, name: str = None) -> ChemicalEntity:
        """
        Constructs an instance of the class from a VASP calculation directory.

        This factory method extracts the total energy, net magnetic moment, and 
        geometric structure from the VASP output files located in the specified 
        directory using 'VaspParser'. It initializes the object and stores the 
        source directory and software origin in its metadata.

        Args:
            directory (str | Path): The path to the directory containing the VASP 
                calculation output files (e.g., OUTCAR, vasprun.xml).

        Returns:
            cls: An initialized instance of the class containing the parsed VASP data.

        Raises:
            ValueError: If neither the total energy nor the geometric structure 
                could be extracted from the provided directory, making initialization 
                impossible.
        """
        entity_name = name if name is not None else directory.name
        vasp_calc = VaspParser(directory=directory)

        total_energy = vasp_calc.get_total_energy()
        net_magmom = vasp_calc.get_net_magmom()
        structure = vasp_calc.get_structure()

        if total_energy is None and structure is None:
            raise ValueError(
                f"Cannot initialize {cls.__name__}: missing essential "
                "VASP data (total energy and geometry) in "
                f"{directory}."
            )

        instance = cls(
            name = entity_name,
            total_energy = total_energy,
            net_magmom = net_magmom,
            structure = structure,
            metadata={"source_directory": str(directory), "software": "VASP"}
        )

        if load_bader:
            instance.load_bader_charges()
            instance.load_bader_magmoms()

        return instance