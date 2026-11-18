import warnings
from pathlib import Path

import numpy as np

from pymatgen.core import Structure

from scimulib.abinitio_parser import VaspParser, BaderParser


class ChemicalEntity:

    def __init__(self, total_energy: float, net_magmom: float, structure: Structure, metadata: dict):
        self._total_energy = total_energy
        self._net_magmom = net_magmom
        self._structure = structure
        self._metadata = metadata if metadata is not None else {}

    def _apply_bader_data(self, bader_method_name: str, site_prop_name: str, data_name: str):
        """
        Internal helper method to fetch, validate, and apply Bader data to the structure.

        Args:
            bader_method_name (str): The name of the method to call on BaderParser.
            site_prop_name (str): The name of the property to add to the structure sites.
            data_name (str): A descriptive name of the data for warning messages.

        Returns:
            None
        """
        bader = BaderParser(directory=self._metadata.get("source_directory"))
        bader_data = getattr(bader, bader_method_name)()
        structure = getattr(self, "_structure", None)

        if not isinstance(structure, Structure):
            warnings.warn(
                "structure attribute is not initialized as a pymatgen Structure object. "
                f"'{site_prop_name}' property will not be added.",
                category=UserWarning,
                stacklevel=2,
            )
            return

        if bader_data is None:
            warnings.warn(
                f"Bader {data_name} data could not be recovered. '{site_prop_name}' "
                "property will not be added to the Structure object.",
                category=UserWarning,
                stacklevel=2,
            )
            return

        if len(bader_data) != len(structure):
            warnings.warn(
                f"Number of Bader {data_name} ({len(bader_data)}) does not match "
                f"number of sites in Structure ({len(structure)}). '{site_prop_name}' "
                "property will not be added.",
                category=UserWarning,
                stacklevel=2,
            )
            return

        structure.add_site_property(site_prop_name, bader_data)

    def load_bader_charges(self):
        """
        Loads Bader charges from the source directory and assigns them to the 
        structure.

        Returns:
            None
        """
        self._apply_bader_data("get_charges", "bader_charge", "charge")

    def load_bader_magmoms(self):
        """
        Loads Bader magnetic moments from the source directory and assigns them 
        to the structure.

        Returns:
            None
        """
        self._apply_bader_data("get_magmoms", "bader_magmom", "magnetic moment")

    @classmethod
    def from_vasp(cls, directory: str | Path, load_bader: bool = True):
        """
        Constructs an instance of the class from a VASP calculation directory.

        This factory method extracts the total energy, net magnetic moment, and 
        geometric structure from the VASP output files located in the specified 
        directory using 'VaspParser'. It initializes the object and stores the 
        source directory and software origin in its metadata.

        Args:
            directory (str | Path): The path to the directory containing the VASP 
                calculation output files (e.g., OUTCAR, vasprun.xml).
            load_bader (bool, optional): If True, attempts to automatically load 
                Bader charges and magnetic moments from the calculation directory 
                after initialization. Defaults to True.

        Returns:
            cls: An initialized instance of the class containing the parsed VASP data.

        Raises:
            ValueError: If neither the total energy nor the geometric structure 
                could be extracted from the provided directory, making initialization 
                impossible.
        """
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
            total_energy = total_energy,
            net_magmom = net_magmom,
            structure = structure,
            metadata={"source_directory": str(directory), "software": "VASP"}
        )

        if load_bader:
            instance.load_bader_charges()
            instance.load_bader_magmoms()

        return instance