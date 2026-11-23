import logging
from pathlib import Path

import numpy as np

from pymatgen.core import Molecule, Structure

from scimulib.import_utils import check_pymatgen_imports
from scimulib.exceptions import (
    VaspExtractionError, BaderExtractionError, SizeMismatchError, InvalidValenceValueError, 
    MissingValenceElementError, ChargeNeutralityError, MagneticMomentMismatchError
)
from scimulib.abinitio_parser import VaspParser, BaderParser
from scimulib.bader_calc import (
    calculate_bader_net_charges, sum_bader_net_charges, sum_bader_magnetic_moments
)


class ChemicalEntity:
    """
    A class representing a chemical entity, encapsulating its name, total energy, 
    net magnetic moment, and geometric structure. It also stores any additional 
    site properties extracted from the calculation.
    """

    # Dictionary mapping elements to their default valence electron counts 
    # as it appears in the VASP POTCAR files.
    valence_electrons = {
        "Au": 11, "Fe": 8, "Os": 8, "Ru": 14, "Ti": 12, "O":  6,
        "Co": 9,  "Rh": 15, "Ir": 9, "Ni": 10, "Pd": 10, "Pt": 10,
        "Cu": 11, "Ag": 11, "Al": 3
    }

    def __init__(
            self, name: str, 
            total_energy: float | None, 
            net_magmom: float | None, 
            geometry: Structure | Molecule | None
    ):
        """
        Initializes a ChemicalEntity instance.

        Args:
            name (str): The name of the chemical entity.
            total_energy (float | None): The total energy of the chemical entity.
            net_magmom (float | None): The net magnetic moment of the chemical entity.
            geometry (Structure | Molecule | None): The geometric structure of the chemical entity.

        Raises:
            ValueError: If both total_energy and geometry are None.
        """
        check_pymatgen_imports()

        self.name = name
        self.total_energy = total_energy
        self.net_magmom = net_magmom
        self.geometry = geometry

        if self.total_energy is None and self.geometry is None and self.net_magmom is None:
            raise ValueError(
                f"Cannot create a {self.__class__.__name__} instance. "
                "At least one property must be provided."
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__} (name={self.name}, "
            f"formula={self._structure.composition.formula if self._structure else 'N/A'}, "
            f"total_energy={self.total_energy if self.total_energy is not None else 'N/A'}"
        )

    @property
    def name(self) -> str:
        """Returns the name of the chemical entity."""
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the name of the chemical entity."""
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Name must be a non-empty string.")
        self._name = value

    @property
    def total_energy(self) -> float | None:
        """Returns the total energy of the chemical entity."""
        return self._total_energy

    @total_energy.setter
    def total_energy(self, value: float | None):
        """Sets the total energy of the chemical entity."""
        if value is not None and not isinstance(value, (float, int)):
            raise TypeError("Total energy must be a float or None.")
        self._total_energy = value

    @property
    def net_magmom(self) -> float | None:
        """Returns the net magnetic moment of the chemical entity."""
        return self._net_magmom

    @net_magmom.setter
    def net_magmom(self, value: float | None):
        """Sets the net magnetic moment of the chemical entity."""
        if value is not None and not isinstance(value, (float, int)):
            raise TypeError("Net magnetic moment must be a float or None.")
        self._net_magmom = value

    @property
    def geometry(self) -> Structure | Molecule | None:
        """
        Returns a clean geometric structure 
        (only species, coords, and lattice if applicable).
        """           
        if isinstance(self._structure, Structure):
            return Structure(
                self._structure.lattice.copy(),
                self._structure.species.copy(),
                self._structure.frac_coords.copy(),
            )

        if isinstance(self._structure, Molecule):
            return Molecule(
                self._structure.species.copy(),
                self._structure.cart_coords.copy(),
            )

        return None

    @geometry.setter
    def geometry(self, object: Structure | Molecule | None):
        """Sets the geometric structure of the chemical entity."""
        if object is not None and not isinstance(object, (Structure, Molecule)):
            raise TypeError("Geometry must be a Structure, Molecule, or None.")
        self._structure = object

    @property
    def fractional_coords(self) -> np.ndarray | None:
        """
        Returns the fractional coordinates of the structure as a
        numpy array, if available.
        """
        if self._structure is None:
            logging.warning(
                "Fractional coordinates are not available since the structure is not "
                f"set. Create a new {self.__class__.__name__} with a structure to "
                "access to the attribute."
            )
            return None
        
        if isinstance(self._structure, Molecule):
            logging.warning(
                "Molecule objects' coordinates cannot be "
                "converted to fractional coordinates."
            )
            return None

        return self._structure.frac_coords.copy()

    @property
    def cartesian_coords(self) -> np.ndarray | None:
        """
        Returns the Cartesian coordinates of the structure as a 
        numpy array, if available.
        """
        if self._structure is None:
            logging.warning(
                "Cartesian coordinates are not available since the structure is not set. "
                f"Create a new {self.__class__.__name__} with a structure to access "
                "the attribute."
            )
            return None

        return self._structure.cart_coords.copy()

    @property
    def composition(self) -> dict[str, float] | None:
        """Returns the composition of the chemical entity, if available."""
        if self._structure is None:
            logging.warning(
                "Composition is not available since the structure is not set. "
                f"Create a new {self.__class__.__name__} with a structure to access "
                "the attribute."
            )
            return None
        return self._structure.composition.get_el_amt_dict()

    @property
    def atomic_fraction(self) -> dict[str, float] | None:
        """Returns the atomic fraction of each element in the chemical entity, if available."""
        comp = self.composition
        if comp is None:
            return None
        total = sum(comp.values())
        return {el: amt / total for el, amt in comp.items()}
    
    @property
    def bader_charges(self) -> np.ndarray | None:
        """Returns the Bader charges as a numpy array, if available."""
        if self._structure is None:
            logging.warning(
                "Bader charges are not available since the structure is not set. "
                f"Create a new {self.__class__.__name__} with a structure to access "
                "the attribute."
            )
            return None
        if "bader_charge" in self._structure.site_properties:
            return np.array(self._structure.site_properties["bader_charge"])
        
        logging.warning("Bader charges are not available.")
        return None

    def bader_net_charges(self, valence_electrons: dict[str, int] = None) -> float | None:
        """
        Calculates the net Bader charges based on the provided valence electron counts.

        Args:
            valence_electrons (dict[str, int], optional): A dictionary mapping element symbols
                to their respective valence electron counts. If not provided, the method
                will use the `valence_electrons` attribute of the instance.

        Returns:
            float: The net Bader charge of the system.
            None: If the calculation cannot be performed.
        """
        if "bader_net_charges" in self._structure.site_properties:
            return np.array(self._structure.site_properties["bader_net_charges"])

        if self._structure is None:
            logging.warning(
                "Net Bader charges are not available since the structure is not set. "
                f"Create a new {self.__class__.__name__} with a structure to access "
                "the attribute."
            )
            return None

        bader_charges = self.bader_charges
        if bader_charges is None:
            logging.warning(
                "Net Bader charges are not available since Bader charges are not set."
            )
            return None
        
        if valence_electrons is not None:
            val_electrons = self.valence_electrons | valence_electrons
        else:
            val_electrons = self.valence_electrons

        try:
            bader_net_charges = calculate_bader_net_charges(
                geometry=self._structure,
                bader_charges=bader_charges,
                valence_electrons=val_electrons
            )
        except (
            SizeMismatchError, InvalidValenceValueError, MissingValenceElementError, 
            ValueError, TypeError
        ) as e:
            logging.warning(f"Failed to calculate net Bader charges: {e}")
            return None

        self._structure.add_site_property("bader_net_charges", bader_net_charges)
        return bader_net_charges


    def sum_bader_charges(self, valence_electrons: dict[str, int] = None) -> dict[str, float] | None:
        """
        Sums the net Bader charges per species.

        Args:
            valence_electrons (dict[str, int], optional): A dictionary mapping element symbols
                to their respective valence electron counts. If not provided, the method
                will use the `valence_electrons` attribute of the instance.

        Returns:
            dict[str, float]: Summed net Bader charges grouped by species, 
                or None if Bader charges are not available.
        """
        if self._structure is None:
            logging.warning(
                "Net Bader charges are not available since the structure is not set. "
                f"Create a new {self.__class__.__name__} with a structure to access "
                "the attribute."
            )
            return None

        bader_charges = self.bader_charges
        if bader_charges is None:
            logging.warning(
                "Net Bader charges are not available since Bader charges are not set."
            )
            return None
    
        try:
            return sum_bader_net_charges(
                geometry=self._structure,
                net_bader_charges=self.bader_net_charges(
                    valence_electrons=valence_electrons 
                    if valence_electrons is not None else None)
            ) if self.bader_charges is not None else None
        except (
            SizeMismatchError, InvalidValenceValueError, MissingValenceElementError, 
            ChargeNeutralityError, ValueError, TypeError
        ) as e:
            logging.warning(f"Failed to sum Bader charges: {e}")
            return None

    @property
    def bader_magnetic_moments(self) -> np.ndarray | None:
        """Returns the Bader magnetic moments as a numpy array, if available."""
        if self._structure is None:
            logging.warning(
                "Bader magnetic moments are not available since the structure is not set. "
                f"Create a new {self.__class__.__name__} with a structure to access "
                "the attribute."
            )
            return None
        if "bader_magmom" in self._structure.site_properties:
            return np.array(self._structure.site_properties["bader_magmom"])
        
        logging.warning("Bader magnetic moments are not available.")
        return None

    @property
    def sum_bader_magmoms(self) -> dict[str, float] | None:
        """Sums the Bader magnetic moments per species."""
        if self._structure is None:
            logging.warning(
                "Bader magnetic moments are not available since the structure is not set. "
                f"Create a new {self.__class__.__name__} with a structure to access "
                "the attribute."
            )
            return None

        bader_magmoms = self.bader_magnetic_moments
        if bader_magmoms is None:
            logging.warning(
                "Bader magnetic moments are not available since Bader magnetic moments are not set."
            )
            return None

        try:
            return sum_bader_magnetic_moments(
                geometry=self._structure,
                bader_magnetic_moments=self.bader_magnetic_moments,
                total_magnetic_moment=self.net_magmom,
            ) if self.bader_magnetic_moments is not None else None
        except (
            SizeMismatchError, InvalidValenceValueError, MissingValenceElementError, 
            MagneticMomentMismatchError, ValueError, TypeError
        ) as e:
            logging.warning(f"Failed to sum Bader magnetic moments: {e}")
            return None

    @classmethod
    def from_vasp(cls, directory: str | Path, name: str = None):
        """
        Constructs an instance of the class from a VASP calculation directory.

        This factory method extracts the total energy, net magnetic moment, and 
        geometric structure from the VASP output files located in the specified 
        directory using `VaspParser`. Aditionally, it attempts to extract Bader 
        charges and magnetic moments using `BaderParser` and adds them as site 
        properties to the structure if available. A name for the chemical entity 
        can be provided; if not, the directory name will be used as the default 
        name.

        Args:
            directory (str | Path): The path to the directory containing the VASP 
                calculation output files (e.g., OUTCAR, vasprun.xml).
            name (str, optional): An optional name for the chemical entity. If not
                provided, the name will default to the directory name.

        Returns:
            cls: An initialized instance of the class containing the parsed VASP data.

        Raises:
            ValueError: If neither the total energy nor the geometric structure 
                could be extracted from the provided directory, making initialization 
                impossible.

        """
        entity_name = name if name is not None else directory.name

        vasp_parser = VaspParser(directory=directory)

        try:
            total_energy = vasp_parser.get_total_energy()
        except VaspExtractionError as e:
            logging.warning(
                f"{cls.__name__} '{entity_name}': {e}"
            )
            total_energy = None

        try:
            net_magmom = vasp_parser.get_net_magmom()
        except VaspExtractionError as e:
            logging.warning(
                f"{cls.__name__} '{entity_name}': {e}"
            )
            net_magmom = None

        try:
            structure = vasp_parser.get_structure()
        except VaspExtractionError as e:
            logging.warning(
                f"{cls.__name__} '{entity_name}': {e}"
            )
            structure = None

        bader_parser = BaderParser(
            directory=directory, 
            expected_atoms=len(structure) if structure is not None else None
        )
        try:
            bader_charges = bader_parser.get_charges()
            if structure is not None:
                structure.add_site_property("bader_charge", bader_charges)
            else:
                logging.warning(
                    f"{cls.__name__} '{entity_name}': Structure is None, cannot add Bader charges."
                )
        except BaderExtractionError as e:
            logging.warning(
                f"{cls.__name__} '{entity_name}': {e}"
            )

        try:
            bader_magmom = bader_parser.get_magnetic_moments()
            if structure is not None:
                structure.add_site_property("bader_magmom", bader_magmom)
            else:
                logging.warning(
                    f"{cls.__name__} '{entity_name}': Structure is None, cannot add Bader magnetic moments."
                )
        except BaderExtractionError as e:
            logging.warning(
                f"{cls.__name__} '{entity_name}': {e}"
            )

        return cls(
            name=entity_name,
            total_energy=total_energy,
            net_magmom=net_magmom,
            geometry=structure
        )