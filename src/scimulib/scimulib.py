import logging
from pathlib import Path
from functools import cached_property
from typing import Any, Literal, Self
from collections.abc import Mapping, Sequence

import numpy as np

from pymatgen.core import Molecule, Structure

from scimulib.import_utils import check_pymatgen_imports
from scimulib.exceptions import (
    VaspExtractionError, BaderExtractionError, 
    MissingDataError, MissingGeometryError, NonPeriodicGeometryError, MissingSitePropertyError,
    MissingReferenceError, StoichiometryError
)
from scimulib.visualization import AxesMode, LabelMode, RenderStyle, build_py3dmol_view
from scimulib.abinitio_parser import VaspParser, BaderParser
from scimulib.bader_calc import (
    calculate_bader_net_charges, sum_bader_net_charges, sum_bader_magnetic_moments
)
from scimulib.energy_calc import calculate_cohesive_energy


class ChemicalEntity:
    """
    A class representing a chemical entity, encapsulating its name, total energy, 
    net magnetic moment, and geometric structure. It also stores any additional 
    site properties extracted from the calculation.
    """

    def __init__(
            self, 
            name: str, 
            total_energy: float | None = None, 
            net_magmom: float | None = None, 
            geometry: Structure | Molecule | None = None,
            valence_electrons: dict[str, float] = {}
    ):
        """
        Initializes a ChemicalEntity instance.

        Args:
            name (str): The name of the chemical entity.
            total_energy (float | None): The total energy of the chemical entity.
            net_magmom (float | None): The net magnetic moment of the chemical entity.
            geometry (Structure | Molecule | None): The geometric structure of the chemical entity.
            valence_electrons (dict[str, float], optional): The valence electrons of each element in the 
                chemical entity. Defaults to an empty dictionary.

        Raises:
            ValueError: If both total_energy and geometry are None.
        """
        check_pymatgen_imports()

        self.name = name
        self.total_energy = float(total_energy)
        self.net_magmom = float(net_magmom)
        self.geometry = geometry
        self.valence_electrons = valence_electrons

        if (
            self.total_energy is None 
            and self._structure is None 
            and self.net_magmom is None
        ):
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

    # In this part, we define property getters and setters for the chemical entity's attributes.
    @property
    def name(self) -> str:
        """Returns the name of the chemical entity."""
        if self._name is None:
            raise MissingDataError(
                f"Name is not available on this "
                f"{self.__class__.__name__} instance."
            )
        return self._name

    @name.setter
    def name(self, value: str):
        """Sets the name of the chemical entity."""
        if not isinstance(value, str) or not value.strip():
            raise TypeError("Name must be a non-empty string.")
        self._name = value

    @property
    def total_energy(self) -> float:
        """
        Returns the total energy of the chemical entity or raises 
        MissingDataError if not available.
        """
        if self._total_energy is None:
            raise MissingDataError(
                f"Energy is not available on this "
                f"{self.__class__.__name__} instance."
            )
        return self._total_energy

    @total_energy.setter
    def total_energy(self, value: float | None):
        """Sets the total energy of the chemical entity."""
        if value is not None and not isinstance(value, (float, int)):
            raise TypeError("Total energy must be a float or None.")
        self._total_energy = value

    @property
    def net_magmom(self) -> float:
        """
        Returns the net magnetic moment of the chemical entity or raises 
        MissingDataError if not available.
        """
        if self._net_magmom is None:
            raise MissingDataError(
                f"Net magnetic moment is not available on this "
                f"{self.__class__.__name__} instance."
            )
        return self._net_magmom

    @net_magmom.setter
    def net_magmom(self, value: float | None):
        """Sets the net magnetic moment of the chemical entity."""
        if value is not None and not isinstance(value, (float, int)):
            raise TypeError("Net magnetic moment must be a float or None.")
        self._net_magmom = value

    @cached_property
    def geometry(self) -> Structure | Molecule:
        """
        Returns a clean geometric structure 
        (only species, coords, and lattice if applicable).
        """      
        if isinstance(self._structure, Structure):
            return Structure(
                self._structure.lattice.copy(),
                self._structure.species,
                self._structure.frac_coords.copy(),
            )

        if isinstance(self._structure, Molecule):
            return Molecule(
                self._structure.species,
                self._structure.cart_coords.copy(),
            )

        raise MissingGeometryError(
            f"Geometry is not available on this "
            f"{self.__class__.__name__} instance."
        )

    @geometry.setter
    def geometry(self, object: Structure | Molecule | None):
        """Sets the geometric structure of the chemical entity."""
        if object is not None and not isinstance(object, (Structure, Molecule)):
            raise TypeError("Geometry must be a Structure, Molecule, or None.")
        self._structure = object

    @property
    def valence_electrons(self) -> dict[str, float]:
        """
        Returns the valence electrons of each element in the chemical entity, 
        if available.
        """
        return self._valence_electrons.copy()

    @valence_electrons.setter
    def valence_electrons(self, value: dict[str, float] | None):
        """Sets the valence electrons of each element in the chemical entity."""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Valence electrons must be a dictionary or None.")
        self._valence_electrons = value if value is not None else {}

    # Here we define property getters for geometric and compositional attributes 
    # (i.e. completely derived from the underlying geometry).
    @property
    def _require_geometry(self) -> Structure | Molecule:
        """Returns the underlying geometry or raises MissingGeometryError."""
        if self._structure is None:
            raise MissingGeometryError(
                f"Cannot access to geometric properties and methods: no geometry is set on "
                f"this {self.__class__.__name__} instance."
            )
        return self._structure

    @property
    def _require_periodic(self) -> Structure:
        """Returns the underlying periodic geometry or raises NonPeriodicGeometryError."""
        geometry = self._require_geometry
        if not isinstance(geometry, Structure):
            raise NonPeriodicGeometryError(
                f"Periodic property or method is not available: {self.__class__.__name__} "
                f"holds a non-periodic '{type(geometry).__name__}' object."
            )
        return geometry

    def _require_site_property(self, property_name: str) -> np.ndarray:
        """Ensures that the specified site property exists and returns it as a numpy array."""
        geometry = self._require_geometry
        if property_name not in geometry.site_properties:
            raise MissingSitePropertyError(
                f"Site property '{property_name}' is not available on this "
                f"{self.__class__.__name__} instance."
            )
        return np.array(geometry.site_properties[property_name], dtype=float)

    def _require_calculation_references(self, entity_type: str, *entities: Self, **named_entities: Self | float):
        calculation_references = {}
        # 1. To process positional cluster references
        for entity in entities:
            if not isinstance(entity, type(self)):
                raise TypeError(
                    f"Positional arguments must be {self.__class__.__name__} instances. "
                    "Pass numeric energies using keywords (e.g., Au=-3.2)."
                )

            comp = entity.composition
            if (entity_type == "atom" or entity_type == "cluster") and len(comp) != 1:
                raise StoichiometryError(
                    f"Positional {entity_type} references must represent a single "
                    f"specie, got: {comp.keys()}."
                )
            if (entity_type == "cluster" or entity_type == "supported") and next(iter(comp.values())) != len(self._require_geometry):
                raise StoichiometryError(
                    f"Positional {entity_type} references must coincide with the number "
                    f"of atoms of {self.__class__.__name__} instance. "
                    f"Got: {next(iter(comp.values()))} atoms, expected: {len(self._require_geometry)}."
                )
            if entity_type == "atom" and next(iter(comp.values())) != 1:
                raise StoichiometryError(
                    f"Positional {entity_type} references must represent a single atom. "
                    f"Got: {next(iter(comp.values()))} atoms."
                )

            symbol = next(iter(comp.keys()))
            if symbol in calculation_references:
                raise ValueError(f"Duplicate {entity_type} reference provided for '{symbol}'.")

            calculation_references[symbol] = entity.total_energy

        # 2. To process explicit keyword cluster references
        for symbol, ref in named_entities.items():
            if symbol in calculation_references:
                raise ValueError(f"Duplicate {entity_type} reference provided for '{symbol}'.")

            calculation_references[symbol] = (
                ref.total_energy if isinstance(ref, type(self)) else float(ref)
            )

        # 3. To check that all species in the structure are covered
        required_species = set(self.symbols)
        missing_species = required_species - calculation_references.keys()

        if missing_species:
            raise MissingReferenceError(
                f"{entity_type.capitalize()} energies must be provided for all species in the structure. "
                f"Missing species: {sorted(missing_species)}"
            )

        filtered_energies = {
            symbol: calculation_references[symbol] for symbol in required_species
        }
        return filtered_energies
    
    @property
    def fractional_coords(self) -> np.ndarray:
        """
        Returns the fractional coordinates of the structure as a
        numpy array, if available.
        """
        return self._require_periodic.frac_coords.copy()

    @property
    def cartesian_coords(self) -> np.ndarray:
        """
        Returns the Cartesian coordinates of the structure as a 
        numpy array, if available.
        """
        return self._require_geometry.cart_coords.copy()

    @cached_property
    def symbols(self) -> tuple[str]:
        """Returns the chemical symbols of the atoms in the structure, if available."""
        return self._require_geometry.symbol_set

    @cached_property
    def atomic_symbols(self) -> list[str]:
        """Returns the atomic symbols of the chemical entity, if available."""
        return [site.species_string for site in self._require_geometry]

    @cached_property
    def composition(self) -> dict[str, float]:
        """Returns the composition of the chemical entity, if available."""
        return self._require_geometry.composition.get_el_amt_dict()

    @cached_property
    def atomic_fraction(self) -> dict[str, float]:
        """Returns the atomic fraction of each element in the chemical entity, if available."""
        comp = self.composition
        total = sum(comp.values())
        return {el: amt / total for el, amt in comp.items()}
    
    # Here we define property getters for Bader analysis attributes.
    @property
    def bader_charges(self) -> np.ndarray:
        """Returns the Bader charges as a numpy array, if available."""
        return self._require_site_property("bader_charge")

    @cached_property
    def bader_net_charges(self) -> np.ndarray:
        """Returns the net Bader charges as a numpy array, if available."""
        return calculate_bader_net_charges(
            species_list=self.atomic_symbols,
            bader_charges=self.bader_charges,
            valence_electrons=self.valence_electrons
        )

    @cached_property
    def sum_bader_charges(self) -> dict[str, float]:
        """Sums the net Bader charges per species."""
        return sum_bader_net_charges(
            species_list=self.atomic_symbols,
            net_bader_charges=self.bader_net_charges
        )

    @property
    def bader_magnetic_moments(self) -> np.ndarray:
        """Returns the Bader magnetic moments as a numpy array, if available."""
        return self._require_site_property("bader_magmom")

    @cached_property
    def sum_bader_magmoms(self) -> dict[str, float]:
        """Sums the Bader magnetic moments per species."""
        return sum_bader_magnetic_moments(
            species_list=self.atomic_symbols,
            bader_magnetic_moments=self.bader_magnetic_moments,
            total_magnetic_moment=self.net_magmom
        )

    # The following methods correspond to energy calculations and related properties
    def cohesive_energy(
        self,
        *atoms: Self,
        **named_atoms: Self | float,
    ) -> float:
        """Calculates the cohesive energy from isolated atomic references.

        Atomic references can be passed positionally as single-atom `Self` 
        (ChemicalEntity and its subclasses) instances (whose element symbol is 
        inferred automatically) and/or as keyword arguments mapping element 
        symbols to `Self` instances or float energies.

        Args:
            *atoms: Isolated single-atom `Self` instances.
            **named_atoms: Keyword arguments mapping atomic symbols (e.g., 'Au', 'O')
                to either a `Self` instance or its total energy as a float.

        Returns:
            float: The cohesive energy of the structure in eV.

        Raises:
            TypeError: If a positional argument is not a `Self` instance.
            StoichiometryError: If a positional `Self` does not represent a
                single isolated atom (1 species and 1 atom).
            ValueError: If duplicate references are provided for the same species.
            MissingReferenceError: If atomic references are missing for any species
                present in the structure.
        """
        atomic_energies = {}

        # 1. Process positional atomic references
        for atom in atoms:
            if not isinstance(atom, type(self)):
                raise TypeError(
                    f"Positional arguments must be {self.__class__.__name__} instances. "
                    "Pass numeric energies using keywords (e.g., Au=-3.2)."
                )

            comp = atom.composition
            if len(comp) != 1 or next(iter(comp.values())) != 1:
                raise StoichiometryError(
                    "Positional atomic references must represent a single isolated atom "
                    f"(1 species, 1 atom), got composition: {comp}."
                )

            symbol = next(iter(comp.keys()))
            if symbol in atomic_energies:
                raise ValueError(f"Duplicate atomic reference provided for '{symbol}'.")

            atomic_energies[symbol] = atom.total_energy

        # 2. To process explicit keyword atomic references
        for symbol, ref in named_atoms.items():
            if symbol in atomic_energies:
                raise ValueError(f"Duplicate atomic reference provided for '{symbol}'.")

            atomic_energies[symbol] = (
                ref.total_energy if isinstance(ref, type(self)) else float(ref)
            )

        # 3. To check that all species in the structure are covered
        required_species = set(self.symbols)
        missing_species = required_species - atomic_energies.keys()

        if missing_species:
            raise MissingReferenceError(
                "Atomic energies must be provided for all species in the structure. "
                f"Missing species: {sorted(missing_species)}"
            )

        filtered_energies = {
            symbol: atomic_energies[symbol] for symbol in required_species
        }

        return calculate_cohesive_energy(
            total_energy=self.total_energy,
            atomic_energies=filtered_energies,
            atomic_counts=self.composition,
        )

    # The following method visualizes the structure using py3Dmol.
    def visualize(
        self,
        *,
        width: int = 600,
        height: int = 450,
        style: RenderStyle = "ball_and_stick",
        repeat: tuple[int, int, int] = (1, 1, 1),
        show_unit_cell: bool = True,
        element_colors: Mapping[str, str] | None = None,
        axes: AxesMode | None = "cartesian",
        axes_length: float = 2.0,
        hover_labels: bool | LabelMode = True,
        index_base: Literal[0, 1] = 0,
        atom_property: str | Sequence[float] | np.ndarray | None = None,
        property_name: str | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric_clim: bool = False,
        value_format: str = ".2f",
    ) -> Any:
        """Creates an interactive py3Dmol 3D viewer for the entity's structure.

        Args:
            width: Viewer canvas width in pixels. Defaults to 600.
            height: Viewer canvas height in pixels. Defaults to 450.
            style: Molecular rendering preset (`'ball_and_stick'`, `'spacefill'`,
                or `'stick'`). Defaults to `'ball_and_stick'`.
            repeat: Supercell replication factors `(nx, ny, nz)` applied when the
                underlying geometry is a periodic `Structure`. Defaults to `(1, 1, 1)`.
            show_unit_cell: Whether to draw the unit cell bounding box for periodic
                `Structure` objects. Defaults to True.
            element_colors: Optional mapping from element symbols to CSS or hex
                color strings that overrides `DEFAULT_ELEMENT_COLORS`. Ignored if
                `atom_property` is provided.
            axes: Reference axes triad to display near the structure corner
                (`'cartesian'`, `'cell'`, or `None` to hide). Defaults to `'cartesian'`.
            axes_length: Visual length of the reference axes arrows in Angstroms.
                Defaults to 2.0.
            hover_labels: Controls interactive mouseover tooltips on atoms. Defaults
                to True.
            index_base: Starting index for site numbering (`0` for Python/Pymatgen,
                `1` for VASP/VESTA). Defaults to 0.
            atom_property: Optional per-atom property to map onto atom colors and
                display in hover tooltips. Pass either a `site_properties` key string
                (e.g., `'bader_net_charges'`, `'magmom'`) or a 1D sequence/NumPy array
                of values matching the site count.
            property_name: Display title for the right-side colorbar and tooltip
                prefix. Defaults to `atom_property` when a key string is passed.
            cmap: Matplotlib colormap name used when `atom_property` is set.
                Defaults to `'viridis'`.
            vmin: Minimum value for color scale normalization.
            vmax: Maximum value for color scale normalization.
            symmetric_clim: If True and `vmin`/`vmax` are None, centers color limits
                symmetrically around zero (`[-max(|v|), +max(|v|)]`).
            value_format: Format specifier for displayed property values (e.g.,
                `'.2f'`, `'+.3f'`). Defaults to `'.2f'`.

        Returns:
            A `ColorbarView` when `atom_property` is provided, or a `py3Dmol.view`
            instance otherwise.
        """
        prop_values: Sequence[float] | np.ndarray | None = None
        resolved_name = property_name

        if isinstance(atom_property, str):
            prop_values = self._require_site_property(atom_property)
            if resolved_name is None:
                resolved_name = atom_property
        elif atom_property is not None:
            prop_values = atom_property

        active_colors = (
            self.DEFAULT_ELEMENT_COLORS
            if element_colors is None
            else self.DEFAULT_ELEMENT_COLORS | dict(element_colors)
        )

        return build_py3dmol_view(
            self._require_structure,
            width=width,
            height=height,
            style=style,
            repeat=repeat,
            show_unit_cell=show_unit_cell,
            element_colors=active_colors,
            axes=axes,
            axes_length=axes_length,
            hover_labels=hover_labels,
            index_base=index_base,
            property_values=prop_values,
            property_name=resolved_name,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            symmetric_clim=symmetric_clim,
            value_format=value_format,
        )

    # Factory methods for constructing instances from external data sources
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

        try:
            valence_electrons = vasp_parser.get_valence_electrons()
        except VaspExtractionError as e:
            logging.warning(
                f"{cls.__name__} '{entity_name}': {e}"
            )
            valence_electrons = {}

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
            geometry=structure,
            valence_electrons=valence_electrons
        )


class Nanocluster(ChemicalEntity):
    """
    Class representing a nanocluster, which represents a small aggregate of atoms. 
    Typically consists of a few to several hundred atoms. An example of a nanocluster 
    could be a nanoparticle model.
    """
    def __init__(
            self, 
            name: str, 
            total_energy: float | None = None, 
            net_magmom: float | None = None, 
            geometry: Molecule | None = None, 
            valence_electrons: dict[str, int] | None = None
    ):
        super().__init__(name=name, total_energy=total_energy, net_magmom=net_magmom,
                         geometry=geometry, valence_electrons=valence_electrons)

    @ChemicalEntity.geometry.setter
    def geometry(self, object: Structure | Molecule | None):
        if isinstance(object, Structure):
            object = Molecule(
                species=object.species,
                coords=object.cart_coords,
                site_properties=object.site_properties
            )
        elif object is not None and not isinstance(object, Molecule):
            raise TypeError("Geometry must be a Structure, Molecule, or None.")
        
        ChemicalEntity.geometry.fset(self, object)

    def mixing_energy(self, *clusters: Self, **named_clusters: Self | float) -> float:
        cluster_energies = {}

        # 1. To process positional cluster references
        for cluster in clusters:
            if not isinstance(cluster, type(self)):
                raise TypeError(
                    f"Positional arguments must be {self.__class__.__name__} instances. "
                    "Pass numeric energies using keywords (e.g., Au=-3.2)."
                )

            comp = cluster.composition
            if len(comp) != 1:
                raise StoichiometryError(
                    "Positional cluster references must represent a single "
                    f"specie, got: {comp.keys()}."
                )
            if next(iter(comp.values())) != len(self._require_geometry):
                raise StoichiometryError(
                    "Positional cluster references must coincide with the number "
                    f"of atoms of {self.__class__.__name__} instance. "
                    f"Got: {next(iter(comp.values()))} atoms, expected: {len(self._require_geometry)}."
                )

            symbol = next(iter(comp.keys()))
            if symbol in cluster_energies:
                raise ValueError(f"Duplicate cluster reference provided for '{symbol}'.")

            cluster_energies[symbol] = cluster.total_energy

        # 2. To process explicit keyword cluster references
        for symbol, ref in named_clusters.items():
            if symbol in cluster_energies:
                raise ValueError(f"Duplicate cluster reference provided for '{symbol}'.")

            cluster_energies[symbol] = (
                ref.total_energy if isinstance(ref, type(self)) else float(ref)
            )

        # 3. To check that all species in the structure are covered
        required_species = set(self.symbols)
        missing_species = required_species - cluster_energies.keys()

        if missing_species:
            raise MissingReferenceError(
                "Cluster energies must be provided for all species in the structure. "
                f"Missing species: {sorted(missing_species)}"
            )

        filtered_energies = {
            symbol: cluster_energies[symbol] for symbol in required_species
        }

        return calculate_cohesive_energy(
            total_energy=self.total_energy,
            atomic_energies=filtered_energies,
            atomic_counts=self.composition,
        )