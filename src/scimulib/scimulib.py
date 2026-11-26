import logging
import math
from pathlib import Path
from functools import cached_property
from typing import Any, Literal, Self
from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

from pymatgen.core import Molecule, Structure
from pymatgen.analysis.molecule_matcher import MoleculeMatcher
from pymatgen.analysis.structure_matcher import StructureMatcher

from scimulib.validation_utils import ReferencePolicy, check_pymatgen_imports
from scimulib.exceptions import (
    VaspExtractionError, BaderExtractionError, 
    MissingDataError, MissingGeometryError, NonPeriodicGeometryError, MissingSitePropertyError,
    MissingReferenceError, StoichiometryError, MismatchedGeometryError
)
from scimulib.visualization import AxesMode, LabelMode, RenderStyle, build_py3dmol_view
from scimulib.abinitio_parser import VaspParser, BaderParser
from scimulib.bader_calc import (
    calculate_bader_net_charges, sum_bader_net_charges, sum_bader_magnetic_moments
)
from scimulib.energy_calc import calculate_cohesive_energy, calculate_ideal_cohesive_energy, calculate_mixing_energy, calculate_strain_mixing_energy, calculate_interact_mixing_energy, calculate_vacancy_formation_energy


_ATOM_POLICY = ReferencePolicy(label="atom", single_species=True, expected_atoms=1)
_CLUSTER_POLICY = ReferencePolicy(label="cluster", single_species=True, same_atom_count=True)
_CLUSTER_GEOMETRY_POLICY = ReferencePolicy(
    label="cluster_geometry", 
    single_species=True, 
    same_atom_count=True, 
    same_geometry=True
)
_SYSTEM_POLICY = ReferencePolicy(label="system", single_species=False, check_mass_balance=True)


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

        Attributes:
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

    # --- Factory Methods ---
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

    # --- Property Getters and Setters ---
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

    @property
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
        if not self._valence_electrons:
            raise MissingDataError(
                f"Valence electrons are not available on this "
                f"{self.__class__.__name__} instance."
            )
        return self._valence_electrons.copy()

    @valence_electrons.setter
    def valence_electrons(self, value: dict[str, float] | None):
        """Sets the valence electrons of each element in the chemical entity."""
        if value is not None and not isinstance(value, dict):
            raise TypeError("Valence electrons must be a dictionary or None.")
        self._valence_electrons = value if value is not None else {}

    # --- Geometric and Compositional Attributes ---
    # Here we define property getters for geometric and compositional attributes 
    # (i.e. completely derived from the underlying geometry).
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
    def atomic_fractions(self) -> dict[str, float]:
        """Returns the atomic fractions of each element in the chemical entity, if available."""
        comp = self.composition
        total = sum(comp.values())
        return {el: amt / total for el, amt in comp.items()}

    # --- Geometry Calculation Methods ---
    # In this section, we define methods that calculate derived properties from 
    # the geometry, such as distances, angles, and coordination numbers.
    def has_same_geometry(
        self,
        other: Self,
        ltol: float = 0.01,
        stol: float = 0.01,
        angle_tol: float = 1.0,
        mol_tol: float = 0.01,
        ignore_species: bool = False,
    ) -> bool:
        """
        Determines if the geometry of this structure is the same as another structure,
        within the specified tolerances. Can optionally ignore species.

        Args:
            other (Self): The other structure to compare with.
            ltol (float): Lattice tolerance for structure comparison.
            stol (float): Site tolerance for structure comparison.
            angle_tol (float): Angle tolerance for structure comparison.
            mol_tol (float): Tolerance for molecule comparison.
            ignore_species (bool): Whether to ignore species in the comparison.

        Returns:
            bool: True if the geometries are considered the same, False otherwise.

        Raises:
            TypeError: If the geometries are of incompatible or unsupported types.
        """
        geom_a = self._require_geometry
        geom_b = other._require_geometry

        if isinstance(geom_a, Structure) and isinstance(geom_b, Structure):
            matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
            if ignore_species:
                return bool(matcher.fit_anonymous(geom_a, geom_b))
            return bool(matcher.fit(geom_a, geom_b))

        if isinstance(geom_a, Molecule) and isinstance(geom_b, Molecule):
            if ignore_species:
                geom_a = geom_a.copy()
                geom_b = geom_b.copy()
                for site in geom_a:
                    site.species = "X"
                for site in geom_b:
                    site.species = "X"
            matcher = MoleculeMatcher(tolerance=mol_tol)
            return bool(matcher.fit(geom_a, geom_b))

        if type(geom_a) is not type(geom_b):
            raise TypeError(
                f"Cannot compare geometries of different types: "
                f"{type(geom_a).__name__} and {type(geom_b).__name__}."
            )

        raise TypeError(
            f"Unsupported geometry type: {type(geom_a).__name__}. "
            "Expected pymatgen Structure or Molecule."
        )

    # --- Bader Analysis Properties ---
    # Here we define property getters for Bader analysis attributes.
    @property
    def bader_charges(self) -> np.ndarray:
        """Returns the Bader charges as a numpy array, if available."""
        return self._require_site_property("bader_charge")

    @cached_property
    def bader_net_charges(self) -> np.ndarray:
        """Returns the net Bader charges as a numpy array, if available."""
        geometry = self._require_geometry
        if "net_bader_charge" in geometry.site_properties:
            return np.array(geometry.site_properties["net_bader_charge"])
        
        bader_net_charges = calculate_bader_net_charges(
            species_list=self.atomic_symbols,
            bader_charges=self.bader_charges,
            valence_electrons=self.valence_electrons
        )
        geometry.add_site_property("net_bader_charge", bader_net_charges)
        return np.array(bader_net_charges)

    @property
    def bader_magnetic_moments(self) -> np.ndarray:
        """Returns the Bader magnetic moments as a numpy array, if available."""
        return self._require_site_property("bader_magmom")

    # --- Summed Bader Analysis Properties ---
    # These properties provide the summed Bader charges and magnetic moments 
    # per species.
    @cached_property
    def sum_bader_charges(self) -> dict[str, float]:
        """Sums the net Bader charges per species."""
        return sum_bader_net_charges(
            species_list=self.atomic_symbols,
            net_bader_charges=self.bader_net_charges
        )

    @cached_property
    def sum_bader_magmoms(self) -> dict[str, float]:
        """Sums the Bader magnetic moments per species."""
        return sum_bader_magnetic_moments(
            species_list=self.atomic_symbols,
            bader_magnetic_moments=self.bader_magnetic_moments,
            total_magnetic_moment=self.net_magmom
        )

    # --- Energy Calculation Methods ---
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
        filtered_energies = self._generate_calculation_references(
            _ATOM_POLICY, 
            *atoms, 
            **named_atoms
        )

        return calculate_cohesive_energy(
            total_energy=self.total_energy,
            atomic_energies=filtered_energies,
            atomic_counts=self.composition,
        )

    # --- Visualization Methods ---
    # The following methods allow visualization of the structure and site 
    # properties using py3Dmol.
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
    ) -> Any:
        """
        Creates an interactive py3Dmol viewer of the entity's geometry.

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
                color strings that overrides the default Jmol colors.
            axes: Reference axes fixed in the viewer's lower-right corner
                (`'cartesian'`, `'cell'`, or `None` to hide). Defaults to `'cartesian'`.
            axes_length: Scale factor for the overlay arrows. Defaults to 2.0.
            hover_labels: Controls interactive mouseover tooltips on atoms. Defaults
                to True.
            index_base: Starting index for site numbering (`0` for Python/Pymatgen,
                `1` for VASP/VESTA). Defaults to 0.

        Returns:
            A `py3Dmol.view` instance.
        """
        return build_py3dmol_view(
            self._require_geometry,
            width=width,
            height=height,
            style=style,
            repeat=repeat,
            show_unit_cell=show_unit_cell,
            element_colors=element_colors,
            axes=axes,
            axes_length=axes_length,
            hover_labels=hover_labels,
            index_base=index_base,
        )

    def visualize_property(
        self,
        *,
        atom_property: str | Sequence[float] | np.ndarray,
        width: int = 600,
        height: int = 450,
        style: RenderStyle = "ball_and_stick",
        repeat: tuple[int, int, int] = (1, 1, 1),
        show_unit_cell: bool = True,
        axes: AxesMode | None = "cartesian",
        axes_length: float = 2.0,
        hover_labels: bool | LabelMode = True,
        index_base: Literal[0, 1] = 0,
        property_name: str | None = None,
        cmap: str = "viridis",
        vmin: float | None = None,
        vmax: float | None = None,
        symmetric_clim: bool = False,
        value_format: str = ".2f",
    ) -> Any:
        """Visualizes a per-atom property as colors with an accompanying colorbar.

        Args:
            atom_property: A `site_properties` key (for example, `'charge'`) or a
                1D sequence/NumPy array containing one value per site.
            width: Viewer canvas width in pixels. Defaults to 600.
            height: Viewer canvas height in pixels. Defaults to 450.
            style: Molecular rendering preset (`'ball_and_stick'`, `'spacefill'`,
                or `'stick'`). Defaults to `'ball_and_stick'`.
            repeat: Supercell replication factors for periodic structures.
                Defaults to `(1, 1, 1)`.
            show_unit_cell: Whether to draw the unit cell for periodic structures.
                Defaults to True.
            axes: Reference axes in the lower-right corner (`'cartesian'`, `'cell'`,
                or `None` to hide). Defaults to `'cartesian'`.
            axes_length: Scale factor for the overlay arrows. Defaults to 2.0.
            hover_labels: Controls interactive mouseover tooltips on atoms.
                Defaults to True.
            index_base: Starting index for site numbering, `0` or `1`. Defaults to 0.
            property_name: Colorbar title and tooltip prefix. Defaults to the
                `atom_property` name when a site property key is passed.
            cmap: Matplotlib colormap name. Defaults to `'viridis'`.
            vmin: Minimum value for color scale normalization.
            vmax: Maximum value for color scale normalization.
            symmetric_clim: If True and `vmin`/`vmax` are None, centers color limits
                symmetrically around zero.
            value_format: Format specifier for property values. Defaults to `'.2f'`.

        Returns:
            A `ColorbarView` containing the viewer and color scale.
        """
        if atom_property is None:
            raise ValueError("'atom_property' is required for property visualization.")
        if isinstance(atom_property, str):
            prop_values = self._require_site_property(atom_property)
            resolved_name = atom_property if property_name is None else property_name
        else:
            prop_values = atom_property
            resolved_name = property_name

        return build_py3dmol_view(
            self._require_geometry,
            width=width,
            height=height,
            style=style,
            repeat=repeat,
            show_unit_cell=show_unit_cell,
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

    # --- Private Helper Methods ---
    # This last section contains private helper methods, which includes:
    #   - Validation of geometry existence and periodicity
    #   - Validation of site properties
    #   - Validation of reference structures according to specified policies
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

    def _validate_reference(self, ref: ChemicalEntity, policy: ReferencePolicy):
        """
        Validates a reference ChemicalEntity against the specified ReferencePolicy.

        Args:
            ref (ChemicalEntity): The reference chemical entity to validate.
            policy (ReferencePolicy): The policy against which to validate the reference.
        
        Raises:
            StoichiometryError: If the reference violates any stoichiometric constraints.
            MismatchedGeometryError: If the reference geometry does not match the expected geometry.
            Any custom exceptions raised by the extra_validator, if provided.
        """
        comp = ref.composition
        total_atoms = sum(comp.values())

        if policy.single_species and len(comp) != 1:
            raise StoichiometryError(
                f"{policy.label.capitalize()} references must represent a single "
                f"species, got: {list(comp.keys())}."
            )

        if policy.expected_atoms is not None and total_atoms != policy.expected_atoms:
            raise StoichiometryError(
                f"{policy.label.capitalize()} references must have {policy.expected_atoms} "
                f"atom(s), got: {total_atoms}."
            )

        if policy.same_atom_count and total_atoms != len(self._require_geometry):
            raise StoichiometryError(
                f"{policy.label.capitalize()} references must match the number of atoms "
                f"({len(self._require_geometry)}), got: {total_atoms}."
            )

        if policy.same_geometry and not self.has_same_geometry(ref, ignore_species=True):
            raise MismatchedGeometryError(
                f"{policy.label.capitalize()} references must share the same geometry "
                f"as {self.name}."
            )

        if policy.extra_validator is not None:
            policy.extra_validator(self, ref)

    def _generate_calculation_references(
        self,
        policy: ReferencePolicy,
        *entities: ChemicalEntity,
        coefficients: Mapping[str, float] | None = None,
        **named_entities: ChemicalEntity | float,
    ) -> dict[str, float]:
        """
        Generates a dictionary of calculation references based on the provided 
        entities and policy.

        Args:
            policy (ReferencePolicy): The policy to apply for reference validation.
            *entities (ChemicalEntity): Positional chemical entity references.
            coefficients: Optional positive stoichiometric coefficients keyed by
                reference name. Every reference must have a coefficient. These
                weight compositions for mass balance, not the returned energies.
            **named_entities (ChemicalEntity | float): Keyword chemical entity 
                references or numeric energies.

        Returns:
            dict[str, float]: A dictionary mapping reference keys to their 
                corresponding energies.

        Raises:
            TypeError: If a positional argument is not a ChemicalEntity.
            ValueError: If there are duplicate reference keys or coefficients
                do not match the reference keys.
            StoichiometryError: If coefficients or the mass balance are invalid.
            MissingReferenceError: If required species are not covered by 
                the references.
        """
        calculation_references = {}
        entity_objects = {}
        covered_species = set()

        # 1. Process positional arguments
        for idx, entity in enumerate(entities):
            if not isinstance(entity, ChemicalEntity):
                raise TypeError(
                    f"Positional arguments must be ChemicalEntity instances. "
                    "Pass numeric energies using keywords."
                )

            self._validate_reference(entity, policy)
            covered_species.update(entity.composition.keys())

            key = (
                next(iter(entity.composition.keys()))
                if policy.single_species
                else f"{policy.label}_{idx}"
            )

            if key in calculation_references:
                raise ValueError(f"Duplicate {policy.label} reference provided for '{key}'.")
            calculation_references[key] = entity.total_energy
            entity_objects[key] = entity

        # 2. Process keyword arguments
        for key, ref in named_entities.items():
            if key in calculation_references:
                raise ValueError(f"Duplicate {policy.label} reference provided for '{key}'.")

            if isinstance(ref, ChemicalEntity):
                self._validate_reference(ref, policy)
                entity_objects[key] = ref
                covered_species.update(ref.composition.keys())
                calculation_references[key] = ref.total_energy
            else:
                calculation_references[key] = float(ref)
                if policy.single_species:
                    covered_species.add(key)

        if coefficients is not None:
            if coefficients.keys() != calculation_references.keys():
                raise ValueError("Stoichiometric coefficients must match the reference keys.")
            for key, coefficient in coefficients.items():
                if not math.isfinite(coefficient) or coefficient <= 0:
                    raise StoichiometryError(
                        f"Stoichiometric coefficient for '{key}' must be positive "
                        f"and finite, got {coefficient}."
                    )

        # 3. Verify that the species match self
        required_species = set(self.symbols)
        all_are_entities = len(entity_objects) == len(calculation_references)

        if policy.single_species or all_are_entities:
            missing_species = required_species - covered_species
            if missing_species:
                raise MissingReferenceError(
                    f"{policy.label.capitalize()} references must cover all species in "
                    f"{self.name}. Missing species: {sorted(missing_species)}."
                )

            # In fragment mode, there should also be no extra species not present in the system
            extra_species = covered_species - required_species
            if extra_species and not policy.single_species:
                raise StoichiometryError(
                    f"{policy.label.capitalize()} references contain species not present in "
                    f"{self.name}: {sorted(extra_species)}."
                )

        # 4. Validate mass balance (when all references are entity objects)
        if policy.check_mass_balance and all_are_entities:
            combined_comp = Counter()
            for key, obj in entity_objects.items():
                coefficient = coefficients[key] if coefficients is not None else 1
                for species, amount in obj.composition.items():
                    combined_comp[species] += coefficient * amount

            expected_comp = self.composition
            if combined_comp.keys() != expected_comp.keys() or any(
                (
                    combined_comp[species] != amount
                    if coefficients is None
                    else not math.isclose(
                        combined_comp[species], amount, rel_tol=0, abs_tol=1e-8
                    )
                )
                for species, amount in expected_comp.items()
            ):
                raise StoichiometryError(
                    f"Mass balance mismatch for {policy.label} references. "
                    f"Combined fragments: {dict(combined_comp)}, "
                    f"expected: {dict(self.composition)}."
                )

        if policy.single_species:
            return {symbol: calculation_references[symbol] for symbol in required_species}
        return calculation_references

class Nanocluster(ChemicalEntity):
    """
    A non-periodic atomic aggregate, such as a nanoparticle model.

    A periodic `Structure` assigned as geometry is converted to a `Molecule`
    using its Cartesian coordinates.
    """
    def __init__(
            self, 
            name: str, 
            total_energy: float | None = None, 
            net_magmom: float | None = None, 
            geometry: Molecule | None = None, 
            valence_electrons: dict[str, int] | None = None
    ):
        """
        Initializes a nanocluster.

        Args:
            name: Non-empty name of the nanocluster.
            total_energy: Total energy; currently must be convertible to float.
            net_magmom: Net magnetic moment; currently must be convertible to float.
            geometry: A `Molecule`, `Structure`, or None. Structures are converted
                to molecules using Cartesian coordinates.
            valence_electrons: Element symbols mapped to valence-electron counts,
                or None if unavailable.
        """
        super().__init__(name=name, total_energy=total_energy, net_magmom=net_magmom,
                         geometry=geometry, valence_electrons=valence_electrons)

    @ChemicalEntity.geometry.setter
    def geometry(self, object: Structure | Molecule | None):
        """Sets the nanocluster geometry, converting a `Structure` to a `Molecule`."""
        if isinstance(object, Structure):
            object = Molecule(
                species=object.species,
                coords=object.cart_coords,
                site_properties=object.site_properties
            )
        elif object is not None and not isinstance(object, Molecule):
            raise TypeError("Geometry must be a Structure, Molecule, or None.")
        
        ChemicalEntity.geometry.fset(self, object)

    def ideal_cohesive_energy(self, **named_clusters: tuple[Self | float, Self | float]) -> float:
        """
        Attempts to calculate the composition-weighted ideal cohesive energy.

        Args:
            **named_clusters: Element symbols mapped to `(cluster, atom)` pairs.
                Each reference may be a `ChemicalEntity` or a numeric total energy.
                Entity cluster references must be monometallic and have the same
                atom count as this nanocluster; entity atom references must contain
                one atom of one species.

        Returns:
            float: Intended ideal cohesive energy in eV/atom.

        Raises:
            MissingGeometryError: If this nanocluster or an entity reference lacks
                the geometry needed to determine its composition or atom count.
            MissingDataError: If an entity reference has no total energy.
            MissingReferenceError: If cluster or atom references do not cover all
                elements in the nanocluster.
            StoichiometryError: If an entity reference is not monometallic, has the
                wrong cluster size, or is not a single-atom atomic reference.
            KeyError: If entity references cover the elements but their keyword
                names do not include every element symbol.
            IndexError: If a reference pair has fewer than two entries.
            TypeError: If a pair cannot be indexed, a reference cannot be converted
                to float, or the current cohesive-energy helper call is reached.
            ValueError: If a reference cannot be converted to float.
        """
        mono_cluster_coh = {}

        mono_clusters = {
            name: pair[0] for name, pair in named_clusters.items()
        }
        atoms = {
            name: pair[1] for name, pair in named_clusters.items()
        }

        mono_clusters_refs = self._generate_calculation_references(
            _CLUSTER_POLICY,
            **mono_clusters
        )
        atom_refs = self._generate_calculation_references(
            _ATOM_POLICY,
            **atoms
        )

        for name, ref in mono_clusters_refs.items():
            mono_cluster_coh[name] = calculate_cohesive_energy(
                total_energy=ref.total_energy if isinstance(ref, ChemicalEntity) else None,
                num_atoms=atoms[name],
                atomic_counts=atom_refs[name].atomic_counts if isinstance(
                    atom_refs[name], ChemicalEntity) else None
            )

        return calculate_ideal_cohesive_energy(
            mono_cohesive_energies=mono_cluster_coh,
            atomic_fractions=self.atomic_fractions
        )

    def mix_cohesive_energy(self, **named_clusters: tuple[Self | float, Self | float]) -> float:
        """
        Attempts to calculate cohesive energy relative to the ideal value.

        Args:
            **named_clusters: Element symbols mapped to `(cluster, atom)` pairs,
                as accepted by `ideal_cohesive_energy`.

        Returns:
            float: Intended difference in cohesive energies, in eV/atom.

        Raises:
            MissingGeometryError: If this nanocluster or an entity reference lacks
                the geometry needed to validate the references.
            MissingDataError: If an entity reference has no total energy.
            MissingReferenceError: If references do not cover every element.
            StoichiometryError: If an entity reference is not monometallic, has an
                incorrect cluster size, or is not a single-atom atomic reference.
            KeyError: If entity references cover the elements but their keyword
                names do not include every element symbol.
            IndexError: If a reference pair has fewer than two entries.
            TypeError: If a pair cannot be indexed, a reference cannot be converted
                to float, or either of the incompatible calls described below is reached.
            ValueError: If a reference cannot be converted to float.
        """
        ideal_coh = self.ideal_cohesive_energy(**named_clusters)
        return self.cohesive_energy - ideal_coh

    def mixing_energy(self, *clusters: Self, **named_clusters: Self | float) -> float:
        """
        Calculates the mixing energy relative to relaxed monometallic clusters.

        Args:
            *clusters: Monometallic `ChemicalEntity` references; element symbols
                are inferred from their compositions.
            **named_clusters: Element symbols mapped to monometallic
                `ChemicalEntity` references or numeric total energies. Entity
                references must have the same number of atoms as this nanocluster.

        Returns:
            float: `E_alloy - sum(x_i * E_relaxed_i)`, in the energy units of the
            inputs (typically eV).

        Raises:
            MissingGeometryError: If this nanocluster or an entity reference lacks
                the geometry needed to validate the references.
            MissingDataError: If this nanocluster or an entity reference lacks a
                total energy.
            TypeError: If a positional reference is not a `ChemicalEntity` or a
                keyword energy cannot be converted to float.
            ValueError: If a reference is duplicated or a keyword energy cannot be
                converted to float.
            StoichiometryError: If an entity reference is not monometallic or has
                the wrong atom count, or if the atomic fractions are invalid.
            MissingReferenceError: If a monometallic reference is missing for any
                element in the nanocluster.
            KeyError: If entity references cover the elements but their keyword
                names do not include every element symbol.
            EnergyCalculationError: If the alloy or a used reference energy is
                non-finite.
        """
        cluster_refs = self._generate_calculation_references(
            _CLUSTER_POLICY, 
            *clusters, 
            **named_clusters
        )

        return calculate_mixing_energy(
            alloy_energy=self.total_energy,
            mono_energies=cluster_refs,
            atomic_fractions=self.atomic_fractions
        )
    
    def strain_mixing_energy(self, **named_clusters: tuple[Self | float, Self | float]) -> float:
        """
        Calculates a composition-weighted difference between cluster references.

        Args:
            **named_clusters: Element symbols mapped to `(frozen, relaxed)` pairs
                of `ChemicalEntity` references or numeric total energies. Entity
                references must be monometallic and match the nanocluster's atom
                count; frozen entity references must also match its geometry.
                Geometry and atom counts cannot be checked for numeric references.

        Returns:
            float: Currently `sum(x_i * (E_relaxed_i - E_frozen_i))`, in the energy
            units of the inputs (typically eV).

        Raises:
            MissingGeometryError: If this nanocluster or an entity reference lacks
                the geometry needed to validate the references.
            MissingDataError: If an entity reference lacks a total energy.
            MissingReferenceError: If a frozen or relaxed reference is missing for
                an element in the nanocluster.
            StoichiometryError: If an entity reference is not monometallic or has
                the wrong atom count, or if the atomic fractions are invalid.
            MismatchedGeometryError: If a frozen entity reference does not match
                this nanocluster's geometry.
            RuntimeError: If the molecule matcher cannot run, for example because
                the Open Babel Python bindings are unavailable.
            KeyError: If entity references cover the elements but their keyword
                names do not include every element symbol.
            IndexError: If a reference pair has fewer than two entries.
            TypeError: If a pair cannot be indexed or an energy cannot be converted
                to float.
            ValueError: If an energy cannot be converted to float.
            EnergyCalculationError: If a used reference energy is non-finite.
        """
        frozen_clusters = {
            name: pair[0] for name, pair in named_clusters.items()
        }
        relaxed_clusters = {
            name: pair[1] for name, pair in named_clusters.items()
        }
        
        frozen_refs = self._generate_calculation_references(
            _CLUSTER_GEOMETRY_POLICY, **frozen_clusters
        )
        relaxed_refs = self._generate_calculation_references(
            _CLUSTER_POLICY, **relaxed_clusters
        )
        
        return calculate_strain_mixing_energy(
            frozen_energies=frozen_refs,
            relaxed_energies=relaxed_refs,
            atomic_fractions=self.atomic_fractions
        )

    def interaction_mixing_energy(self, *clusters: Self, **named_clusters: Self | float) -> float:
        """
        Attempts to calculate the interaction contribution to mixing energy.

        Args:
            *clusters: Frozen monometallic `ChemicalEntity` references whose
                element symbols are inferred from their compositions.
            **named_clusters: Element symbols mapped to frozen monometallic
                `ChemicalEntity` references or numeric total energies. Entity
                references must match the nanocluster's atom count and geometry;
                numeric references cannot be checked for either.

        Returns:
            float: Intended interaction contribution, `E_alloy - sum(x_i *
            E_frozen_i)`, in the energy units of the inputs (typically eV).

        Raises:
            MissingGeometryError: If this nanocluster or an entity reference lacks
                the geometry needed to validate the references.
            MissingDataError: If this nanocluster or an entity reference lacks a
                total energy.
            MissingReferenceError: If a frozen reference is missing for an element.
            StoichiometryError: If an entity reference is not monometallic or has
                the wrong atom count.
            MismatchedGeometryError: If a frozen entity reference does not match
                this nanocluster's geometry.
            RuntimeError: If the molecule matcher cannot run, for example because
                the Open Babel Python bindings are unavailable.
            KeyError: If entity references cover the elements but their keyword
                names do not include every element symbol.
            TypeError: If a positional reference is not a `ChemicalEntity`, a
                keyword energy cannot be converted to float, or the current
                interaction-energy helper call is reached.
            ValueError: If a reference is duplicated or a keyword energy cannot be
                converted to float.
        """
        frozen_refs = self._generate_calculation_references(
            _CLUSTER_GEOMETRY_POLICY,
            *clusters,
            **named_clusters
        )

        return calculate_interact_mixing_energy(
            alloy_energy=self.total_energy,
            frozen_energies=frozen_refs,
            atomic_fractions=self.atomic_fractions
        )


class Surface(ChemicalEntity):
    """A periodic surface model representing a great variety of materials."""

    def __init__(
        self, 
        name: str, 
        total_energy: float | None = None, 
        net_magmom: float | None = None, 
        geometry: Structure | None = None, 
        valence_electrons: dict[str, int] | None = None
    ):
        """
        Initializes a surface model.

        Attributes:
            name (str): The name of the surface model.
            total_energy (float | None): The total energy of the surface model.
            net_magmom (float | None): The net magnetic moment of the surface model.
            geometry (Structure | None): The geometry of the surface model.
            valence_electrons (dict[str, int] | None): The valence electrons of the 
                surface model.
        """
        super().__init__(name=name, total_energy=total_energy, net_magmom=net_magmom,
                         geometry=geometry, valence_electrons=valence_electrons)

        
class MetalOxide(Surface):
    """A periodic surface model representing a metal oxide."""

    def __init__(
        self, 
        name: str, 
        total_energy: float | None = None, 
        net_magmom: float | None = None, 
        geometry: Structure | None = None, 
        valence_electrons: dict[str, int] | None = None
    ):
        """
        Initializes a metal oxide surface model.

        Attributes:
            name (str): The name of the metal oxide surface model.
            total_energy (float | None): The total energy of the metal oxide surface model.
            net_magmom (float | None): The net magnetic moment of the metal oxide surface model.
            geometry (Structure | None): The geometry of the metal oxide surface model.
            valence_electrons (dict[str, int] | None): The valence electrons of the 
                metal oxide surface model.
        """
        super().__init__(name=name, total_energy=total_energy, net_magmom=net_magmom,
                         geometry=geometry, valence_electrons=valence_electrons)

    def vacancy_formation_energy(
        self,
        pristine: MetalOxide,
        oxygen: ChemicalEntity,
        num_vacancies: int = 1
    ) -> float:
        """Calculate the total oxygen-vacancy formation energy.

        This instance is the defective oxide. `pristine` is the oxide before
        oxygen removal and `oxygen` is an O2 reference with its total molecular
        energy. All three energies must use the same units (typically eV).

        The result is E_defective - E_pristine + num_vacancies * E_O2 / 2.
        Entity geometries are required to validate the oxygen reference and
        the mass balance. The result is total energy, not energy per vacancy.

        Raises:
            TypeError: If either reference is not a chemical entity.
            StoichiometryError: If the vacancy count, O2 reference, or mass
                balance is invalid.
            MissingGeometryError: If an entity lacks geometry.
            MissingDataError: If an entity lacks total energy.
            EnergyCalculationError: If an energy is non-finite.
        """
        if not isinstance(pristine, MetalOxide) or not isinstance(oxygen, ChemicalEntity):
            raise TypeError("Pristine must be a MetalOxide and oxygen a ChemicalEntity.")
        if isinstance(num_vacancies, bool) or not isinstance(num_vacancies, int) or num_vacancies <= 0:
            raise StoichiometryError("The number of vacancies must be a positive integer.")
        if oxygen.composition != {"O": 2}:
            raise StoichiometryError("The oxygen reference must represent an O2 molecule.")

        references = pristine._generate_calculation_references(
            _SYSTEM_POLICY,
            coefficients={"defective": 1, "oxygen": num_vacancies / 2},
            defective=self,
            oxygen=oxygen,
        )
        return calculate_vacancy_formation_energy(
            defective_energy=references["defective"],
            pristine_energy=pristine.total_energy,
            defective_site_energy=references["oxygen"] / 2,
            num_vacancies=num_vacancies,
        )

    