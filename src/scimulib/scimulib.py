from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Literal
from collections.abc import Mapping, Sequence

import numpy as np
import math

from pymatgen.core import Element, Molecule, Structure
from pymatgen.analysis.molecule_matcher import MoleculeMatcher
from pymatgen.analysis.structure_matcher import StructureMatcher

from scimulib.validation_utils import check_pymatgen_imports, validate_vacancy_count
from scimulib.references import (
    ATOMIC, MASS_BALANCE, Entry, ReferenceSpec, Refs, check_mass_balance,
    key_in_composition, keys_cover, keys_cover_target, keys_within, same_atom_count,
    same_geometry, select_target_species, single_species, stored_keys_in_target,
)
from scimulib.exceptions import (
    VaspExtractionError, BaderExtractionError, 
    MissingDataError, MissingGeometryError, NonPeriodicGeometryError, MissingSitePropertyError,
    MissingReferenceError, StoichiometryError, EnergyCalculationError
)
from scimulib.visualization import AxesMode, LabelMode, RenderStyle, build_py3dmol_view
from scimulib.abinitio_parser import VaspParser, BaderParser
from scimulib.bader_calc import (
    calculate_bader_net_charges, sum_bader_net_charges, sum_bader_magnetic_moments
)
from scimulib.energy_calc import (
    calculate_cohesive_energy, calculate_ideal_cohesive_energy, calculate_mixing_energy,
    calculate_strain_mixing_energy, calculate_interact_mixing_energy,
    calculate_vacancy_formation_energy, calculate_adsorption_energy
)


def _validate_finite_reference_energy(energy: float, label: str) -> None:
    """Require a reference energy to be finite.

    Args:
        energy: Energy value to validate.
        label: Reference kind included in the error message.

    Raises:
        EnergyCalculationError: If `energy` is not finite.
    """
    if not math.isfinite(energy):
        raise EnergyCalculationError(f"{label} reference energy must be finite.")


def _infer_oxygen_vacancies(
    pristine: Mapping[str, float], defective: Mapping[str, float]
) -> int:
    """Infer the number of oxygen vacancies from pristine and defective formulas.

    Args:
        pristine: Composition of the oxide before oxygen removal.
        defective: Composition of the oxide after oxygen removal.

    Returns:
        The positive integer difference in oxygen counts.

    Raises:
        StoichiometryError: If the defective composition has no positive integer
            oxygen deficit relative to the pristine composition.
    """
    difference = pristine.get("O", 0) - defective.get("O", 0)
    if difference <= 0 or not float(difference).is_integer():
        raise StoichiometryError("The number of oxygen vacancies must be a positive integer.")
    return int(difference)


class ChemicalEntity:
    """
    A class representing a chemical entity, encapsulating its name, total energy, 
    net magnetic moment, and geometric structure. It also stores any additional 
    site properties extracted from the calculation.

    Optional reference mappings are copied, but entity references remain live:
    their current geometry and energy are checked when used. Method arguments
    override stored references for the same element only for that call.

    For example: `entity.cohesive_energy()` can use references supplied with
    `atomic_references={"Au": atom, "Cu": -2.0}` at construction.
    """

    def __init__(
            self, 
            name: str, 
            total_energy: float | None = None, 
            net_magmom: float | None = None, 
            geometry: Structure | Molecule | None = None,
            valence_electrons: dict[str, float] | None = None,
            *,
            atomic_references: Mapping[str, ChemicalEntity | float] | None = None,
    ):
        """
        Initializes a ChemicalEntity instance.

        Attributes:
            name (str): The name of the chemical entity.
            total_energy (float | None): The total energy of the chemical entity.
            net_magmom (float | None): The net magnetic moment of the chemical entity.
            geometry (Structure | Molecule | None): The geometric structure of the
                chemical entity. A copy is stored so later changes to the supplied
                object do not change this entity.
            valence_electrons (dict[str, float], optional): The valence electrons of each element in the 
                chemical entity. Defaults to an empty dictionary.
            atomic_references: Optional element-to-isolated-atom references or
                total energies. They may be supplemented or overridden in a call.

        Raises:
            ValueError: If energy, magnetic moment, and geometry are all None.
        """
        check_pymatgen_imports()

        self.name = name
        self.total_energy = total_energy
        self.net_magmom = net_magmom
        self.geometry = geometry
        self.valence_electrons = valence_electrons

        if (
            self._total_energy is None
            and self._structure is None 
            and self._net_magmom is None
        ):
            raise ValueError(
                f"Cannot create a {self.__class__.__name__} instance. "
                "At least one property must be provided."
            )
        self._atomic_references = ATOMIC.store(self, atomic_references)

    def __repr__(self):
        """Return a concise representation with the name, formula, and energy."""
        return (
            f"{self.__class__.__name__} (name={self.name}, "
            f"formula={self._structure.composition.formula if self._structure else 'N/A'}, "
            f"total_energy={self._total_energy if self._total_energy is not None else 'N/A'})"
        )

    # --- Factory Methods ---
    # Factory methods for constructing instances from external data sources
    @classmethod
    def from_vasp(cls, directory: str | Path, name: str | None = None, **references: Any):
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
            **references: Optional reference arguments accepted by the class
                constructor, such as `atomic_references`.

        Returns:
            cls: An initialized instance of the class containing the parsed VASP data.

        Raises:
            ValueError: If energy, magnetic moment, and geometry could not be
                extracted from the provided directory.

        """
        entity_name = name if name is not None else Path(directory).name

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
            log = logging.debug if isinstance(e.__cause__, FileNotFoundError) else logging.warning
            log(
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
            log = logging.debug if isinstance(e.__cause__, FileNotFoundError) else logging.warning
            log(
                f"{cls.__name__} '{entity_name}': {e}"
            )

        return cls(
            name=entity_name,
            total_energy=total_energy,
            net_magmom=net_magmom,
            geometry=structure,
            valence_electrons=valence_electrons,
            **references,
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
        self._structure = object.copy() if object is not None else None
        self._bader_net_charges_calculated = False

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
        self._valence_electrons = value.copy() if value is not None else {}
        if self._bader_net_charges_calculated and self._structure is not None:
            self._structure.remove_site_property("net_bader_charge")
            self._bader_net_charges_calculated = False

    @property
    def atomic_references(self) -> dict[str, ChemicalEntity | float]:
        """Stored isolated-atom references, keyed by element."""
        return self._atomic_references.copy()

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

    @property
    def symbols(self) -> tuple[str, ...]:
        """Returns the chemical symbols of the atoms in the structure, if available."""
        return self._require_geometry.symbol_set

    @property
    def atomic_symbols(self) -> list[str]:
        """Returns the atomic symbols of the chemical entity, if available."""
        return [site.species_string for site in self._require_geometry]

    @property
    def composition(self) -> dict[str, float]:
        """Returns the composition of the chemical entity, if available."""
        return self._require_geometry.composition.get_el_amt_dict()

    @property
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
        other: "ChemicalEntity",
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
            other (ChemicalEntity): The other entity to compare with.
            ltol (float): Lattice tolerance for structure comparison.
            stol (float): Site tolerance for structure comparison.
            angle_tol (float): Angle tolerance for structure comparison.
            mol_tol (float): Tolerance for molecule comparison.
            ignore_species (bool): Whether to compare positions without regard
                to the species or their distribution.

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
                geom_a = geom_a.copy()
                geom_b = geom_b.copy()
                for structure in (geom_a, geom_b):
                    for index in range(len(structure)):
                        structure.replace(index, "H")
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

    @property
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
        self._bader_net_charges_calculated = True
        return np.array(bader_net_charges)

    @property
    def bader_magnetic_moments(self) -> np.ndarray:
        """Returns the Bader magnetic moments as a numpy array, if available."""
        return self._require_site_property("bader_magmom")

    # --- Summed Bader Analysis Properties ---
    # These properties provide the summed Bader charges and magnetic moments 
    # per species.
    @property
    def sum_bader_charges(self) -> dict[str, float]:
        """Sums the net Bader charges per species."""
        return sum_bader_net_charges(
            species_list=self.atomic_symbols,
            net_bader_charges=self.bader_net_charges
        )

    @property
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
        **named_atoms: ChemicalEntity | float,
    ) -> float:
        """Calculates the cohesive energy from isolated atomic references.

        Atomic references are passed by element symbol, as single-atom
        `ChemicalEntity` instances or numeric energies.
        Omitted elements use `atomic_references` stored at construction; supplied
        elements override them for this call without changing the stored mapping.

        Args:
            **named_atoms: Keyword arguments mapping atomic symbols (e.g., 'Au', 'O')
                to either a `ChemicalEntity` instance or its total energy as a float.

        Returns:
            float: The cohesive energy per atom, typically in eV/atom.

        Raises:
            StoichiometryError: If an entity reference is not a single atom or
                does not match its element symbol.
            MissingReferenceError: If atomic references are missing for any species
                present in the structure.
        """
        filtered_energies = ATOMIC.resolve(
            self, named_atoms, stored=self._atomic_references
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
    # Reference validation lives in `scimulib.references` as declarative specs.
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


def _validate_oxygen_reference(oxygen: ChemicalEntity) -> None:
    """Require the oxygen reference to be an O2 chemical entity.

    Args:
        oxygen: Entity used as the molecular oxygen reference.

    Raises:
        TypeError: If `oxygen` is not a `ChemicalEntity`.
        MissingGeometryError: If the entity has no geometry from which to read
            its composition.
        StoichiometryError: If its composition is not exactly O2.
    """
    if not isinstance(oxygen, ChemicalEntity):
        raise TypeError("Oxygen must be a ChemicalEntity.")
    if oxygen.composition != {"O": 2}:
        raise StoichiometryError("The oxygen reference must represent an O2 molecule.")


def _check_multimetallic(cluster: ChemicalEntity) -> None:
    """Require a cluster to contain at least two distinct metallic elements.

    Args:
        cluster: Nanocluster whose species are checked.

    Raises:
        MissingGeometryError: If the cluster has no geometry.
        StoichiometryError: If fewer than two species are metallic.
    """
    if sum(Element(symbol).is_metal for symbol in cluster.symbols) < 2:
        raise StoichiometryError(
            "Stored monometallic cluster references require a multimetallic nanocluster."
        )


def _stored_need_multimetallic_target(refs: Refs) -> None:
    """Require a multimetallic target when it has stored cluster references.

    Args:
        refs: Reference-validation context for the target cluster.

    Raises:
        MissingGeometryError: If the target has no geometry.
        StoichiometryError: If stored cluster references are used with a target
            containing fewer than two metallic species.
    """
    if refs.stored:
        _check_multimetallic(refs.target)


_RELAXED_CLUSTER = ReferenceSpec(
    "cluster",
    key_rules=(_stored_need_multimetallic_target, stored_keys_in_target),
    entity_rules=(single_species, key_in_composition, same_atom_count),
    set_rules=(keys_cover_target,),
    select=select_target_species,
)
_FROZEN_CLUSTER = _RELAXED_CLUSTER.extended("frozen cluster", same_geometry)


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
            valence_electrons: dict[str, int] | None = None,
            *,
            atomic_references: Mapping[str, ChemicalEntity | float] | None = None,
            relaxed_references: Mapping[str, ChemicalEntity | float] | None = None,
            frozen_references: Mapping[str, ChemicalEntity | float] | None = None,
    ):
        """
        Initializes a nanocluster.

        Args:
            name: Non-empty name of the nanocluster.
            total_energy: Total energy, or None if unavailable.
            net_magmom: Net magnetic moment, or None if unavailable.
            geometry: A `Molecule`, `Structure`, or None. Structures are converted
                to molecules using Cartesian coordinates.
            valence_electrons: Element symbols mapped to valence-electron counts,
                or None if unavailable.
            atomic_references: Isolated atoms or their energies, keyed by element.
            relaxed_references: Relaxed monometallic clusters or their energies.
            frozen_references: Frozen monometallic clusters or their energies.
                Cluster references are only accepted for systems containing at
                least two metallic elements. Numeric energies cannot be checked
                for cluster size, composition, or geometry.
        """
        super().__init__(name=name, total_energy=total_energy, net_magmom=net_magmom,
                         geometry=geometry, valence_electrons=valence_electrons,
                         atomic_references=atomic_references)
        self._relaxed_references = _RELAXED_CLUSTER.store(self, relaxed_references)
        self._frozen_references = _FROZEN_CLUSTER.store(self, frozen_references)

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

    @property
    def relaxed_references(self) -> dict[str, ChemicalEntity | float]:
        """Stored relaxed monometallic references, keyed by element."""
        return self._relaxed_references.copy()

    @property
    def frozen_references(self) -> dict[str, ChemicalEntity | float]:
        """Stored frozen monometallic references, keyed by element."""
        return self._frozen_references.copy()

    def ideal_cohesive_energy(
        self, **named_clusters: tuple[ChemicalEntity | float, ChemicalEntity | float]
    ) -> float:
        """
        Calculates the composition-weighted ideal cohesive energy.

        Missing elements use stored relaxed cluster and atomic references.
        A supplied pair overrides both references for its element for this call.

        Args:
            **named_clusters: Element symbols mapped to `(cluster, atom)` pairs.
                Each reference may be a `ChemicalEntity` or a numeric total energy.
                Entity cluster references must be monometallic and have the same
                atom count as this nanocluster; entity atom references must contain
                one atom of one species.

        Returns:
            float: Ideal cohesive energy in eV/atom.

        Raises:
            MissingGeometryError: If this nanocluster or an entity reference lacks
                the geometry needed to determine its composition or atom count.
            MissingDataError: If an entity reference has no total energy.
            MissingReferenceError: If cluster or atom references do not cover all
                elements in the nanocluster.
            StoichiometryError: If an entity reference is not monometallic, has the
                wrong cluster size, or is not a single-atom atomic reference.
            IndexError: If a reference pair has fewer than two entries.
            TypeError: If a pair cannot be indexed or a reference cannot be
                converted to float.
            ValueError: If a reference cannot be converted to float.
        """
        mono_cluster_coh = {}

        mono_clusters = {
            name: pair[0] for name, pair in named_clusters.items()
        }
        atoms = {
            name: pair[1] for name, pair in named_clusters.items()
        }

        mono_clusters_refs = _RELAXED_CLUSTER.resolve(
            self, mono_clusters, stored=self._relaxed_references
        )
        atom_refs = ATOMIC.resolve(self, atoms, stored=self._atomic_references)

        total_atoms = len(self._require_geometry)
        for name, ref in mono_clusters_refs.items():
            mono_cluster_coh[name] = calculate_cohesive_energy(
                total_energy=ref,
                atomic_energies={name: atom_refs[name]},
                atomic_counts={name: total_atoms},
            )

        return calculate_ideal_cohesive_energy(
            mono_cohesive_energies=mono_cluster_coh,
            atomic_fractions=self.atomic_fractions
        )

    def mix_cohesive_energy(
        self, **named_clusters: tuple[ChemicalEntity | float, ChemicalEntity | float]
    ) -> float:
        """
        Calculates cohesive energy relative to the ideal value.

        Missing elements use the stored relaxed cluster and atomic references.

        Args:
            **named_clusters: Element symbols mapped to `(cluster, atom)` pairs,
                as accepted by `ideal_cohesive_energy`.

        Returns:
            float: Difference in cohesive energies, in eV/atom.

        Raises:
            MissingGeometryError: If this nanocluster or an entity reference lacks
                the geometry needed to validate the references.
            MissingDataError: If an entity reference has no total energy.
            MissingReferenceError: If references do not cover every element.
            StoichiometryError: If an entity reference is not monometallic, has an
                incorrect cluster size, or is not a single-atom atomic reference.
            IndexError: If a reference pair has fewer than two entries.
            TypeError: If a pair cannot be indexed or a reference cannot be
                converted to float.
            ValueError: If a reference cannot be converted to float.
        """
        ideal_coh = self.ideal_cohesive_energy(**named_clusters)
        atoms = {name: pair[1] for name, pair in named_clusters.items()}
        return self.cohesive_energy(**atoms) - ideal_coh

    def mixing_energy(self, **named_clusters: ChemicalEntity | float) -> float:
        """
        Calculates the mixing energy relative to relaxed monometallic clusters.

        Omitted elements use stored relaxed references; supplied elements override
        them for this call.

        Args:
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
            TypeError: If a keyword energy cannot be converted to float.
            ValueError: If a keyword energy cannot be converted to float.
            StoichiometryError: If an entity reference is not monometallic or has
                the wrong atom count, or if the atomic fractions are invalid.
            MissingReferenceError: If a monometallic reference is missing for any
                element in the nanocluster.
            KeyError: If entity references cover the elements but their keyword
                names do not include every element symbol.
            EnergyCalculationError: If the alloy or a used reference energy is
                non-finite.
        """
        cluster_refs = _RELAXED_CLUSTER.resolve(
            self, named_clusters, stored=self._relaxed_references
        )

        return calculate_mixing_energy(
            alloy_energy=self.total_energy,
            mono_energies=cluster_refs,
            atomic_fractions=self.atomic_fractions
        )
    
    def strain_mixing_energy(
        self, **named_clusters: tuple[ChemicalEntity | float, ChemicalEntity | float]
    ) -> float:
        """
        Calculates a composition-weighted difference between cluster references.

        Omitted elements use stored frozen and relaxed references. A supplied
        pair overrides both references for its element for this call.

        Args:
            **named_clusters: Element symbols mapped to `(frozen, relaxed)` pairs
                of `ChemicalEntity` references or numeric total energies. Entity
                references must be monometallic and match the nanocluster's atom
                count; frozen entity references must also match its geometry.
                Geometry and atom counts cannot be checked for numeric references.

        Returns:
            float: `sum(x_i * (E_frozen_i - E_relaxed_i))`, in the energy
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
        
        frozen_refs = _FROZEN_CLUSTER.resolve(
            self, frozen_clusters, stored=self._frozen_references
        )
        relaxed_refs = _RELAXED_CLUSTER.resolve(
            self, relaxed_clusters, stored=self._relaxed_references
        )
        
        return calculate_strain_mixing_energy(
            frozen_energies=frozen_refs,
            relaxed_energies=relaxed_refs,
            atomic_fractions=self.atomic_fractions
        )

    def interaction_mixing_energy(self, **named_clusters: ChemicalEntity | float) -> float:
        """
        Attempts to calculate the interaction contribution to mixing energy.

        Omitted elements use stored frozen references; supplied elements override
        them for this call.

        Args:
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
            TypeError: If a keyword energy cannot be converted to float.
            ValueError: If a keyword energy cannot be converted to float.
        """
        frozen_refs = _FROZEN_CLUSTER.resolve(
            self, named_clusters, stored=self._frozen_references
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
        valence_electrons: dict[str, int] | None = None,
        *,
        atomic_references: Mapping[str, ChemicalEntity | float] | None = None,
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
            atomic_references: Isolated atoms or their energies, keyed by element.
        """
        super().__init__(name=name, total_energy=total_energy, net_magmom=net_magmom,
                         geometry=geometry, valence_electrons=valence_electrons,
                         atomic_references=atomic_references)

        
class MetalOxide(Surface):
    """A periodic surface model representing a metal oxide."""

    def __init__(
        self, 
        name: str, 
        total_energy: float | None = None, 
        net_magmom: float | None = None, 
        geometry: Structure | None = None, 
        valence_electrons: dict[str, int] | None = None,
        *,
        atomic_references: Mapping[str, ChemicalEntity | float] | None = None,
        pristine_reference: MetalOxide | None = None,
        oxygen_reference: ChemicalEntity | None = None,
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
            atomic_references: Isolated atoms or their energies, keyed by element.
            pristine_reference: Pristine oxide for vacancy calculations.
            oxygen_reference: O2 molecule for vacancy calculations.
                When both are provided, their mass balance is checked against
                the number of missing oxygen atoms inferred from the geometries.
        """
        super().__init__(name=name, total_energy=total_energy, net_magmom=net_magmom,
                         geometry=geometry, valence_electrons=valence_electrons,
                         atomic_references=atomic_references)
        if pristine_reference is not None:
            if not isinstance(pristine_reference, MetalOxide):
                raise TypeError("Pristine must be a MetalOxide.")
            pristine_reference._require_geometry
            _validate_finite_reference_energy(pristine_reference.total_energy, "Pristine")
        if oxygen_reference is not None:
            _validate_oxygen_reference(oxygen_reference)
            _validate_finite_reference_energy(oxygen_reference.total_energy, "Oxygen")
        self._pristine_reference = pristine_reference
        self._oxygen_reference = oxygen_reference
        if pristine_reference is not None and oxygen_reference is not None:
            oxygen_difference = _infer_oxygen_vacancies(
                pristine_reference.composition, self.composition
            )
            check_mass_balance(
                pristine_reference,
                {"defective": self, "oxygen": oxygen_reference},
                coefficients={"defective": 1, "oxygen": oxygen_difference / 2},
            )

    @property
    def pristine_reference(self) -> MetalOxide | None:
        """Stored pristine oxide for vacancy calculations."""
        return self._pristine_reference

    @property
    def oxygen_reference(self) -> ChemicalEntity | None:
        """Stored O2 molecule for vacancy calculations."""
        return self._oxygen_reference

    def vacancy_formation_energy(
        self,
        *,
        pristine: MetalOxide | None = None,
        oxygen: ChemicalEntity | None = None,
        num_vacancies: int = 1
    ) -> float:
        """Calculate the total oxygen-vacancy formation energy.

        This instance is the defective oxide. `pristine` is the oxide before
        oxygen removal and `oxygen` is an O2 reference with its total molecular
        energy. All three energies must use the same units (typically eV).
        Omitted references use those stored on the instance; explicit references
        must be passed by keyword and override them for this call only. The
        vacancy count defaults to one even when references are stored, and must
        be specified for multiple vacancies.

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
        pristine = pristine if pristine is not None else self._pristine_reference
        oxygen = oxygen if oxygen is not None else self._oxygen_reference
        if pristine is None or oxygen is None:
            raise MissingReferenceError("Pristine oxide and O2 references are required.")
        if not isinstance(pristine, MetalOxide) or not isinstance(oxygen, ChemicalEntity):
            raise TypeError("Pristine must be a MetalOxide and oxygen a ChemicalEntity.")
        validate_vacancy_count(num_vacancies)
        _validate_oxygen_reference(oxygen)

        references = MASS_BALANCE.resolve(
            pristine,
            {"defective": self, "oxygen": oxygen},
            coefficients={"defective": 1, "oxygen": num_vacancies / 2},
        )
        return calculate_vacancy_formation_energy(
            defective_energy=references["defective"],
            pristine_energy=pristine.total_energy,
            defective_site_energy=references["oxygen"] / 2,
            num_vacancies=num_vacancies,
        )

def _nanocluster_species(system: SupportedSystem) -> set[str]:
    """Return the distinct element symbols in a supported system's nanocluster.

    Args:
        system: Supported system whose nanocluster is inspected.

    Returns:
        The set of element symbols in the nanocluster.

    Raises:
        MissingGeometryError: If the nanocluster has no geometry.
    """
    return set(system.nanocluster.symbols)


def _stored_need_multimetallic_nanocluster(refs: Refs) -> None:
    """Require a multimetallic nanocluster when supported references are stored.

    Args:
        refs: Reference-validation context for a supported system.

    Raises:
        MissingGeometryError: If the target nanocluster has no geometry.
        StoichiometryError: If stored supported references are used with a
            nanocluster containing fewer than two metallic species.
    """
    if refs.stored:
        _check_multimetallic(refs.target.nanocluster)


def _supported_composition(entry: Entry) -> None:
    """Require a supported reference to hold the support plus one cluster species.

    Args:
        entry: Supported-system entity reference being validated.

    Raises:
        MissingGeometryError: If a component or reference lacks a geometry
            needed to determine its composition.
        StoichiometryError: If the reference composition differs from the
            expected support-plus-cluster composition.
    """
    system = entry.target
    expected = dict(system.surface.composition)
    expected[entry.key] = expected.get(entry.key, 0) + sum(system.nanocluster.composition.values())
    if entry.composition != expected:
        raise StoichiometryError(
            f"Supported reference '{entry.key}' has composition "
            f"{entry.composition}, expected {expected}."
        )


_SUPPORTED_RELAXED = ReferenceSpec(
    "supported cluster",
    key_rules=(
        _stored_need_multimetallic_nanocluster,
        keys_within(_nanocluster_species),
        keys_cover(_nanocluster_species),
    ),
    entity_rules=(_supported_composition,),
)
_SUPPORTED_FROZEN = _SUPPORTED_RELAXED.extended("frozen supported cluster", same_geometry)


class SupportedSystem(ChemicalEntity):
    """A nanocluster supported on a surface.

    The component geometries and the supported geometry must have identical
    combined compositions. Components and entity references remain live and
    are revalidated when used.

    Relaxed and frozen references are complete supported systems: each has the
    same surface composition and a monometallic cluster of the original size.
    Frozen entity references must also match the complete system geometry.
    Numeric references are total energies of those complete systems; their
    composition and geometry cannot be checked.

    Atomic, relaxed and frozen references may be stored or supplied to energy
    methods. Stored mappings may be partial; supplied values override them for
    one call, and live references are checked again when used. For oxygen
    vacancies, a pristine supported system and O2 can also be stored;
    otherwise O2 may come from the oxide surface.
    """

    def __init__(
        self,
        name: str,
        total_energy: float | None = None,
        net_magmom: float | None = None,
        geometry: Structure | Molecule | None = None,
        valence_electrons: dict[str, float] | None = None,
        *,
        nanocluster: Nanocluster,
        surface: Surface,
        atomic_references: Mapping[str, ChemicalEntity | float] | None = None,
        relaxed_references: Mapping[str, ChemicalEntity | float] | None = None,
        frozen_references: Mapping[str, ChemicalEntity | float] | None = None,
        pristine_reference: "SupportedSystem | None" = None,
        oxygen_reference: ChemicalEntity | None = None,
    ):
        """Create a supported system and validate its components and references.

        Args:
            name: Name of the supported system.
            total_energy: Total energy, or None if unavailable.
            net_magmom: Net magnetic moment, or None if unavailable.
            geometry: Combined geometry of the supported system.
            valence_electrons: Valence-electron counts keyed by element.
            nanocluster: Isolated nanocluster component.
            surface: Isolated surface component.
            atomic_references: Optional isolated-atom references keyed by element.
            relaxed_references: Optional relaxed supported references keyed by
                nanocluster species.
            frozen_references: Optional frozen supported references keyed by
                nanocluster species.
            pristine_reference: Optional pristine supported system for vacancy
                calculations.
            oxygen_reference: Optional O2 reference for vacancy calculations.

        Raises:
            TypeError: If a component or vacancy reference has an unsupported type.
            MissingGeometryError: If a required component or entity reference
                lacks geometry.
            MissingDataError: If a validated entity reference lacks total energy.
            MismatchedGeometryError: If a frozen reference does not match the
                supported-system geometry.
            StoichiometryError: If component or reference compositions do not
                balance, or the oxygen reference is not O2.
            EnergyCalculationError: If a stored vacancy reference energy is
                non-finite.
        """
        if not isinstance(nanocluster, Nanocluster):
            raise TypeError("Nanocluster must be a Nanocluster.")
        if not isinstance(surface, Surface):
            raise TypeError("Surface must be a Surface.")
        self._nanocluster = nanocluster
        self._surface = surface
        super().__init__(
            name, total_energy, net_magmom, geometry, valence_electrons,
            atomic_references=atomic_references,
        )
        self._validate_components()
        self._relaxed_references = _SUPPORTED_RELAXED.store(self, relaxed_references)
        self._frozen_references = _SUPPORTED_FROZEN.store(self, frozen_references)
        if pristine_reference is not None:
            if not isinstance(pristine_reference, SupportedSystem):
                raise TypeError("Pristine must be a SupportedSystem.")
            if not isinstance(pristine_reference.surface, MetalOxide):
                raise TypeError("Pristine must have a MetalOxide surface.")
            pristine_reference._validate_components()
            _validate_finite_reference_energy(pristine_reference.total_energy, "Pristine")
        if oxygen_reference is not None:
            _validate_oxygen_reference(oxygen_reference)
            _validate_finite_reference_energy(oxygen_reference.total_energy, "Oxygen")
        self._pristine_reference = pristine_reference
        self._oxygen_reference = oxygen_reference
        if pristine_reference is not None and oxygen_reference is not None:
            oxygen_difference = _infer_oxygen_vacancies(
                pristine_reference.surface.composition, self.surface.composition
            )
            self._validate_vacancy_composition(
                pristine_reference, oxygen_reference, oxygen_difference
            )

    @property
    def nanocluster(self) -> Nanocluster:
        """The isolated nanocluster used as a component reference."""
        return self._nanocluster

    @property
    def surface(self) -> Surface:
        """The isolated surface used as a component reference."""
        return self._surface

    @property
    def relaxed_references(self) -> dict[str, ChemicalEntity | float]:
        """Stored relaxed supported references, keyed by nanocluster element."""
        return self._relaxed_references.copy()

    @property
    def frozen_references(self) -> dict[str, ChemicalEntity | float]:
        """Stored frozen supported references, keyed by nanocluster element."""
        return self._frozen_references.copy()

    @property
    def pristine_reference(self) -> "SupportedSystem | None":
        """Stored pristine supported system for vacancy calculations."""
        return self._pristine_reference

    @property
    def oxygen_reference(self) -> ChemicalEntity | None:
        """Stored O2 reference for vacancy calculations."""
        return self._oxygen_reference

    @property
    def _mixing_atomic_fractions(self) -> dict[str, float]:
        """Return the atomic fractions of the nanocluster component."""
        return self.nanocluster.atomic_fractions

    def _validate_components(self) -> None:
        """Require the system composition to equal its surface and cluster parts.

        Raises:
            MissingGeometryError: If the supported system or a component lacks
                geometry.
            StoichiometryError: If the component compositions do not balance
                with the supported-system composition.
        """
        check_mass_balance(self, {"nanocluster": self.nanocluster, "surface": self.surface})

    def adsorption_energy(self) -> float:
        """Return E_supported - E_surface - E_nanocluster."""
        self._validate_components()
        return calculate_adsorption_energy(
            self.total_energy, self.surface.total_energy, self.nanocluster.total_energy
        )

    def cohesive_energy(self, **named_atoms: ChemicalEntity | float) -> float:
        """Return the cohesive energy per atom of the complete supported system."""
        self._validate_components()
        return super().cohesive_energy(**named_atoms)

    def ideal_cohesive_energy(
        self,
        *,
        atomic_references: Mapping[str, ChemicalEntity | float] | None = None,
        **named_clusters: tuple[ChemicalEntity | float, ChemicalEntity | float],
    ) -> float:
        """Average complete-reference cohesive energies using cluster fractions.

        Pairs map cluster species to (supported monometallic reference, atom).
        Atomic references for surface species may be stored at construction or
        supplied through `atomic_references`; pairs override them for this call.
        """
        clusters = {name: pair[0] for name, pair in named_clusters.items()}
        atoms = dict(atomic_references) if atomic_references is not None else {}
        atoms.update({name: pair[1] for name, pair in named_clusters.items()})
        self._validate_components()
        cluster_refs = _SUPPORTED_RELAXED.resolve(
            self, clusters, stored=self._relaxed_references
        )
        atom_refs = ATOMIC.resolve(self, atoms, stored=self._atomic_references)
        surface_comp = self.surface.composition
        cluster_size = sum(self.nanocluster.composition.values())
        cohesive = {}
        for symbol, energy in cluster_refs.items():
            counts = dict(surface_comp)
            counts[symbol] = counts.get(symbol, 0) + cluster_size
            cohesive[symbol] = calculate_cohesive_energy(
                energy, atom_refs, counts
            )
        return calculate_ideal_cohesive_energy(cohesive, self._mixing_atomic_fractions)

    def mix_cohesive_energy(
        self,
        *,
        atomic_references: Mapping[str, ChemicalEntity | float] | None = None,
        **named_clusters: tuple[ChemicalEntity | float, ChemicalEntity | float],
    ) -> float:
        """Return complete-system cohesion minus the ideal cohesive energy."""
        atoms = dict(atomic_references) if atomic_references is not None else {}
        atoms.update({name: pair[1] for name, pair in named_clusters.items()})
        return self.cohesive_energy(**atoms) - self.ideal_cohesive_energy(
            atomic_references=atomic_references, **named_clusters
        )

    def mixing_energy(self, **named_clusters: ChemicalEntity | float) -> float:
        """Return mixing energy using relaxed complete-system references.

        References are keyed by nanocluster species and weighted by their
        nanocluster fractions; supplied values override stored references.
        """
        self._validate_components()
        cluster_refs = _SUPPORTED_RELAXED.resolve(
            self, named_clusters, stored=self._relaxed_references
        )
        return calculate_mixing_energy(
            alloy_energy=self.total_energy,
            mono_energies=cluster_refs,
            atomic_fractions=self._mixing_atomic_fractions,
        )

    def strain_mixing_energy(
        self, **named_clusters: tuple[ChemicalEntity | float, ChemicalEntity | float]
    ) -> float:
        """Return strain mixing energy using (frozen, relaxed) supported references.

        Entity references contain the complete support and monometallic
        cluster; frozen geometries must match the complete system.
        """
        frozen_clusters = {name: pair[0] for name, pair in named_clusters.items()}
        relaxed_clusters = {name: pair[1] for name, pair in named_clusters.items()}
        self._validate_components()
        frozen_refs = _SUPPORTED_FROZEN.resolve(
            self, frozen_clusters, stored=self._frozen_references
        )
        relaxed_refs = _SUPPORTED_RELAXED.resolve(
            self, relaxed_clusters, stored=self._relaxed_references
        )
        return calculate_strain_mixing_energy(
            frozen_energies=frozen_refs,
            relaxed_energies=relaxed_refs,
            atomic_fractions=self._mixing_atomic_fractions,
        )

    def interaction_mixing_energy(self, **named_clusters: ChemicalEntity | float) -> float:
        """Return interaction mixing energy using frozen complete-system references."""
        self._validate_components()
        frozen_refs = _SUPPORTED_FROZEN.resolve(
            self, named_clusters, stored=self._frozen_references
        )
        return calculate_interact_mixing_energy(
            alloy_energy=self.total_energy,
            frozen_energies=frozen_refs,
            atomic_fractions=self._mixing_atomic_fractions,
        )

    def vacancy_formation_energy(
        self,
        *,
        pristine: "SupportedSystem | None" = None,
        oxygen: ChemicalEntity | None = None,
        num_vacancies: int = 1,
    ) -> float:
        """Return oxygen-vacancy formation energy relative to a pristine supported system.

        Omitted references use those stored on this instance. If no O2 is
        stored here, the reference stored on the defective oxide is used.
        Both supported systems must balance with `num_vacancies / 2` O2.
        """
        if not isinstance(self.surface, MetalOxide):
            raise TypeError("Vacancy formation requires a MetalOxide surface.")
        pristine = pristine if pristine is not None else self.pristine_reference
        if pristine is None:
            raise MissingReferenceError("A pristine supported system is required.")
        if not isinstance(pristine, SupportedSystem):
            raise TypeError("Pristine must be a SupportedSystem.")
        if not isinstance(pristine.surface, MetalOxide):
            raise TypeError("Pristine must have a MetalOxide surface.")
        if oxygen is None:
            oxygen = self.oxygen_reference
        if oxygen is None:
            oxygen = self.surface.oxygen_reference
        if oxygen is None:
            raise MissingReferenceError("An O2 reference is required.")
        validate_vacancy_count(num_vacancies)
        _validate_oxygen_reference(oxygen)
        self._validate_components()
        pristine._validate_components()
        self._validate_vacancy_composition(pristine, oxygen, num_vacancies)
        references = MASS_BALANCE.resolve(
            pristine,
            {"defective": self, "oxygen": oxygen},
            coefficients={"defective": 1, "oxygen": num_vacancies / 2},
        )
        return calculate_vacancy_formation_energy(
            defective_energy=references["defective"],
            pristine_energy=pristine.total_energy,
            defective_site_energy=references["oxygen"] / 2,
            num_vacancies=num_vacancies,
        )

    def _validate_vacancy_composition(
        self, pristine: "SupportedSystem", oxygen: ChemicalEntity, num_vacancies: int
    ) -> None:
        """Validate cluster consistency and surface mass balance for vacancies.

        Args:
            pristine: Pristine supported system used as the vacancy reference.
            oxygen: O2 entity used to balance the removed oxygen.
            num_vacancies: Number of oxygen vacancies represented.

        Raises:
            MissingGeometryError: If a required composition cannot be read.
            StoichiometryError: If cluster compositions differ or the defective
                surface plus oxygen does not balance with the pristine surface.
        """
        if self.nanocluster.composition != pristine.nanocluster.composition:
            raise StoichiometryError("Pristine and defective nanoclusters must have the same composition.")
        check_mass_balance(
            pristine.surface,
            {"defective": self.surface, "oxygen": oxygen},
            coefficients={"defective": 1, "oxygen": num_vacancies / 2},
        )
