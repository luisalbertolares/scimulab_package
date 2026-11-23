import numpy as np

from pymatgen.core import Structure, Molecule

from scimulib.exceptions import (
    SizeMismatchError, InvalidValenceValueError, MissingValenceElementError, 
    ChargeNeutralityError, MagneticMomentMismatchError
)


def calculate_bader_net_charges(
    geometry: Structure | Molecule, 
    bader_charges: np.ndarray | list[float] | tuple[float], 
    valence_electrons: dict[str, float]
) -> np.ndarray:
    """Calculate the net Bader charge for each atom in a pymatgen Structure or Molecule.

    The net charge is computed as `valence_electrons - bader_charges`, following
    the standard chemical convention where positive values indicate electron
    depletion (cations) and negative values indicate electron accumulation (anions).

    Args:
        geometry (Structure | Molecule): The atomic geometry whose site ordering 
            matches `bader_charges`.
        bader_charges (array_like): A 1D array-like sequence of Bader electron
            counts (e.g., from ACF.dat), with the same length and atomic
            ordering as `geometry`.
        valence_electrons (dict): Mapping of element symbols (e.g., 'Ti', 'O')
            to their number of valence electrons as floats (e.g., ZVAL values
            from the pseudopotential).

    Returns:
        numpy.ndarray: A 1D float array containing the net Bader charge for each
        atom in `geometry`.

    Raises:
        TypeError: If `geometry` is not a pymatgen `Structure` or `Molecule`.
        ValueError: If `bader_charges` is not a 1D array or contains 
            non-finite values.
        SizeMismatchError: If the number of sites in `geometry` does not
            match the length of `bader_charges`.
        InvalidValenceValueError: If any value in `valence_electrons` is not a
            finite float (e.g., NaN, inf, or a non-float type).
        MissingValenceElementError: If any element present in `geometry` is
            missing from `valence_electrons`.
    """
    if not isinstance(geometry, (Structure, Molecule)):
        raise TypeError("geometry must be a pymatgen Structure or Molecule object.")

    bader_charges = np.asarray(bader_charges, dtype=float)
    if bader_charges.ndim != 1:
        raise ValueError(f"bader_charges must be a 1D array, got {bader_charges.ndim}D.")

    if not np.all(np.isfinite(bader_charges)):
        raise ValueError("bader_charges contains non-finite values (NaN or inf).")

    if len(geometry) != len(bader_charges):
        raise SizeMismatchError(
            f"Size mismatch: geometry has {len(geometry)} atoms, "
            f"but bader_charges has {len(bader_charges)} elements."
        )

    for element, value in valence_electrons.items():
        if not isinstance(value, float) or not np.isfinite(value):
            raise InvalidValenceValueError(
                f"Valence value for '{element}' ({value!r}) is not a valid finite float."
            )

    geometry_symbols = [site.specie.symbol for site in geometry]
    missing_elements = set(geometry_symbols) - set(valence_electrons.keys())

    if missing_elements:
        raise MissingValenceElementError(
            f"The following geometry elements are missing from valence_electrons: "
            f"{sorted(missing_elements)}"
        )

    valence_array = np.array([valence_electrons[el] for el in geometry_symbols], dtype=float)

    return valence_array - bader_charges


def _sum_bader_property_by_species(
    geometry: Structure | Molecule, 
    bader_property: np.ndarray | list[float] | tuple[float]
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Sum a Bader atomic property for each species present in a Pymatgen geometry.

    If `geometry` contains a 'component' site property (e.g., 'solid' and
    'adsorbate_CO'), the sums are grouped into a nested dictionary by component
    and then by species symbol. Otherwise, a flat dictionary mapping each
    species symbol to its summed property is returned.

    Args:
        geometry (Structure | Molecule): The atomic geometry whose site ordering 
            matches `bader_property`.
        bader_property (array_like): A 1D array-like sequence of Bader values
            (e.g., net charges or magnetic moments) with the same length and
            atomic ordering as `geometry`.

    Returns:
        dict: If 'component' is not in `geometry.site_properties`, a dictionary
        of the form `{species_symbol: float_sum}`. If 'component' is present,
        a nested dictionary of the form `{component_name: {species_symbol: float_sum}}`.

    Raises:
        TypeError: If `geometry` is not a pymatgen `Structure` or `Molecule`.
        ValueError: If `bader_property` is not a 1D array or contains non-finite
            values.
        SizeMismatchError: If the number of sites in `geometry` does not
            match the length of `bader_property`.
    """
    if not isinstance(geometry, (Structure, Molecule)):
        raise TypeError("geometry must be a pymatgen Structure or Molecule object.")

    bader_property = np.asarray(bader_property, dtype=float)
    if bader_property.ndim != 1:
        raise ValueError(f"bader_property must be a 1D array, got {bader_property.ndim}D.")

    if not np.all(np.isfinite(bader_property)):
        raise ValueError("bader_property contains non-finite values (NaN or inf).")

    if len(geometry) != len(bader_property):
        raise SizeMismatchError(
            f"Size mismatch: geometry has {len(geometry)} atoms, "
            f"but bader_property has {len(bader_property)} elements."
        )

    has_component = "component" in geometry.site_properties
    result = {}

    if has_component:
        for site, value in zip(geometry, bader_property):
            component = site.properties["component"]
            symbol = site.specie.symbol
            if component not in result:
                result[component] = {}
            result[component][symbol] = float(result[component].get(symbol, 0.0) + value)
    else:
        for site, value in zip(geometry, bader_property):
            symbol = site.specie.symbol
            result[symbol] = float(result.get(symbol, 0.0) + value)

    return result


def sum_bader_net_charges(
    geometry: Structure | Molecule, 
    net_bader_charges: np.ndarray | list[float] | tuple[float], 
    atol: float = 1e-3
):
    """Sum net Bader charges per species and verify system charge neutrality.

    Wraps `sum_bader_property_by_species` and validates that the total net charge
    of the system is zero within the specified absolute tolerance.

    Args:
        geometry (Structure | Molecule): The atomic geometry whose site ordering 
            matches `net_bader_charges`.
        net_bader_charges (array_like): A 1D array-like sequence of net Bader
            charges for each atom in `geometry`.
        atol (float, optional): Absolute tolerance allowed when checking that
            the total net charge equals zero. Defaults to 1e-3.

    Returns:
        dict: Summed net Bader charges grouped by species (or by component and
        species if 'component' is present in `geometry.site_properties`).

    Raises:
        TypeError: If `geometry` is not a pymatgen `Structure` or `Molecule`.
        ValueError: If `net_bader_charges` is not a valid 1D finite array.
        SizeMismatchError: If `geometry` and `net_bader_charges` differ in
            length.
        ChargeNeutralityError: If the sum of `net_bader_charges` deviates from
            zero by more than `atol`.
    """
    charges_by_species = _sum_bader_property_by_species(geometry, net_bader_charges)
    total_charge = float(np.sum(net_bader_charges))

    if not np.isclose(total_charge, 0.0, atol=atol):
        raise ChargeNeutralityError(
            f"System is not charge-neutral: sum of net Bader charges is "
            f"{total_charge:.6f} (expected 0.0 within atol={atol})."
        )

    return charges_by_species


def sum_bader_magnetic_moments(
    geometry: Structure | Molecule, 
    bader_magnetic_moments: np.ndarray | list[float] | tuple[float], 
    total_magnetic_moment: float, 
    atol: float = 1e-3
):
    """Sum Bader magnetic moments per species and verify against the total magnetic moment.

    Wraps `sum_bader_property_by_species` and validates that the sum of atomic
    Bader magnetic moments matches `total_magnetic_moment` within the specified
    absolute tolerance.

    Args:
        geometry (Structure | Molecule): The atomic geometry whose site ordering 
            matches `bader_magnetic_moments`.
        bader_magnetic_moments (array_like): A 1D array-like sequence of Bader
            magnetic moments for each atom in `geometry`.
        total_magnetic_moment (float): The expected net magnetic moment of the
            entire system (e.g., from OSZICAR or OUTCAR).
        atol (float, optional): Absolute tolerance allowed when comparing the
            summed moments to `total_magnetic_moment`. Defaults to 1e-3.

    Returns:
        dict: Summed Bader magnetic moments grouped by species (or by component
        and species if 'component' is present in `geometry.site_properties`).

    Raises:
        TypeError: If `geometry` is not a pymatgen `Structure` or `Molecule`, or
            if `total_magnetic_moment` is not a finite number.
        ValueError: If `bader_magnetic_moments` is not a valid 1D finite array.
        SizeMismatchError: If `geometry` and `bader_magnetic_moments`
            differ in length.
        MagneticMomentMismatchError: If the sum of `bader_magnetic_moments`
            deviates from `total_magnetic_moment` by more than `atol`.
    """
    if (
        isinstance(total_magnetic_moment, bool) 
        or not isinstance(total_magnetic_moment, (int, float)) 
        or not np.isfinite(total_magnetic_moment)
    ):
        raise TypeError(
            f"total_magnetic_moment ({total_magnetic_moment!r}) must be a finite number."
        )

    magmoms_by_species = _sum_bader_property_by_species(geometry, bader_magnetic_moments)
    calculated_total = float(np.sum(bader_magnetic_moments))

    if not np.isclose(calculated_total, total_magnetic_moment, atol=atol):
        raise MagneticMomentMismatchError(
            f"Magnetic moment mismatch: sum of Bader magnetic moments is "
            f"{calculated_total:.6f}, expected {total_magnetic_moment:.6f} "
            f"(within atol={atol})."
        )

    return magmoms_by_species