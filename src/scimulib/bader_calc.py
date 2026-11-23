import numpy as np

from pymatgen.core import Structure, Molecule

from scimulib.exceptions import (
    SizeMismatchError, InvalidValenceValueError, MissingValenceElementError, 
    ChargeNeutralityError, MagneticMomentMismatchError
)


def calculate_bader_net_charges(
    species_list: np.ndarray | list[str] | tuple[str], 
    bader_charges: np.ndarray | list[float] | tuple[float], 
    valence_electrons: dict[str, float]
) -> np.ndarray:
    """Calculate the net Bader charge for each atom given its species and Bader charge from ACF.dat.

    The net charge is computed as `valence_electrons - bader_charges`, following
    the standard chemical convention where positive values indicate electron
    depletion (cations) and negative values indicate electron accumulation (anions).

    Args:
        species_list (array_like): A 1D array-like sequence of element symbols (e.g., 'Ti', 'O')
            whose ordering matches `bader_charges` from ACF.dat.
        bader_charges (array_like): A 1D array-like sequence of Bader electron
            counts (e.g., from ACF.dat), with the same length and atomic
            ordering as `species_list`.
        valence_electrons (dict): Mapping of element symbols (e.g., 'Ti', 'O')
            to their number of valence electrons as floats (e.g., ZVAL values
            from the pseudopotential).

    Returns:
        numpy.ndarray: A 1D float array containing the net Bader charge for each
        atom in `species_list`.

    Raises:
        TypeError: If `species_list` is not a 1D array-like sequence of element symbols.
        ValueError: If `bader_charges` is not a 1D array or contains 
            non-finite values.
        SizeMismatchError: If the length of `species_list` does not match the length of `bader_charges`.
        InvalidValenceValueError: If any value in `valence_electrons` is not a
            finite float (e.g., NaN, inf, or a non-float type).
        MissingValenceElementError: If any element present in `species_list` is
            missing from `valence_electrons`.
    """
    species_list = np.asarray(species_list, dtype=str)
    if species_list.ndim != 1:
        raise ValueError(f"The sequence of species in the geometry must be a 1D array, got {species_list.ndim}D.")

    bader_charges = np.asarray(bader_charges, dtype=float)
    if bader_charges.ndim != 1:
        raise ValueError(f"The array of Bader charges must be a 1D array, got {bader_charges.ndim}D.")

    if not np.all(np.isfinite(bader_charges)):
        raise ValueError("The array of Bader charges contains non-finite values (NaN or inf).")

    if len(species_list) != len(bader_charges):
        raise SizeMismatchError(
            f"Size mismatch: species_list has {len(species_list)} atoms, "
            f"but bader_charges has {len(bader_charges)} elements."
        )

    for element, value in valence_electrons.items():
        if not isinstance(value, float) or not np.isfinite(value):
            raise InvalidValenceValueError(
                f"Valence value for '{element}' ({value!r}) is not a valid finite float."
            )

    missing_elements = set(species_list) - set(valence_electrons.keys())

    if missing_elements:
        raise MissingValenceElementError(
            f"The following species are missing from the valence electron dictionary: "
            f"{sorted(missing_elements)}"
        )

    valence_array = np.array([valence_electrons[el] for el in species_list], dtype=float)

    return valence_array - bader_charges


def _sum_bader_property_by_species(
    species_list: np.ndarray | list[str] | tuple[str], 
    bader_property: np.ndarray | list[float] | tuple[float],
    bader_property_name: str = "Bader property",
    type: np.ndarray | list[str] | tuple[str] | None = None
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Sum a Bader atomic property for each species present in a Pymatgen geometry.

    If the geometry contains a 'type' site property (e.g., 'solid' and
    'adsorbate_CO'), the sums are grouped into a nested dictionary by type
    and then by species symbol. Otherwise, a flat dictionary mapping each
    species symbol to its summed property is returned.

    Args:
        species_list (array_like): A 1D array-like sequence of species symbols
            whose ordering matches `bader_property`.
        bader_property (array_like): A 1D array-like sequence of Bader values
            (e.g., net charges or magnetic moments) with the same length and
            atomic ordering as `species_list`.
        bader_property_name (str, optional): The name of the Bader property 
            being summed. Defaults to "Bader property".
        type (array_like, optional): A 1D array-like sequence of type
            labels (e.g., 'solid' or 'adsorbate_CO') with the same length and
            atomic ordering as `species_list`. Defaults to None.

    Returns:
        dict: A dictionary of the form `{species_symbol: float_sum}`.
        If `type` information is provided separately, a nested dictionary
        of the form `{type_name: {species_symbol: float_sum}}`.

    Raises:
        TypeError: If `species_list` is not an array-like sequence of species symbols.
        ValueError: If `bader_property` is not a 1D array or contains non-finite
            values.
        SizeMismatchError: If the length of `species_list` does not
            match the length of `bader_property` and `type` if provided.
    """
    species_list = np.asarray(species_list, dtype=str)
    if species_list.ndim != 1:
        raise ValueError(f"The sequence of species in the geometry must be a 1D array, got {species_list.ndim}D.")

    bader_property = np.asarray(bader_property, dtype=float)
    if bader_property.ndim != 1:
        raise ValueError(f"The {bader_property_name} array must be a 1D array, got {bader_property.ndim}D.")

    if not np.all(np.isfinite(bader_property)):
        raise ValueError(f"The {bader_property_name} array contains non-finite values (NaN or inf).")

    if len(species_list) != len(bader_property):
        raise SizeMismatchError(
            f"Size mismatch: The number of atoms in the geometry is {len(species_list)}, "
            f"but {bader_property_name} has {len(bader_property)} elements."
        )

    result = {}

    if type is not None:
        type = np.asarray(type, dtype=str)
        if type.ndim != 1:
            raise ValueError(f"The type array must be a 1D array, got {type.ndim}D.")
        if len(type) != len(species_list):
            raise SizeMismatchError(
                f"Size mismatch: the type array has {len(type)} elements, "
                f"but the number of atoms in the geometry is {len(species_list)}."
            )
        for t, species, value in zip(type, species_list, bader_property):
            if t not in result:
                result[t] = {}
            result[t][species] = float(result[t].get(species, 0.0) + value)
    else:
        for species, value in zip(species_list, bader_property):
            result[species] = float(result.get(species, 0.0) + value)

    return result


def sum_bader_net_charges(
    species_list: np.ndarray | list[str] | tuple[str], 
    net_bader_charges: np.ndarray | list[float] | tuple[float], 
    atol: float = 1e-3
):
    """Sum net Bader charges per species and verify system charge neutrality.

    Wraps `sum_bader_property_by_species` and validates that the total net charge
    of the system is zero within the specified absolute tolerance.

    Args:
        species_list (array_like): A 1D array-like sequence of atomic species whose ordering 
            matches `net_bader_charges`.
        net_bader_charges (array_like): A 1D array-like sequence of net Bader
            charges for each atom in `geometry`.
        atol (float, optional): Absolute tolerance allowed when checking that
            the total net charge equals zero. Defaults to 1e-3.

    Returns:
        dict: Summed net Bader charges grouped by species (or by component and
        species if 'component' is present in `geometry.site_properties`).

    Raises:
        TypeError: If `species_list` is not a valid array-like sequence.
        ValueError: If `net_bader_charges` is not a valid 1D finite array.
        SizeMismatchError: If `species_list` and `net_bader_charges` differ in
            length.
        ChargeNeutralityError: If the sum of `net_bader_charges` deviates from
            zero by more than `atol`.
    """
    charges_by_species = _sum_bader_property_by_species(species_list, net_bader_charges, bader_property_name="net Bader charges")
    total_charge = float(np.sum(net_bader_charges))

    if not np.isclose(total_charge, 0.0, atol=atol):
        raise ChargeNeutralityError(
            f"System is not charge-neutral: sum of net Bader charges is "
            f"{total_charge:.6f} (expected 0.0 within atol={atol})."
        )

    return charges_by_species


def sum_bader_magnetic_moments(
    species_list: np.ndarray | list[str] | tuple[str], 
    bader_magnetic_moments: np.ndarray | list[float] | tuple[float], 
    total_magnetic_moment: float, 
    atol: float = 1e-3
):
    """Sum Bader magnetic moments per species and verify against the total magnetic moment.

    Wraps `sum_bader_property_by_species` and validates that the sum of atomic
    Bader magnetic moments matches `total_magnetic_moment` within the specified
    absolute tolerance.

    Args:
        species_list (array_like): A 1D array-like sequence of atomic species whose ordering 
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
        SizeMismatchError: If `species_list` and `bader_magnetic_moments`
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

    magmoms_by_species = _sum_bader_property_by_species(species_list, bader_magnetic_moments, bader_property_name="Bader magnetic moments")
    calculated_total = float(np.sum(bader_magnetic_moments))

    if not np.isclose(calculated_total, total_magnetic_moment, atol=atol):
        raise MagneticMomentMismatchError(
            f"Magnetic moment mismatch: sum of Bader magnetic moments is "
            f"{calculated_total:.6f}, expected {total_magnetic_moment:.6f} "
            f"(within atol={atol})."
        )

    return magmoms_by_species