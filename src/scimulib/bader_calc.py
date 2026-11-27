import numpy as np

from scimulib.exceptions import (
    SizeMismatchError, InvalidValenceValueError, MissingValenceElementError, 
    ChargeNeutralityError, MagneticMomentMismatchError
)


def _validate_tolerance(atol: float) -> None:
    if (
        isinstance(atol, bool)
        or not isinstance(atol, (int, float))
        or not np.isfinite(atol)
        or atol < 0
    ):
        raise ValueError(
            f"Absolute tolerance must be finite and non-negative, got {atol!r}."
        )


def calculate_bader_net_charges(
    species_list: np.ndarray | list[str] | tuple[str, ...],
    bader_charges: np.ndarray | list[float] | tuple[float, ...],
    valence_electrons: dict[str, float]
) -> np.ndarray:
    """
    Calculate the Bader net charge for each atom given its species and Bader charge from ACF.dat.

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
        numpy.ndarray: A 1D float array containing the Bader net charge for each
        atom in `species_list`.

    Raises:
        ValueError: If `bader_charges` or `species_list` is not a 1D array. Additionally, 
            if `bader_charges` contains non-finite values (NaN or inf).
        SizeMismatchError: If the length of `species_list` does not match the 
            length of `bader_charges`.
        InvalidValenceValueError: If any value in `valence_electrons` is not a
            finite, non-negative real number.
        MissingValenceElementError: If any element present in `species_list` is
            missing from `valence_electrons`.
    """
    species_list = np.asarray(species_list, dtype=str)
    if species_list.ndim != 1:
        raise ValueError(
            f"The list of atom species in the geometry must be a 1D array, "
            f"got {species_list.ndim}D."
        )

    bader_charges = np.asarray(bader_charges, dtype=float)
    if bader_charges.ndim != 1:
        raise ValueError(
            f"The Bader charges must be a 1D array, got {bader_charges.ndim}D."
        )

    if not np.all(np.isfinite(bader_charges)):
        raise ValueError(
            "The Bader charges contains non-finite values (NaN or inf)."
        )

    if len(species_list) != len(bader_charges):
        raise SizeMismatchError(
            f"Size mismatch: the number of atoms is {len(species_list)}, "
            f"but the Bader charges array has {len(bader_charges)} elements."
        )

    for element, value in valence_electrons.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not np.isfinite(value)
            or value < 0
        ):
            raise InvalidValenceValueError(
                f"Valence value for '{element}' ({value!r}) must be finite and non-negative."
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
    species_list: np.ndarray | list[str] | tuple[str, ...],
    bader_property: np.ndarray | list[float] | tuple[float, ...],
    bader_property_name: str = "Bader property",
    type: np.ndarray | list[str] | tuple[str, ...] | None = None
) -> dict[str, float] | dict[str, dict[str, float]]:
    """
    Sum a Bader atomic property for each species present in a Pymatgen geometry.

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
        ValueError: If `bader_property`, `species_list`, or `type` is not a 
            1D array. Additionally, if `bader_property` contains non-finite 
            values.
        SizeMismatchError: If the length of `species_list` does not
            match the length of `bader_property` and `type` if provided.
    """
    species_list = np.asarray(species_list, dtype=str)
    if species_list.ndim != 1:
        raise ValueError(
            f"The list of atom species in the geometry must be a 1D array, "
            f"got {species_list.ndim}D."
        )

    bader_property = np.asarray(bader_property, dtype=float)
    if bader_property.ndim != 1:
        raise ValueError(
            f"The {bader_property_name} array must be a 1D array, got {bader_property.ndim}D."
        )

    if not np.all(np.isfinite(bader_property)):
        raise ValueError(
            f"The {bader_property_name} array contains non-finite values (NaN or inf)."
        )

    if len(species_list) != len(bader_property):
        raise SizeMismatchError(
            f"Size mismatch: The number of atoms in the geometry is {len(species_list)}, "
            f"but the {bader_property_name} array has {len(bader_property)} elements."
        )

    result = {}

    if type is not None:
        type = np.asarray(type, dtype=str)
        if type.ndim != 1:
            raise ValueError(
                f"The type array must be a 1D array, got {type.ndim}D."
            )
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
    species_list: np.ndarray | list[str] | tuple[str, ...],
    bader_net_charges: np.ndarray | list[float] | tuple[float, ...],
    type: np.ndarray | list[str] | tuple[str, ...] | None = None,
    atol: float = 1e-3,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """
    Sum net Bader charges per species and verify system charge neutrality.

    Wraps `sum_bader_property_by_species` and validates that the total net charge
    of the system is zero within the specified absolute tolerance.

    Args:
        species_list (array_like): A 1D array-like sequence of atomic species 
            whose ordering matches `bader_net_charges`.
        bader_net_charges (array_like): A 1D array-like sequence of Bader net
            charges for each atom in the geometry.
        type (array_like, optional): A 1D array-like sequence of types corresponding
            to each atom in the geometry. If provided, the summed charges will be
            grouped by type and species.
        atol (float, optional): Absolute tolerance allowed when checking that
            the total net charge equals zero. Defaults to 1e-3.

    Returns:
        dict: Summed net Bader charges grouped by species (or by type and species 
            if `type` is provided).

    Raises:
        ValueError: If `bader_net_charges`, `species_list`, or `type` is not a 
            valid 1D finite array. Additionally, if Bader net charges are not 
            finite.
        SizeMismatchError: If `species_list` and `bader_net_charges` differ in
            length.
        ChargeNeutralityError: If the sum of `bader_net_charges` deviates from
            zero by more than `atol`.
    """
    _validate_tolerance(atol)
    charge_values = np.asarray(bader_net_charges, dtype=float)
    charges_by_species = _sum_bader_property_by_species(
        species_list=species_list, 
        bader_property=charge_values,
        bader_property_name="net Bader charges", 
        type=type
    )
    total_charge = float(np.sum(charge_values))

    if not np.isclose(total_charge, 0.0, rtol=0.0, atol=atol):
        raise ChargeNeutralityError(
            f"System is not charge-neutral: sum of net Bader charges is "
            f"{total_charge:.6f} (expected 0.0 within atol={atol})."
        )

    return charges_by_species


def sum_bader_magnetic_moments(
    species_list: np.ndarray | list[str] | tuple[str, ...],
    bader_magnetic_moments: np.ndarray | list[float] | tuple[float, ...],
    total_magnetic_moment: float,
    type: np.ndarray | list[str] | tuple[str, ...] | None = None,
    atol: float = 1e-3,
) -> dict[str, float] | dict[str, dict[str, float]]:
    """Sum Bader magnetic moments per species and verify against the total magnetic moment.

    Wraps `sum_bader_property_by_species` and validates that the sum of atomic
    Bader magnetic moments matches `total_magnetic_moment` within the specified
    absolute tolerance.

    Args:
        species_list (array_like): A 1D array-like sequence of atomic species 
            whose ordering matches `bader_magnetic_moments`.
        bader_magnetic_moments (array_like): A 1D array-like sequence of Bader
            magnetic moments for each atom in the geometry.
        total_magnetic_moment (float): The expected net magnetic moment of the
            entire system (e.g., from OSZICAR or OUTCAR).
        type (array_like, optional): A 1D array-like sequence of types corresponding
            to each atom in the geometry. If provided, the summed moments will be
            grouped by type and species.
        atol (float, optional): Absolute tolerance allowed when comparing the
            summed moments to `total_magnetic_moment`. Defaults to 1e-3.

    Returns:
        dict: Summed Bader magnetic moments grouped by species, or by type and species
            when type labels are provided.

    Raises:
        ValueError: If `bader_magnetic_moments`, `species_list`, or `type` 
            is not a valid 1D finite array. Additionally, if Bader magnetic 
            moments are not finite numbers.
        SizeMismatchError: If `species_list` and `bader_magnetic_moments`
            differ in length.
        MagneticMomentMismatchError: If the sum of `bader_magnetic_moments`
            deviates from `total_magnetic_moment` by more than `atol`.
        ValueError: If `atol` is negative or non-finite, or if the expected
            total magnetic moment is non-finite.
    """
    _validate_tolerance(atol)
    moment_values = np.asarray(bader_magnetic_moments, dtype=float)
    if not np.isfinite(total_magnetic_moment):
        raise ValueError(
            f"Total magnetic moment must be finite, got {total_magnetic_moment!r}."
        )

    magmoms_by_species = _sum_bader_property_by_species(
        species_list=species_list, 
        bader_property=moment_values,
        bader_property_name="Bader magnetic moments", 
        type=type
    )
    calculated_total = float(np.sum(moment_values))

    if not np.isclose(calculated_total, total_magnetic_moment, rtol=0.0, atol=atol):
        raise MagneticMomentMismatchError(
            f"Magnetic moment mismatch: sum of Bader magnetic moments is "
            f"{calculated_total:.6f}, expected {total_magnetic_moment:.6f} "
            f"(within atol={atol})."
        )

    return magmoms_by_species