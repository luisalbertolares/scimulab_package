from collections.abc import Mapping

import math
import numpy as np

from scimulib.exceptions import EnergyCalculationError, StoichiometryError, MissingReferenceError


def _calculate_energy_difference(
    final_energy: float,
    reference_energies: dict[str, float],
    coefficients: dict[str, float],
) -> float:
    """Computes a generic relative energy against a linear combination of references.

    Performs the operation:
        E_rel = E_final - sum(coeff_i * E_ref_i)

    Args:
        final_energy: Total energy of the final state (e.g., cluster, bulk, or
            adsorbed complex) in consistent energy units (typically eV).
        reference_energies: Dictionary of component/species identifiers to their
            reference energies.
        coefficients: Dictionary of component/species identifiers to their weighting
            or stoichiometric coefficients in the initial state.

    Returns:
        float: The relative energy difference (final state minus weighted initial state).

    Raises:
        StoichiometryError: If `coefficients` is empty or any coefficient is
            non-finite (NaN or Inf).
        MissingReferenceError: If `reference_energies` is empty or any key in 
            `coefficients` is missing from `reference_energies`.
        EnergyCalculationError: If `final_energy` or any used reference energy
            is non-finite (NaN or Inf).
    """
    if not reference_energies:
        raise MissingReferenceError(
            "Reference energies dictionary cannot be empty.", missing_species=[]
        )

    if not coefficients:
        raise StoichiometryError("Coefficients dictionary cannot be empty.")

    if not math.isfinite(final_energy):
        raise EnergyCalculationError(f"Final energy must be finite, got {final_energy}.")

    missing_refs = sorted(coefficients.keys() - reference_energies.keys())
    if missing_refs:
        raise MissingReferenceError(
            f"Missing reference energies for: {missing_refs}",
            missing_species=missing_refs,
        )

    terms = []
    for key, coeff in coefficients.items():
        if not math.isfinite(coeff):
            raise StoichiometryError(
                f"Stoichiometric coefficient for '{key}' must be finite, got {coeff}."
            )
        ref_energy = reference_energies[key]
        if not math.isfinite(ref_energy):
            raise EnergyCalculationError(
                f"Reference energy for '{key}' must be finite, got {ref_energy}."
            )
        terms.append(coeff * ref_energy)

    return final_energy - math.fsum(terms)


def _calculate_weighted_energy(
    system_energy: float,
    reference_energies: dict[str, float],
    atomic_fractions: dict[str, float],
) -> float:
    """Calculates relative energy using fractional composition weights.

    If `system_energy` is per atom or per formula unit, the reference energies
    should be on the same basis. Designed for quantities such as mixing, excess,
    or cohesive energies where reference energies are weighted by their
    respective atomic fractions:
        E_rel = E_system - sum(x_i * E_ref_i),  with sum(x_i) == 1.0

    Args:
        system_energy: Energy of the final state of the system.
        reference_energies: Dictionary of element symbols to their reference
            energies.
        atomic_fractions: Dictionary of element symbols to their atomic fractions
            in the system. Values must be non-negative and sum to 1.0 within a
            tolerance of 1e-4.

    Returns:
        float: The composition-weighted relative energy in the same units as the inputs.

    Raises:
        StoichiometryError: If `atomic_fractions` is empty, contains negative
            or non-finite values, or its sum deviates from 1.0 by more than 1e-4.
        MissingReferenceError: If any element in `atomic_fractions` is missing
            from `reference_energies`.
        EnergyCalculationError: If any input energy is non-finite.
    """
    if not atomic_fractions:
        raise StoichiometryError("Atomic fractions dictionary cannot be empty.")

    for species, fraction in atomic_fractions.items():
        if not math.isfinite(fraction) or fraction < 0.0:
            raise StoichiometryError(
                f"Atomic fraction for '{species}' must be non-negative and "
                f"finite, got {fraction}."
            )

    total_fraction = math.fsum(atomic_fractions.values())
    if not math.isclose(total_fraction, 1.0, rel_tol=0.0, abs_tol=1e-4):
        raise StoichiometryError(
            f"Atomic fractions must sum to 1.0. Current sum: {total_fraction:.4f}. "
            "Ensure you are passing fractions, not total atom counts."
        )

    return _calculate_energy_difference(system_energy, reference_energies, atomic_fractions)


def _calculate_interaction_energy(
    system_energy: float,
    reference_energies: dict[str, float],
    stoichiometry: dict[str, int | float] | None = None,
) -> float:
    """Calculates interaction, adsorption, or binding energy from reference subsystems.

    Designed for systems where the final state is formed by combining isolated
    fragments or fractional molecular references (e.g., a nanoparticle deposited
    on an oxide slab, or atomic H adsorption referenced to 0.5 * E(H2)):
        E_int = E_system - sum(n_i * E_ref_i)

    Args:
        system_energy: Total energy of the combined final system.
        reference_energies: Dictionary of subsystem or reference species names to
            the total energies of their isolated states.
        stoichiometry: Optional dictionary of reference names to positive
            stoichiometric coefficients (integers or floats) in the final
            system. If `None`, a stoichiometric coefficient of 1.0 is assumed
            for every entry in `reference_energies`.

    Returns:
        float: The total interaction/adsorption energy (negative values typically
            indicate thermodynamically favorable binding).

    Raises:
        StoichiometryError: If no reference subsystems are provided or if any
            stoichiometric coefficient is non-finite or less than or equal to zero.
        MissingReferenceError: If any subsystem specified in `stoichiometry` is
            missing from `reference_energies`.
        EnergyCalculationError: If any input energy is non-finite.
    """
    coeffs = (
        stoichiometry
        if stoichiometry is not None
        else dict.fromkeys(reference_energies, 1.0)
    )

    if not coeffs:
        raise StoichiometryError("At least one reference subsystem must be provided.")

    for subsystem, count in coeffs.items():
        if not math.isfinite(count) or count <= 0.0:
            raise StoichiometryError(
                f"Stoichiometric coefficient for '{subsystem}' must be positive "
                f"and finite, got {count}."
            )

    return _calculate_energy_difference(system_energy, reference_energies, coeffs)


def calculate_cohesive_energy(
    total_energy: float, 
    atomic_energies: dict[str, float], 
    atomic_counts: dict[str, int]
) -> float:
    """
    Calculates the cohesive energy per atom for a multi-atomic system.

    Args:
        total_energy (float): The total energy of the system.
        atomic_energies (dict[str, float]): Reference energies for each atomic species.
        atomic_counts (dict[str, int]): Number of atoms for each species in the system.

    Returns:
        float: The cohesive energy per atom (eV/atom).

    Raises:
        StoichiometryError: If any stoichiometric coefficient is non-finite or less 
            than or equal to zero.
        MissingReferenceError: If any atomic species in `atomic_counts` is missing 
            from `atomic_energies`.
        EnergyCalculationError: If any input energy is non-finite.

    Note:
        Formula: E_coh = (sum(x_i * E_atom_i) - E_total) / total_atoms
    """
    total_atoms = sum(atomic_counts.values())
    return - _calculate_interaction_energy(
        system_energy=total_energy,
        reference_energies=atomic_energies,
        stoichiometry=atomic_counts
    ) / total_atoms

def calculate_ideal_cohesive_energy(
    mono_cohesive_energies: dict[str, float], 
    atomic_fractions: dict[str, float]
) -> float:
    """
    Calculates the ideal cohesive energy (without mixing interaction) for a 
    multi-metallic cluster.

    This represents the weighted average of the pure monometallic cohesive 
    energies.

    Args:
        mono_cohesive_energies (dict[str, float]): Cohesive energies of the 
            pure monometallic clusters.
            Example: {'Au': -3.12, 'Co': -4.39, 'Cu': -3.49}.
        atomic_fractions (dict[str, float]): Atomic fractions of each metal 
            in the cluster.
            Example: {'Au': 0.5, 'Co': 0.25, 'Cu': 0.25}.

    Returns:
        float: The ideal cohesive energy (eV/atom).

    Raises:
        StoichiometryError: If any atomic fraction is non-finite or less than zero. 
        MissingReferenceError: If any atomic species in `atomic_fractions` is missing 
            from `mono_cohesive_energies`.
        EnergyCalculationError: If any input energy is non-finite.

    Note:
        Formula: E_ideal_coh = sum(x_i * E_coh_i)
    """
    neg_coh_energies = {subsys: -energy for subsys, energy in mono_cohesive_energies.items()}
    return _calculate_weighted_energy(0.0, neg_coh_energies, atomic_fractions)


def calculate_mixing_cohesive_energy(mixing_energy: float, total_atoms: int) -> float:
    """
    Calculates the cohesive energy of mixing for a cluster.

    Args:
        mixing_energy (float): The total mixing energy of the cluster.
        total_atoms (int): The total number of atoms in the cluster.

    Returns:
        float: The cohesive energy of mixing (eV/atom).

    Raises:
        StoichiometryError: If the total number of atoms is zero or negative.

    Note:
        Formula: E_mix_coh = - mixing_energy / total_atoms
    """
    if total_atoms <= 0:
        raise StoichiometryError(
            "The total number of atoms must be greater than zero. "
            f"Current value: {total_atoms}"
        )
    
    return -(mixing_energy / total_atoms)


def calculate_mixing_energy(
    alloy_energy: float, 
    mono_energies: dict[str, float], 
    atomic_fractions: dict[str, float]
) -> float:
    """
    Calculates the total mixing energy for a multi-metallic cluster.

    The mixing energy measures the relative stability of the alloy against
    segregation into its pure monometallic components of equal size.

    Args:
        alloy_energy (float): Total energy of the relaxed mixed cluster (alloy).
        mono_energies (dict[str, float]): Total energies of the fully relaxed 
            pure monometallic clusters. Example: {'Au': -150.2, 'Co': -180.5}.
        atomic_fractions (dict[str, float]): Atomic fractions of each metal in the cluster.

    Returns:
        float: The mixing energy in the same units as the inputs (typically eV).

    Raises:
        StoichiometryError: If the atomic fractions do not sum to 1.
        MissingReferenceError: If a required reference energy for a monometallic 
            component is missing.
        EnergyCalculationError: If any input energy is non-finite.

    Note:
        Formula: E_mix = E_alloy - sum(x_i * E_mono_i)
    """
    return _calculate_weighted_energy(alloy_energy, mono_energies, atomic_fractions)


def calculate_interact_mixing_energy(
    alloy_energy: float, 
    frozen_energies: dict[str, float], 
    atomic_fractions: dict[str, float]
) -> float:
    """
    Calculates the interaction contribution to the mixing energy.

    This property isolates the purely chemical and electronic interaction energy 
    by artificially removing structural relaxation effects.

    Args:
        alloy_energy (float): Total energy of the relaxed mixed cluster (alloy).
        frozen_energies (dict[str, float]): Energies of the pure monometallic clusters 
            with frozen geometries (unrelaxed, matching the alloy's structural lattice).
        atomic_fractions (dict[str, float]): Atomic fractions of each metal in the cluster.

    Returns:
        float: Interaction contribution to the mixing energy (eV).

    Raises:
        StoichiometryError: If the atomic fractions do not sum to 1.
        MissingReferenceError: If a required reference energy for a monometallic 
            component is missing.
        EnergyCalculationError: If any input energy is non-finite.

    Note:
        Formula: E_interact = E_mix - sum(x_i * E_frozen_i)
        By definition, E_mix = E_interact + E_strain.
    """
    return _calculate_weighted_energy(alloy_energy, frozen_energies, atomic_fractions)


def calculate_strain_mixing_energy(
    frozen_energies: dict[str, float], 
    relaxed_energies: dict[str, float], 
    atomic_fractions: dict[str, float]
) -> float:
    """
    Calculates the strain contribution to the mixing energy in a multi-metallic cluster.

    This contribution reflects how much of the mixing energy is consumed or released 
    by the structural deformation required to accommodate different atomic sizes 
    in the alloy lattice.

    Args:
        frozen_energies (dict[str, float]): Energies of the monometallic clusters 
            with frozen geometries (matching the alloy's structure).
        relaxed_energies (dict[str, float]): Energies of the fully relaxed pure 
            monometallic clusters.
        atomic_fractions (dict[str, float]): Atomic fractions of each metal in the cluster.

    Returns:
        float: Strain contribution to the mixing energy (eV).

    Raises:
        StoichiometryError: If the atomic fractions do not sum to 1.
        MissingReferenceError: If a required reference energy for a 
            monometallic component is missing or an atomic fraction 
            is missing.
        EnergyCalculationError: If any input energy is non-finite.

    Note:
        Formula: E_strain = sum(x_i * (E_frozen_i - E_relaxed_i))
        By definition, E_mix = E_interact + E_strain.
    """
    # Validate missing references across both dictionaries
    elements = set(atomic_fractions.keys())
    if (not elements.issubset(frozen_energies.keys()) 
        or not elements.issubset(relaxed_energies.keys())
    ):
        raise MissingReferenceError(
            "Missing frozen or relaxed reference energies for one or more elements."
        )

    neg_relaxed_energies = {subsys: -energy for subsys, energy in relaxed_energies.items()}
    strain_energy_frozen = _calculate_weighted_energy(0.0, frozen_energies, atomic_fractions)
    strain_energy_relaxed = _calculate_weighted_energy(0.0, neg_relaxed_energies, atomic_fractions)
    
    return strain_energy_frozen + strain_energy_relaxed


def calculate_adsorption_energy(
    total_energy: float, 
    adsorbent: float, 
    *adsorbate_energies: float
) -> float:
    """
    Calculates the adsorption energy of an adsorbate on a surface.

    Args:
        total_energy (float): Total energy of the adsorbate-surface system.
        adsorbent (float): Energy of the clean surface.
        adsorbate_energies (float | dict[str, float]): Energy of the adsorbate(s). 
            Can be a single float or a dictionary of adsorbate energies.

    Returns:
        float: Adsorption energy (eV).

    Raises:
        TypeError: If the adsorbate energies are not provided as a float or a dictionary.
        StoichiometryError: If the adsorbate nor clean surface energies are 
            provided correctly.
        MissingReferenceError: If a required reference energy for the clean 
            surface or adsorbate is missing.
        EnergyCalculationError: If any input energy is non-finite.
        
    Note:
        Formula: E_ads = E_total - E_adsorbent - sum(E_adsorbate_i)
    """
    reference_energies = { "adsorbent": adsorbent }
    for i, energy in enumerate(adsorbate_energies):
        reference_energies[f"adsorbate_{i}"] = energy

    return _calculate_interaction_energy(
        total_energy, 
        reference_energies
    )


def calculate_vacancy_formation_energy(
    defective_energy: float,
    pristine_energy: float,
    defective_site_energy: float,
    num_vacancies: int = 1
) -> float:
    """
    Calculates the vacancy formation energy for a defective system.

    Args:
        defective_energy (float): Energy of the defective system.
        pristine_energy (float): Energy of the pristine system.
        defective_site_energy (float): Chemical potential of the defective site.
        num_vacancies (int, optional): Number of vacancies. Defaults to 1.

    Returns:
        float: Vacancy formation energy (eV).

    Raises:
        StoichiometryError: If the number of vacancies is not a positive integer.
        MissingReferenceError: If a required reference energy for the pristine 
            system or defective site is missing.
        EnergyCalculationError: If any input energy is non-finite.

    Note:
        Formula: E_vac = E_defective - E_pristine + num_vacancies * mu_defective_site
    """
    return _calculate_interaction_energy(
        defective_energy,
        { "pristine_system": pristine_energy, "defective_site": -1 * defective_site_energy },
        { "pristine_system": 1, "defective_site": num_vacancies }
    )