from __future__ import annotations
from typing import TYPE_CHECKING

import math
import numpy as np

if TYPE_CHECKING:
    from ase import Atoms


def cohesive_energy(total_energy: float, atomic_energies: dict[str, float], atom_counts: dict[str, int]) -> float:
    """
    Calculates the cohesive energy per atom of a system.

    This function evaluates stability based on the difference between the sum 
    of the isolated atoms' energies and the optimized system's total energy, 
    normalized by the total number of particles.

    Args:
        total_energy (float): The total energy of the system.
        atomic_energies (dict[str, float]): Reference energies for isolated atoms.
            Example: {'Au': -0.12, 'Ti': -7.85}.
        atom_counts (dict[str, int]): Stoichiometry of the current system.
            Example: {'Au': 12, 'Ti': 48, 'O': 95}.

    Returns:
        float: The cohesive energy per atom (in the same units as the 
                inputs, typically eV/atom).

    Raises:
        KeyError: If there are missing reference energies for any element in the system.
        ValueError: If the system contains no valid atoms (zero particles).
    """
    
    # 1. Validate that no atom energy references are missing
    missing_refs = set(atom_counts.keys()) - set(atomic_energies.keys())
    if missing_refs:
        raise KeyError(f"Missing reference energies for: {missing_refs}")

    # 2. Filter valid quantities and count total particles
    valid_counts = {elem: count for elem, count in atom_counts.items() if count > 0}
    total_particles = sum(valid_counts.values())
    
    if total_particles == 0:
        raise ValueError("The system has no valid atoms. Cannot calculate cohesive energy.")

    # 3. Direct mathematical calculation
    isolated_total_energy = sum(valid_counts[elem] * atomic_energies[elem] for elem in valid_counts)
    
    return (isolated_total_energy - total_energy) / total_particles


def ideal_cohesive_energy(mono_cohesive_energies: dict[str, float], atomic_fractions: dict[str, float]) -> float:
    """
    Calculates the ideal cohesive energy (without mixing interaction) for a multi-metallic cluster.

    This represents the weighted average of the pure monometallic cohesive energies.

    Args:
        mono_cohesive_energies (dict[str, float]): Cohesive energies of the pure monometallic clusters.
            Example: {'Au': -3.12, 'Co': -4.39, 'Cu': -3.49}.
        atomic_fractions (dict[str, float]): Atomic fractions of each metal in the cluster.
            Example: {'Au': 0.5, 'Co': 0.25, 'Cu': 0.25}.

    Returns:
        float: The ideal cohesive energy (eV/atom).

    Raises:
        KeyError: If there are missing reference energies for any element in the cluster.
        ValueError: If the sum of atomic fractions is not equal to 1.0.
    """
    
    # 1. Validate missing monometallic references
    missing_refs = set(atomic_fractions.keys()) - set(mono_cohesive_energies.keys())
    if missing_refs:
        raise KeyError(f"Missing monometallic cohesive energies for: {missing_refs}")

    # 2. Thermodynamic constraint: Fractions must sum to 1.0
    total_fraction = sum(atomic_fractions.values())
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(
            f"Atomic fractions must sum to 1.0. Current sum: {total_fraction:.4f}. "
            "Ensure you are passing fractions, not total atom counts."
        )

    # 3. Mathematical calculation: sum(x_i * E_coh_i)
    return sum(fraction * mono_cohesive_energies[elem] for elem, fraction in atomic_fractions.items())


def mixing_cohesive_energy(mixing_energy: float, total_atoms: int) -> float:
    """
    Calculates the cohesive energy of mixing for a cluster.

    Args:
        mixing_energy (float): The total mixing energy of the cluster.
        total_atoms (int): The total number of atoms in the cluster.

    Returns:
        float: The cohesive energy of mixing (eV/atom).

    Raises:
        ValueError: If the total number of atoms is zero or negative.
    """
    
    # 1. Physical constraint validation
    if total_atoms <= 0:
        raise ValueError(f"The total number of atoms must be greater than zero. Current value: {total_atoms}")

    # 2. Mathematical calculation
    # Python will naturally raise a TypeError if the inputs are strings or incompatible types
    return -(mixing_energy / total_atoms)


def _calculate_relative_energy(cluster_energy: float, reference_energies: dict[str, float], atomic_fractions: dict[str, float]) -> float:
    """
    Internal function to calculate relative energies based on a weighted reference.
    
    Performs the operation: E_cluster - sum(fraction_i * E_reference_i).
    """
    missing_refs = set(atomic_fractions.keys()) - set(reference_energies.keys())
    if missing_refs:
        raise KeyError(f"Missing reference energies for: {missing_refs}")

    total_fraction = sum(atomic_fractions.values())
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(
            f"Atomic fractions must sum to 1.0. Current sum: {total_fraction:.4f}. "
            "Ensure you are passing fractions, not total atom counts."
        )

    weighted_energy = sum(fraction * reference_energies[elem] for elem, fraction in atomic_fractions.items())
    
    return cluster_energy - weighted_energy


def mixing_energy(alloy_energy: float, relaxed_energies: dict[str, float], atomic_fractions: dict[str, float]) -> float:
    """
    Calculates the total mixing energy for a multi-metallic cluster.

    The mixing energy measures the relative stability of the alloy against
    segregation into its pure monometallic components of equal size.

    Args:
        alloy_energy (float): Total energy of the relaxed mixed cluster (alloy).
        relaxed_energies (dict[str, float]): Total energies of the fully relaxed 
            pure monometallic clusters. Example: {'Au': -150.2, 'Co': -180.5}.
        atomic_fractions (dict[str, float]): Atomic fractions of each metal in the cluster.

    Returns:
        float: The mixing energy in the same units as the inputs (typically eV).

    Note:
        Formula: E_mix = E_alloy - sum(x_i * E_relaxed_i)
        A negative value indicates that the formation of the alloy is 
        thermodynamically favorable compared to the segregated pure metals.
    """
    return _calculate_relative_energy(alloy_energy, relaxed_energies, atomic_fractions)


def interact_mixing_energy(alloy_energy: float, frozen_energies: dict[str, float], atomic_fractions: dict[str, float]) -> float:
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

    Note:
        Formula: E_interact = E_alloy - sum(x_i * E_frozen_i)
    """
    return _calculate_relative_energy(alloy_energy, frozen_energies, atomic_fractions)


def strain_mixing_energy(frozen_energies: dict[str, float], relaxed_energies: dict[str, float], atomic_fractions: dict[str, float]) -> float:
    """
    Calculates the strain contribution to the mixing energy in a multi-metallic cluster.

    This contribution reflects how much of the mixing energy is consumed or released 
    by the structural deformation required to accommodate different atomic sizes 
    in the alloy lattice.

    Args:
        frozen_energies (dict[str, float]): Energies of the monometallic clusters 
            with frozen geometries (matching the alloy's structural lattice).
        relaxed_energies (dict[str, float]): Energies of the fully relaxed pure 
            monometallic clusters.
        atomic_fractions (dict[str, float]): Atomic fractions of each metal in the cluster.

    Returns:
        float: Strain contribution to the mixing energy (eV).

    Raises:
        KeyError: If reference energies are missing in either dictionary.
        ValueError: If atomic fractions do not sum to 1.0.

    Note:
        Formula: E_strain = sum(x_i * (E_frozen_i - E_relaxed_i))
        By definition, E_mix = E_interact + E_strain.
    """
    
    # Validate missing references across both dictionaries
    elements = set(atomic_fractions.keys())
    if not elements.issubset(frozen_energies.keys()) or not elements.issubset(relaxed_energies.keys()):
        raise KeyError("Missing frozen or relaxed reference energies for one or more elements.")

    # Thermodynamic constraint
    total_fraction = sum(atomic_fractions.values())
    if not math.isclose(total_fraction, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(f"Atomic fractions must sum to 1.0. Current sum: {total_fraction:.4f}")

    # Calculate strain: sum(x_i * (E_frozen_i - E_relaxed_i))
    strain_energy = sum(
        fraction * (frozen_energies[elem] - relaxed_energies[elem]) 
        for elem, fraction in atomic_fractions.items()
    )
    
    return strain_energy


def calculate_pdf(atoms: Atoms, r_final: float = 5.0, dr: float = 0.05, sigma: float = 0.05) -> np.ndarray:
    """
    Calculates a pseudo-continuous Pair Distribution Function (PDF).

    Generates a profile of interatomic distances broadened by Gaussian functions.
    The calculation is fully vectorized using NumPy broadcasting for high performance,
    making it suitable for analyzing large trajectories or massive cluster batches.

    Args:
        atoms (Atoms): ASE Atoms object containing the structure to analyze.
        r_final (float, optional): Maximum distance in Ångströms for the calculation. 
            Defaults to 5.0.
        dr (float, optional): Resolution of the distance mesh (grid step size). 
            Defaults to 0.05.
        sigma (float, optional): Standard deviation of the Gaussian broadening. 
            Controls the "smoothness" of the profile. Defaults to 0.05.

    Returns:
        np.ndarray: A 1D array containing the PDF intensity values corresponding 
            to the distance grid defined by dr and r_final.

    Raises:
        ValueError: If physical parameters (dr, sigma, r_final) are not strictly positive,
            or if the system has fewer than 2 atoms.
        AttributeError: If the 'atoms' object lacks the 'get_all_distances' method natively.

    Note:
        The PDF is computed over unique distances only (upper triangle of the 
        distance matrix) to avoid duplicate counting and self-distances (d=0).
    """

    # 1. Physical and mathematical validation
    if len(atoms) < 2:
        raise ValueError(f"At least 2 atoms are required to calculate a PDF. Found: {len(atoms)}")

    if dr <= 0 or sigma <= 0 or r_final <= 0:
        raise ValueError("Parameters 'dr', 'sigma', and 'r_final' must be strictly positive.")

    # 2. Distance extraction
    # Python will naturally raise an AttributeError if 'atoms' isn't a valid ASE object
    distances = atoms.get_all_distances()

    # Extract only unique distances (upper triangle, k=1 excludes the d=0 diagonal)
    num_atoms = len(atoms)
    unique_distances = distances[np.triu_indices(num_atoms, k=1)]

    # 3. Vectorized PDF Calculation (Broadcasting)
    # r_grid shape: (N,) -> reshaped to (N, 1) to form a column vector
    r_grid = np.arange(0, r_final, dr)
    
    # diff shape becomes (N_grid_points, N_unique_distances) matrix
    diff_matrix = r_grid[:, np.newaxis] - unique_distances
    
    # Calculate all Gaussians simultaneously in C-optimized memory
    gaussians = np.exp(-0.5 * (diff_matrix / sigma)**2)
    
    # Sum across the distance axis (axis=1) and apply the normalization constant
    normalization = sigma * np.sqrt(2 * np.pi)
    pdf = np.sum(gaussians, axis=1) / normalization

    return pdf