def cohesive_energy(total_energy: float, atomic_energies: dict[str, float], atom_counts: dict[str, int]) -> float:
    """
    Calculates the cohesive energy per atom of a system.

    This function evaluates stability based on the difference between the sum 
    of the isolated atoms' energies and the optimized system's total energy, 
    normalized by the total number of particles.

    Args:
        total_energy (float): The total energy of the system in eV.
        atomic_energies (dict[str, float]): Reference energies for isolated atoms in eV.
            Example: {'Au': -0.12, 'Ti': -7.85}.
        atom_counts (dict[str, int]): Stoichiometry of the current system.
            Example: {'Au': 12, 'Ti': 48, 'O': 95}.

    Returns:
        float: The cohesive energy per atom in eV/atom.

    Raises:
        KeyError: If there are missing reference energies for any element in the system.
        ValueError: If the system contains no valid atoms (zero particles).
    """
    
    # 1. Validate business logic (check for missing references)
    missing_refs = set(atom_counts.keys()) - set(atomic_energies.keys())
    if missing_refs:
        raise KeyError(f"Missing reference energies for: {missing_refs}")

    # 2. Filter valid quantities and count total particles
    valid_counts = {elem: count for elem, count in atom_counts.items() if count > 0}
    total_particles = sum(valid_counts.values())
    
    if total_particles == 0:
        raise ValueError("The system has no valid atoms. Cannot calculate cohesive energy.")

    # 3. Direct mathematical calculation
    isolated_energy_total = sum(valid_counts[elem] * atomic_energies[elem] for elem in valid_counts)
    total_cohesive_energy = isolated_energy_total - total_energy
    
    return total_cohesive_energy / total_particles