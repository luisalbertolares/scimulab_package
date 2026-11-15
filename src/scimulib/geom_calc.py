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