import os
import warnings
from pathlib import Path
from collections import deque
from collections.abc import Callable

from pymatgen.core import Structure, Lattice




def extract_outcar_property(
        outcar_path: str | Path, 
        target_patterns: str | list[str], 
        property_name: str,
        extract_func: Callable[[str], float] = lambda line: float(line.split()[-1])
) -> float:
    """
    Generalized function to extract a scalar property from the end of an OUTCAR file.

    Performs a highly optimized backward chunk-based binary scan. It searches for 
    the last line that contains all the strings specified in `target_patterns`.
    Once found, it applies `extract_func` to parse and return the value.

    Args:
        outcar_path (str | Path): The full path to the OUTCAR file.
        target_patterns (str | list[str]): A single string or a list of strings 
            that must all be present in the target line.
        property_name (str): Human-readable name of the property (used for error contextualization).
        extract_func (Callable[[str], float]): A function that takes the matched line (str) 
            and returns the parsed value (float). Defaults to extracting the last 
            whitespace-separated element and converting it to float.

    Returns:
        float: The successfully extracted and parsed property value.

    Raises:
        FileNotFoundError: If the specified OUTCAR file does not exist.
        OutcarExtractionError: If a system-related error occurs while attempting to read the file,
            or if the extraction function fails to parse the line (e.g., ValueError, IndexError).
        PropertyNotFoundError: If the target pattern(s) are not found before reaching the 
            beginning of the file, or if the file is completely empty.
    """
    # Normalize target_patterns to a list for unified processing
    if isinstance(target_patterns, str):
        target_patterns = [target_patterns]
        
    chunk_size = 8192  # 8 KB chunks
    
    # 1. Handle file opening and I/O errors explicitly
    try:
        f = open(outcar_path, 'rb')
    except FileNotFoundError as e:
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}.") from e
    except OSError as e:
        raise OutcarExtractionError(f"System error while opening OUTCAR at {outcar_path}.") from e

    # 2. Process the open file
    with f:
        try:
            # Move disk pointer to the end of the file
            f.seek(0, os.SEEK_END)
            position = f.tell()
            buffer = b""

            # Scan backwards from end-of-file
            while position > 0:
                read_size = min(chunk_size, position)
                position -= read_size
                f.seek(position)
                
                chunk = f.read(read_size)
                buffer = chunk + buffer

                # Process lines when a newline byte is found
                if b'\n' in buffer:
                    lines = buffer.split(b'\n')
                    buffer = lines[0]  # Retain incomplete boundary line
                    
                    # Parse completed lines bottom-up
                    for line in reversed(lines[1:]):
                        decoded_line = line.decode('utf-8', errors='ignore')
                        
                        # Check if ALL required patterns exist in this line
                        if all(pattern in decoded_line for pattern in target_patterns):
                            return extract_func(decoded_line)

            # Check residual buffer (the very first line of the file)
            if buffer:
                decoded_line = buffer.decode('utf-8', errors='ignore')
                if all(pattern in decoded_line for pattern in target_patterns):
                    return extract_func(decoded_line)
                    
        except (IndexError, ValueError, TypeError) as e:
            # Catch parsing errors from 'extract_func' or decoding issues and bubble them up
            raise OutcarExtractionError(
                f"Failed to parse the value for '{property_name}' using the extraction function."
            ) from e

    # 3. If the loop completes without returning, the pattern simply isn't there
    raise PropertyNotFoundError(
        f"Pattern(s) {target_patterns} not found for '{property_name}'. "
        "File may be empty, incomplete, or the calculation did not reach this step."
    )


def extract_outcar_structure(outcar_path: str | Path) -> 'Structure':
    """
    Generalized function to extract the final pymatgen Structure from an OUTCAR file.

    This function employs a two-phase reading strategy to maximize performance and 
    minimize memory usage for large OUTCAR files:
    1. A fast forward scan restricted to the file header to extract elemental species 
       ('VRHFIN') and the number of atoms per species ('ions per type').
    2. A highly optimized backward chunk-based binary scan. By utilizing a double-ended 
       queue (`deque`), lines read backwards are naturally reconstructed into their 
       original forward order. Once the target headers for lattice vectors or atomic 
       positions are found, the corresponding data block is instantly parsed.

    Args:
        outcar_path (str | Path): The full path to the OUTCAR file.

    Returns:
        Structure: A pymatgen Structure object representing the final ionic step of the calculation.

    Raises:
        ImportError: If the `pymatgen` library (Structure, Lattice) is not available.
        FileNotFoundError: If the specified OUTCAR file does not exist.
        StructureNotFoundError: If essential structural patterns (species, lattice, or coords) 
            are missing, suggesting a corrupted, incomplete, or empty file.
        OutcarExtractionError: If a system-related error occurs during reading, or if 
            pymatgen fails to instantiate the Structure object.
    """ 
    # Validate pymatgen imports (assuming they might be locally imported or global)
    try:
        if Structure is None or Lattice is None:
            raise ImportError("pymatgen is required to create a Structure object.")
    except NameError as e:
        raise ImportError("pymatgen classes (Structure, Lattice) are not imported.") from e

    outcar_path = Path(outcar_path)
    if not outcar_path.is_file():
        raise FileNotFoundError(f"OUTCAR file not found at {outcar_path}.")

    # 1. Fast forward scan (Species and Ions)
    species_names = []
    num_ions = []
    
    try:
        with open(outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
            for _ in range(5000):  # Safety limit for header search
                line = f.readline()
                if not line:
                    break
                    
                if "VRHFIN" in line:
                    # Format: " VRHFIN =Si: s p d" -> "Si"
                    elem = line.split("=")[1].split(":")[0].strip()
                    species_names.append(elem)
                    
                elif "ions per type" in line:
                    # Format: "   ions per type =               4   4"
                    parts = line.split("=")[1].split()
                    num_ions = [int(x) for x in parts]
                    break  # Found the complete configuration, exit loop
    except OSError as e:
        raise OutcarExtractionError(f"System error reading header of OUTCAR at {outcar_path}.") from e
                
    if not species_names or not num_ions:
        raise StructureNotFoundError(
            "Could not find 'VRHFIN' or 'ions per type' in the header. File may be empty or invalid."
        )
        
    # Flatten the species list (e.g., ['Si', 'O'] & [2, 4] -> ['Si', 'Si', 'O', 'O', 'O', 'O'])
    species = []
    for name, count in zip(species_names, num_ions):
        species.extend([name] * count)
        
    total_atoms = sum(num_ions)
    
    # 2. Backward chunk-based binary scan (Coordinates & Lattice)
    chunk_size = 16384  # 16 KB
    lattice_matrix = None
    cart_coords = None
    
    # deque(appendleft) rebuilds the original forward order while reading backwards.
    history = deque(maxlen=max(total_atoms + 50, 1000))
    
    try:
        with open(outcar_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            position = f.tell()
            buffer = b""
            
            while position > 0:
                read_size = min(chunk_size, position)
                position -= read_size
                f.seek(position)
                
                chunk = f.read(read_size)
                buffer = chunk + buffer
                
                if b'\n' in buffer:
                    lines = buffer.split(b'\n')
                    buffer = lines[0]  # Retain the incomplete boundary line
                    
                    # Process completed lines bottom-up
                    for line_bytes in reversed(lines[1:]):
                        decoded_line = line_bytes.decode('utf-8', errors='ignore')
                        history.appendleft(decoded_line)
                        
                        # A) Extract final Lattice
                        if "direct lattice vectors" in decoded_line and lattice_matrix is None:
                            try:
                                lat_matrix = []
                                for i in range(1, 4):
                                    parts = history[i].split()
                                    lat_matrix.append([float(parts[0]), float(parts[1]), float(parts[2])])
                                lattice_matrix = lat_matrix
                            except (IndexError, ValueError):
                                # It's normal to hit this if the block was cut by the chunk boundary.
                                # The deque will fill up in the next iterations.
                                pass 
                                
                        # B) Extract final Coordinates
                        elif "POSITION" in decoded_line and "TOTAL-FORCE" in decoded_line and cart_coords is None:
                            try:
                                coords = []
                                for i in range(2, 2 + total_atoms):
                                    parts = history[i].split()
                                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                                cart_coords = coords
                            except (IndexError, ValueError):
                                # Same as above: keep scanning until deque contains the full block
                                pass 
                                
                        # Early exit: Stop reading the millisecond we have both data blocks
                        if lattice_matrix is not None and cart_coords is not None:
                            break
                            
                if lattice_matrix is not None and cart_coords is not None:
                    break
    except OSError as e:
        raise OutcarExtractionError(f"System error during backward scan of OUTCAR at {outcar_path}.") from e
                
    # 3. Validation and Structure creation
    if lattice_matrix is None or cart_coords is None:
        raise StructureNotFoundError(
            "Failed to extract lattice vectors or coordinates in the backward scan. "
            "The calculation may have crashed before writing structural data."
        )
        
    try:
        return Structure(
            lattice=Lattice(lattice_matrix),
            species=species,
            coords=cart_coords,
            coords_are_cartesian=True
        )
    except Exception as e:
        # Catch any internal pymatgen errors (e.g., mismatched arrays)
        raise OutcarExtractionError(f"Failed to instantiate pymatgen Structure: {e}") from e