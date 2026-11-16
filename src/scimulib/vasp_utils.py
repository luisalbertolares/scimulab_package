import os
import warnings
from pathlib import Path
from collections import deque
from collections.abc import Callable

import numpy as np

from pymatgen.core import Structure, Lattice


def extract_outcar_property(
        outcar_path: str | Path, 
        target_patterns: str | list[str], 
        property_name: str,
        extract_func: Callable[[str], float] = lambda line: float(line.split()[-1])
) -> float | None:
    """
    Generalized function to extract a scalar property from the end of an OUTCAR file.

    Performs a highly optimized backward chunk-based binary scan. It searches for 
    the last line that contains all the strings specified in `target_patterns`.
    Once found, it applies `extract_func` to parse and return the value.

    Args:
        outcar_path (str | Path): The full path to the OUTCAR file.
        target_patterns (str | list[str]): A single string or a list of strings 
            that must all be present in the target line.
        property_name (str): Human-readable name of the property (used for warnings).
        extract_func (Callable): A function that takes the matched line (str) 
            and returns the parsed value (float). 
            Defaults to extracting the last whitespace-separated element.

    Returns:
        float | None: The extracted property value, or None if the pattern 
        is missing, the file is empty, or an error occurs.

    Raises:
        ValueError: If the target pattern(s) are not found before reaching the beginning 
            of the file, or if the file is completely empty.
        FileNotFoundError: If the specified OUTCAR file does not exist at 'outcar_path'.
        OSError: If a system-related error occurs while attempting to read the file chunks.
        IndexError: If the line splitting inside 'extract_func' fails (e.g., empty string).
        TypeError: If an invalid type is processed during the extraction or decoding steps.
        
        Note: These exceptions are caught internally by the function, converted into 
        RuntimeWarning or UserWarning messages, and cause the function to return None.
    """
    # Normalize target_patterns to a list for unified processing
    if isinstance(target_patterns, str):
        target_patterns = [target_patterns]
        
    try:
        chunk_size = 8192  # 8 KB chunks
        
        with open(outcar_path, 'rb') as f:
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

        # Reached EOF without finding the pattern (or file is empty)
        raise ValueError(f"Pattern(s) {target_patterns} not found. File may be empty or lacking this property.")

    except FileNotFoundError:
        warnings.warn(
            f"OUTCAR file was not found at {outcar_path}.",
            category=UserWarning,
            stacklevel=2
        )
        return None

    except (OSError, ValueError, IndexError, TypeError) as e:
        warnings.warn(
            f"Failed to extract {property_name} from OUTCAR: {e}",
            category=RuntimeWarning,
            stacklevel=2
        )
        return None


def extract_outcar_structure(outcar_path: str | Path) -> Structure | None:
    """
    Generalized function to extract the final pymatgen Structure from an OUTCAR file.

    This function employs a two-phase reading strategy to maximize performance and 
    minimize memory usage for large OUTCAR files:
    1. A fast forward scan restricted to the file header to extract elemental species 
       ('VRHFIN') and the number of atoms per species ('ions per type').
    2. A highly optimized backward chunk-based binary scan. By utilizing a double-ended 
       queue ('deque'), lines read backwards are naturally reconstructed into their 
       original forward order. Once the target headers for lattice vectors or atomic 
       positions are found, the corresponding data block is instantly parsed.

    Args:
        outcar_path (str | Path): The full path to the OUTCAR file.

    Returns:
        Structure | None: A pymatgen Structure object representing the final ionic 
        step of the calculation, or None if the file is missing, empty, or parsing fails.

    Raises:
        ImportError: If the 'pymatgen' library is not installed in the environment.
        ValueError: If essential structural patterns (species, lattice, or coords) 
            are missing, suggesting a corrupted, incomplete, or empty file.
        FileNotFoundError: If the OUTCAR file does not exist at 'outcar_path'.
        OSError: If a system-related error occurs during chunk reading.
        
        Note: File-parsing exceptions are caught internally, converted into 
        RuntimeWarning or UserWarning messages, and cause the function to return None.
    """
    if Structure is None or Lattice is None:
        raise ImportError("pymatgen is required to create a Structure. " \
        "Install it with 'pip install pymatgen'.")
        
    try:
        # 1. Fast forward scan (Species and Ions)
        species_names = []
        num_ions = []
        
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
                    
        if not species_names or not num_ions:
            raise ValueError("Could not find 'VRHFIN' or 'ions per type' at the beginning of the file.")
            
        # Flatten the species list (e.g., ['Si', 'O'] & [2, 4] -> ['Si', 'Si', 'O', 'O', 'O', 'O'])
        species = []
        for name, count in zip(species_names, num_ions):
            species.extend([name] * count)
            
        total_atoms = sum(num_ions)
        
        # 2. Backward chunk-based binary scan (Coordinates and Lattice)
        chunk_size = 16384  # 16 KB (large enough to safely encompass the entire atom block)
        lattice_matrix = None
        cart_coords = None
        
        # deque(appendleft) rebuilds the original forward order while reading backwards.
        # Its maximum size is dynamically adjusted based on the total number of atoms.
        history = deque(maxlen=max(total_atoms + 50, 1000))
        
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
                    buffer = lines[0]  # Retain the incomplete boundary line for the next chunk
                    
                    # Process completed lines bottom-up
                    for line_bytes in reversed(lines[1:]):
                        decoded_line = line_bytes.decode('utf-8', errors='ignore')
                        history.appendleft(decoded_line)
                        
                        # A) Extract final Lattice
                        if "direct lattice vectors" in decoded_line and lattice_matrix is None:
                            try:
                                # history[0] is the header
                                # history[1], history[2], history[3] are the vectors
                                lat_matrix = []
                                for i in range(1, 4):
                                    parts = history[i].split()
                                    lat_matrix.append([float(parts[0]), float(parts[1]), float(parts[2])])
                                lattice_matrix = lat_matrix
                            except (IndexError, ValueError):
                                pass  # Keep scanning if the block was cut and isn't fully in history yet
                                
                        # B) Extract final Coordinates
                        elif "POSITION" in decoded_line and "TOTAL-FORCE" in decoded_line and cart_coords is None:
                            try:
                                # history[0] is the header
                                # history[1] is the "------" separator
                                # history[2] to history[2 + total_atoms - 1] are atomic coordinates
                                coords = []
                                for i in range(2, 2 + total_atoms):
                                    parts = history[i].split()
                                    coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                                cart_coords = coords
                            except (IndexError, ValueError):
                                pass 
                                
                        # Early exit: Stop reading the millisecond we have both data blocks
                        if lattice_matrix is not None and cart_coords is not None:
                            break
                            
                if lattice_matrix is not None and cart_coords is not None:
                    break
                    
        # --- 3. Validation and Structure creation ---
        if lattice_matrix is None or cart_coords is None:
            raise ValueError("Failed to extract lattice or coordinates in the backward scan.")
            
        return Structure(
            lattice=Lattice(lattice_matrix),
            species=species,
            coords=cart_coords,
            coords_are_cartesian=True
        )
        
    except FileNotFoundError:
        warnings.warn(
            f"OUTCAR file was not found at {outcar_path}.",
            category=UserWarning,
            stacklevel=2
        )
        return None
        
    except (ValueError, OSError, IndexError, TypeError) as e:
        warnings.warn(
            f"Failed to extract Structure from OUTCAR: {e}",
            category=RuntimeWarning,
            stacklevel=2
        )
        return None