import os
import warnings
from pathlib import Path
from typing import Union, List, Callable, Optional

import numpy as np

def extract_outcar_property(
        outcar_path: str | Path, 
        target_patterns: Union[str, List[str]], 
        property_name: str,
        extract_func: Callable[[str], float] = lambda line: float(line.split()[-1])
) -> Optional[float]:
    """
    Generalized function to extract a scalar property from the end of an OUTCAR file.

    Performs a highly optimized backward chunk-based binary scan. It searches for 
    the last line that contains all the strings specified in `target_patterns`.
    Once found, it applies `extract_func` to parse and return the value.

    Args:
        outcar_path (str | Path): The full path to the OUTCAR file (or directory if you append it).
        target_patterns (str | List[str]): A single string or a list of strings 
            that must all be present in the target line.
        property_name (str): Human-readable name of the property (used for warnings).
        extract_func (Callable): A function that takes the matched line (str) 
            and returns the parsed value (float). 
            Defaults to extracting the last whitespace-separated element.

    Returns:
        Optional[float]: The extracted property value, or None if the pattern 
        is missing, the file is empty, or an error occurs.
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
            f"OUTCAR file was not found at {outcar_path}. Failed to extract {property_name}.",
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

def parse_acf_file(
        acf_path: str | Path,
        column_index: int = 4,
        expected_atoms: Optional[int] = None
) -> Optional[np.ndarray]:
    """
    Parses a Bader ACF file to extract a specific column of atomic properties 
    (e.g., charge, volume, or magnetic moment) as a NumPy array.

    Bader ACF files typically contain 2 header lines and 4 footer lines. 
    The atomic data is located between these blocks.

    Args:
        acf_path (str | Path): The full path to the ACF file (e.g., 'ACF.dat').
        column_index (int): The zero-based index of the column to extract. 
            Defaults to 4 (corresponds to CHARGE in ACF.dat).
        expected_atoms (int, optional): The expected number of atoms.

    Returns:
        Optional[np.ndarray]: A 1D NumPy array of extracted values for each atom. 
        Returns None if the file is missing, malformed, or an error occurs.
    """
    try:
        with open(acf_path, 'r') as f:
            lines = f.readlines()
        
        if len(lines) <= 6:
            raise ValueError("File is too short to be a valid Bader ACF file.")
            
        extracted_values = []
        
        for i, line in enumerate(lines[2:-4], start=3):
            parts = line.split()
            if len(parts) > column_index:
                extracted_values.append(float(parts[column_index]))
            else:
                raise IndexError(
                    f"Line {i} lacks sufficient columns (expected at least {column_index + 1})."
                )
        
        if expected_atoms is not None and len(extracted_values) != expected_atoms:
            warnings.warn(
                f"Atom count mismatch in {acf_path}: Expected {expected_atoms}, "
                f"but found {len(extracted_values)}.",
                category=UserWarning,
                stacklevel=2
            )
            return None
            
        return np.array(extracted_values, dtype=float)
        
    except FileNotFoundError:
        warnings.warn(
            f"ACF file was not found at {acf_path}.",
            category=UserWarning,
            stacklevel=2
        )
        return None
        
    except (ValueError, IndexError, TypeError) as e:
        warnings.warn(
            f"Failed to parse ACF file at {acf_path}: {e}",
            category=RuntimeWarning,
            stacklevel=2
        )
        return None