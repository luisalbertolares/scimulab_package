import os
import logging
from pathlib import Path
from collections import deque
from collections.abc import Callable

import numpy as np

from pymatgen.io.vasp.outputs import Vasprun, Oszicar
from pymatgen.core import Structure, Lattice

from scimulib.exceptions import VasprunError, OszicarError, OutcarError, \
    PropertyNotFoundError, StructureNotFoundError


def _check_pymatgen_imports():
    """
    Checks if the required pymatgen classes are available. Raises ImportError if not.
    """
    try:
        from pymatgen.io.vasp.outputs import Vasprun, Oszicar
        from pymatgen.core import Structure, Lattice
    except ImportError as e:
        raise ImportError(
            "The 'pymatgen' library is required for VASP parsing. "
            "Please install it via 'pip install pymatgen'."
        ) from e

class VaspParser:
    """
    A class to parse the results of a VASP calculation. It provides methods to 
    extract key properties such as total energy, net magnetic moment and geometry
    from the VASP output files.
    """

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)

        _check_pymatgen_imports()
        if not self._directory.is_dir():
            raise NotADirectoryError(
                f"The specified directory '{self._directory}' does not exist or is invalid."
            )
        
        self._vasprun_path: Path = self._directory / "vasprun.xml"
        self._poscar_path: Path = self._directory / "POSCAR"
        self._contcar_path: Path = self._directory / "CONTCAR"
        self._outcar_path: Path = self._directory / "OUTCAR"
        self._oszicar_path: Path = self._directory / "OSZICAR"
        self._doscar_path: Path = self._directory / "DOSCAR"

        self._vasprun: Vasprun | None = None
        self._vasprun_params: dict = None
        self._oszcicar: Oszicar | None = None

    def load_vasprun(self, ionic_step_offset=-1, ionic_step_skip=None, occu_tol=1e-8, 
                     parse_dos=False, parse_eigen=False, parse_projected_eigen=False, 
                     parse_potcar_file=False, exception_on_bad_xml=True):
        """
        Loads the vasprun.xml file using Pymatgen's Vasprun class. 
        Re-parses the file if the requested parameters change.

        Args:
            ionic_step_offset (int): Offset to read only a subset of ionic steps.
            ionic_step_skip (int | None): Number of ionic steps to skip.
            occu_tol (float): Tolerance for determining if occupancies are fractional.
            parse_dos (bool): Whether to parse the density of states.
            parse_eigen (bool): Whether to parse eigenvalues.
            parse_projected_eigen (bool): Whether to parse projected eigenvalues.
            parse_potcar_file (bool): Whether to parse the POTCAR file to obtain elements and masses.
            exception_on_bad_xml (bool): Whether to throw an exception if the XML is malformed.

        Returns:
            Vasprun: An instance of the Vasprun class containing parsed data.

        Raises:
            FileNotFoundError: If the vasprun.xml file does not exist in the specified directory.
            VasprunError: If the vasprun.xml file cannot be parsed due to internal Pymatgen 
                errors or corruption.
        """
        if not self._vasprun_path.is_file():
            raise FileNotFoundError(f"vasprun.xml file not found at {self._vasprun_path}.")

        current_params = {
            "ionic_step_offset": ionic_step_offset,
            "ionic_step_skip": ionic_step_skip,
            "occu_tol": occu_tol,
            "parse_dos": parse_dos,
            "parse_eigen": parse_eigen,
            "parse_projected_eigen": parse_projected_eigen,
            "parse_potcar_file": parse_potcar_file,
            "exception_on_bad_xml": exception_on_bad_xml
        }

        if self._vasprun is None or self._vasprun_params != current_params:
            try:
                self._vasprun = Vasprun(self._vasprun_path, **current_params)
                self._vasprun_params = current_params
            except Exception as e:
                raise VasprunError(
                    f"Internal pymatgen failure in Vasprun: {type(e).__name__} - {e}"
                ) from e

        return self._vasprun

    def load_oszicar(self):
        """
        Loads the OSZICAR file using Pymatgen's Oszicar class. 

        Returns:
            Oszicar: An instance of the Oszicar class containing parsed data.

        Raises:
            FileNotFoundError: If the OSZICAR file does not exist in the specified directory.

            OszicarError: If the OSZICAR file cannot be parsed due to internal Pymatgen 
                errors or corruption.
        """
        if not self._oszicar_path.is_file():
            raise FileNotFoundError(f"OSZICAR file not found at {self._oszicar_path}.")

        if self._oszcicar is None:
            try:
                self._oszcicar = Oszicar(self._oszicar_path)
            except Exception as e:
                raise OszicarError(
                    f"Internal pymatgen failure in Oszicar: {type(e).__name__} - {e}"
                ) from e

        return self._oszcicar

    def extract_outcar_property(
            self, 
            target_patterns: str | list[str], 
            property_name: str,
            extract_func: Callable[[str], float] = lambda line: float(line.split()[-1])
    ) -> float:
        """
        Generalized method to extract a scalar property from the end of an OUTCAR file.

        Performs a highly optimized backward chunk-based binary scan. It searches for 
        the last line that contains all the strings specified in `target_patterns`.
        Once found, it applies `extract_func` to parse and return the value.

        Args:
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
            OutcarError: If a system-related error occurs while attempting to read the file,
                or if the extraction function fails to parse the line (e.g., ValueError, IndexError).
            PropertyNotFoundError: If the target pattern(s) are not found before reaching the 
                beginning of the file, or if the file is completely empty.
        """
        if not self._outcar_path.is_file():
            raise FileNotFoundError(f"OUTCAR file not found at {self._outcar_path}.")

        if isinstance(target_patterns, str):
            target_patterns = [target_patterns]
            
        chunk_size = 8192
        
        try:
            with open(self._outcar_path, 'rb') as f:
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
                        buffer = lines[0]
                        
                        for line in reversed(lines[1:]):
                            decoded_line = line.decode('utf-8', errors='ignore')
                            
                            if all(pattern in decoded_line for pattern in target_patterns):
                                try:
                                    return extract_func(decoded_line)
                                except (IndexError, ValueError, TypeError) as e:
                                    raise OutcarError(
                                        f"Failed to parse the value for '{property_name}' using the extraction function."
                                    ) from e

                if buffer:
                    decoded_line = buffer.decode('utf-8', errors='ignore')
                    if all(pattern in decoded_line for pattern in target_patterns):
                        try:
                            return extract_func(decoded_line)
                        except (IndexError, ValueError, TypeError) as e:
                            raise OutcarError(
                                f"Failed to parse the value for '{property_name}' using the extraction function."
                            ) from e
                            
        except OSError as e:
            raise OutcarError(f"System error while reading OUTCAR at {self._outcar_path}.") from e

        raise PropertyNotFoundError(
            f"Pattern(s) {target_patterns} not found for '{property_name}'. "
            "File may be empty, incomplete, or the calculation did not reach this step."
        )

    def extract_outcar_structure(self) -> 'Structure':
        """
        Generalized method to extract the final pymatgen Structure from an OUTCAR file.

        This method employs a two-phase reading strategy to maximize performance and 
        minimize memory usage for large OUTCAR files:
        1. A fast forward scan restricted to the file header to extract elemental species 
           ('VRHFIN') and the number of atoms per species ('ions per type').
        2. A highly optimized backward chunk-based binary scan. By utilizing a double-ended 
           queue (`deque`), lines read backwards are naturally reconstructed into their 
           original forward order. Once the target headers for lattice vectors or atomic 
           positions are found, the corresponding data block is instantly parsed.

        Returns:
            Structure: A pymatgen Structure object representing the final ionic step of the calculation.

        Raises:
            FileNotFoundError: If the specified OUTCAR file does not exist.
            OutcarError: If a system-related error occurs during reading, or if 
                pymatgen fails to instantiate the Structure object.
            StructureNotFoundError: If essential structural patterns (species, lattice, or coords) 
                are missing, suggesting a corrupted, incomplete, or empty file.
        """
        if not self._outcar_path.is_file():
            raise FileNotFoundError(f"OUTCAR file not found at {self._outcar_path}.")

        species_names = []
        num_ions = []
        
        try:
            with open(self._outcar_path, 'r', encoding='utf-8', errors='ignore') as f:
                for _ in range(5000):
                    line = f.readline()
                    if not line:
                        break
                        
                    if "VRHFIN" in line:
                        elem = line.split("=")[1].split(":")[0].strip()
                        species_names.append(elem)
                        
                    elif "ions per type" in line:
                        parts = line.split("=")[1].split()
                        num_ions = [int(x) for x in parts]
                        break
        except OSError as e:
            raise OutcarError(f"System error reading header of OUTCAR at {self._outcar_path}.") from e
                    
        if not species_names or not num_ions:
            raise StructureNotFoundError(
                "Could not find 'VRHFIN' or 'ions per type' in the header. File may be empty or invalid."
            )
            
        species = []
        for name, count in zip(species_names, num_ions):
            species.extend([name] * count)
            
        total_atoms = sum(num_ions)
        chunk_size = 16384
        lattice_matrix = None
        cart_coords = None
        history = deque(maxlen=max(total_atoms + 50, 1000))
        
        try:
            with open(self._outcar_path, 'rb') as f:
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
                        buffer = lines[0]
                        
                        for line_bytes in reversed(lines[1:]):
                            decoded_line = line_bytes.decode('utf-8', errors='ignore')
                            history.appendleft(decoded_line)
                            
                            if "direct lattice vectors" in decoded_line and lattice_matrix is None:
                                try:
                                    lat_matrix = []
                                    for i in range(1, 4):
                                        parts = history[i].split()
                                        lat_matrix.append([float(parts[0]), float(parts[1]), float(parts[2])])
                                    lattice_matrix = lat_matrix
                                except (IndexError, ValueError):
                                    pass 
                                    
                            elif "POSITION" in decoded_line and "TOTAL-FORCE" in decoded_line and cart_coords is None:
                                try:
                                    coords = []
                                    for i in range(2, 2 + total_atoms):
                                        parts = history[i].split()
                                        coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
                                    cart_coords = coords
                                except (IndexError, ValueError):
                                    pass 
                                    
                            if lattice_matrix is not None and cart_coords is not None:
                                break
                                
                    if lattice_matrix is not None and cart_coords is not None:
                        break
        except OSError as e:
            raise OutcarError(f"System error during backward scan of OUTCAR at {self._outcar_path}.") from e
                    
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
            raise OutcarError(f"Failed to instantiate pymatgen Structure: {e}") from e
        
    def get_total_energy(self) -> float | None:
        """
        Extracts the total energy from the final ionic step of a VASP calculation.

        Attempts to recover the total energy (in eV) across three sequential stages:
            1. vasprun.xml: Parses the structured XML file via Pymatgen.
            2. OSZICAR: Extracts the 'E0' key from the last recorded ionic step.
            3. OUTCAR: Scans backward from the end of the file in binary mode to find 
               'energy(sigma->0)', avoiding high RAM usage on giant files.

        Returns:
            float | None: The total energy if successfully parsed, None otherwise.
        """
        try:
            return float(self.load_vasprun().final_energy)
        except (FileNotFoundError, VasprunError, AttributeError, TypeError) as e:
            logging.warning(f"vasprun.xml failed to provide total energy ({type(e).__name__}). Trying OSZICAR.")

        try:
            return float(self.load_oszicar().ionic_steps[-1]['E0'])
        except (FileNotFoundError, OszicarError, AttributeError, IndexError, KeyError, TypeError) as e:
            logging.warning(f"OSZICAR failed to provide total energy ({type(e).__name__}). Trying OUTCAR.")

        try:
            return self.extract_outcar_property(
                target_patterns="energy(sigma->0)",
                property_name="total energy"
            )
        except (FileNotFoundError, OutcarError, PropertyNotFoundError) as e:
            logging.warning(f"OUTCAR failed to provide total energy ({type(e).__name__}).")

        logging.error(f"Unable to extract total energy in: {self._directory}")
        return None

    def get_net_magmom(self) -> float | None:
        """
        Extracts the final net magnetic moment from a VASP calculation directory.

        Attempts to recover the net magnetic moment (in Bohr magnetons) across two sequential stages:
            1. OSZICAR: Parses the file via Pymatgen and extracts the 'mag' key from 
               the last recorded ionic step.
            2. OUTCAR: Scans backward from the end of the file in binary mode to find 
               the line containing 'magnetization' and 'number of electron', avoiding 
               high RAM usage on large files.

        Note:
            If the calculation is non-spin-polarized (ISPIN = 1), the magnetic moment 
            is inherently zero.

        Returns:
            float | None: The net magnetic moment if successfully parsed, None otherwise.
        """
        try:
            return float(self.load_oszicar().ionic_steps[-1]['mag'])
        except (FileNotFoundError, OszicarError, AttributeError, IndexError, KeyError, TypeError) as e:
            logging.warning(f"OSZICAR failed to provide net magnetic moment ({type(e).__name__}). Trying OUTCAR.")

        try:
            return self.extract_outcar_property(
                target_patterns=["magnetization", "number of electron"],
                property_name="net magnetic moment"
            )
        except (FileNotFoundError, OutcarError, PropertyNotFoundError) as e:
            logging.warning(f"OUTCAR failed to provide net magnetic moment ({type(e).__name__}).")

        logging.error(f"Unable to extract net magnetic moment in: {self._directory}")
        return None

    def get_structure(self) -> Structure | None:
        """
        Creates a Pymatgen Structure object from the VASP calculation directory.

        First, attempts to recover the geometry as a Pymatgen Structure object across three 
        sequential stages:
            1. vasprun.xml: Parses the structured XML file via Pymatgen and converts the 
               last recorded structure into a Pymatgen Structure object.
            2. POSCAR: Falls back to reading the geometry file using Pymatgen.
            3. CONTCAR: Falls back to reading the geometry file using Pymatgen.
            4. OUTCAR: As a last resort, attempts to extract the geometry from the OUTCAR 
               file using a custom parser.

        Returns:
            Structure | None: A Pymatgen Structure object if successfully parsed, None otherwise.
        """
        try:
            return self.load_vasprun().structures[-1]
        except (FileNotFoundError, VasprunError, AttributeError, IndexError) as e:
            logging.warning(
                f"vasprun.xml failed to provide geometry ({type(e).__name__}). Trying POSCAR/CONTCAR."
            )
            
        try:
            return Structure.from_file(self._poscar_path)
        except Exception:
            pass
        try:
            return Structure.from_file(self._contcar_path)
        except (FileNotFoundError, ValueError, OSError) as e:
            logging.warning(
                f"POSCAR/CONTCAR failed to provide geometry ({type(e).__name__}). Trying OUTCAR."
            )

        try:
            return self.extract_outcar_structure()
        except (FileNotFoundError, OutcarError, StructureNotFoundError) as e:
            logging.warning(f"OUTCAR failed to provide geometry ({type(e).__name__}).")

        logging.error(f"Unable to extract geometry in: {self._directory}")
        return None


class BaderParser:
    """
    A class to parse Bader charge and magnetic moment analysis results from a VASP 
    calculation directory. It provides methods to extract atomic charges and magnetic 
    moments from the corresponding ACF files generated by the Bader analysis.
    """

    FILE_MAP = {
        "charge": ["ACF.dat", "ACF_charge.dat"],
        "magmom": ["ACF_mag.dat"]
    }

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)

        if not self._directory.is_dir():
            raise FileNotFoundError(
                f"The specified directory '{self._directory}' does not exist or is invalid."
            )

    @staticmethod
    def parse_acf_file(acf_path: str | Path, 
                        column_index: int = 4, 
                        expected_atoms: int | None = None
    ) -> np.ndarray | None:
        """
        Parses a Bader ACF file to extract a specific column of atomic properties 
        (e.g., charge, volume, or magnetic moment) as a NumPy array.

        Bader ACF files typically contain 2 header lines and 4 footer lines. 
        The atomic data is located between these blocks.

        Args:
            acf_path (str | Path): The full path to the ACF file (e.g., 'ACF.dat').
            column_index (int): The zero-based index of the column to extract. 
                Defaults to 4 (corresponds to CHARGE in ACF.dat).
            expected_atoms (int | None, optional): The expected number of atoms.

        Returns:
            np.ndarray | None: A 1D NumPy array of extracted values for each atom. 
            Returns None if the file is missing, malformed, or an error occurs.

        Raises:
            ValueError: If the file has 6 or fewer lines, rendering it too short to be a 
                valid Bader ACF file, or if the string cannot be cast to a float.
            IndexError: If any parsed data line lacks sufficient columns based on 'column_index'.
            FileNotFoundError: If the ACF file cannot be found at the specified path.
            TypeError: If an unexpected type is encountered during the iteration or casting process.
            
            Note: These exceptions are caught internally by the function, converted into 
            RuntimeWarning or UserWarning messages, and cause the function to return None.
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

    def get_bader_property(self, property: str) -> np.ndarray | None:
        """
        Extracts a specific Bader analysis property by locating and parsing the corresponding ACF file.

        This method checks the target directory for files associated with the 
        requested property (as defined in 'FILE_MAP'). If a matching file is found, 
        it parses and returns the data. If no valid file exists, it issues a warning 
        and returns 'None'.

        Args:
            property (str): The Bader property to extract. Must be a valid 
                key defined in 'FILE_MAP' (e.g., 'charge', 'magmom').

        Returns:
            np.ndarray | None: A NumPy array containing the parsed property values, 
            or 'None' if the corresponding ACF file is not found in the directory.

        Raises:
            ValueError: If the requested 'property' is not a recognized key in 'FILE_MAP'.
        """
        # Validate the requested property against the expetected ones in FILE_MAP
        if property not in self.FILE_MAP:
            warnings.warn(
                f"Invalid property '{property}' requested. "
                f"Expected one of: {list(self.FILE_MAP.keys())}.",
                category=UserWarning,
                stacklevel=2
            )

        # List of potential ACF files for the requested property
        possible_acf_files = self.FILE_MAP[property]

        # Attempt to locate the first existing ACF file in the directory
        acf_path = next(
        (self.directory / fname for fname in possible_acf_files if (self.directory / fname).is_file()), 
        None
        )

        if acf_path is None:
            warnings.warn(
                f"No {possible_acf_files} found for property '{property}' in directory '{self.directory}'. "
                "No Bader property could be extracted.",
                category=UserWarning,
                stacklevel=2
            )
            return None

        return self.parse_acf_file(acf_path=acf_path)

    def get_charges(self) -> np.ndarray:
        """
        Wraps the get_bader_property method to extract the Bader charges specifically. 

        Returns:
            np.ndarray: A 1D NumPy array containing the Bader charge for each atom.
            None: If the ACF charge file is missing, empty, or corrupted.

        Raises:
            UserWarning/RuntimeWarning: Issued by the underlying parser if the file 
                                        is missing or malformed.
        """
        return self.get_bader_property(property="charge")

    def get_magmoms(self) -> np.ndarray:
        """
        Wraps the get_bader_property method to extract the Bader magnetic moments 
        specifically. 

        Returns:
            np.ndarray: A 1D NumPy array containing the magnetic moment for each atom.
            None: If the ACF magnetic file is missing, empty, or corrupted.

        Raises:
            UserWarning/RuntimeWarning: Issued by the underlying parser if the file 
                                        is missing or malformed.
        """
        return self.get_bader_property(property="magmom")