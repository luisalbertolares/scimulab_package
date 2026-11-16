import warnings
from pathlib import Path

import numpy as np

from pymatgen.io.vasp.outputs import Vasprun, Oszicar
from pymatgen.core import Structure

from scimulib.vasp_utils import extract_outcar_property, extract_outcar_structure


class VaspParser:
    """
    A class to parse the results of a VASP calculation. It provides methods to 
    extract key properties such as total energy, net magnetic moment and geometry
    from the VASP output files.
    """

    def __init__(self, directory: str | Path):
        self._directory = Path(directory)

        if not self._directory.is_dir():
            raise FileNotFoundError(
                f"The specified directory '{self._directory}' does not exist or is invalid."
            )
        
        self._vasprun_path: Path = self._directory / "vasprun.xml"
        self._poscar_path: Path = self._directory / "POSCAR"
        self._contcar_path: Path = self._directory / "CONTCAR"
        self._outcar_path: Path = self._directory / "OUTCAR"
        self._oszicar_path: Path = self._directory / "OSZICAR"
        self._doscar_path: Path = self._directory / "DOSCAR"
    
    def get_total_energy(self) -> float | None:
        """
        Extracts the total energy from a VASP calculation directory.

        Attempts to recover the total energy (in eV) across three sequential stages:
            1. vasprun.xml: Parses the structured XML file via Pymatgen.
            2. OSZICAR: Extracts the 'E0' key from the last recorded ionic step.
            3. OUTCAR: Scans backward from the end of the file in binary mode to find 
               'energy(sigma->0)', avoiding high RAM usage on giant files.

        Returns:
            float | None: The total energy if successfully parsed, None otherwise.

        Raises:
            UserWarning/RuntimeWarning: Issued with the technical error message if 
                                        a stage fails.
        """

        try:
            # Extracting final energy from vasprun.xml without full parsing
            vasprun = Vasprun(self._vasprun_path,
                                parse_dos=False,
                                parse_eigen=False,
                                parse_projected_eigen=False,
                                parse_potcar_file=False,
                                ionic_step_offset=-1
            )
            return float(vasprun.final_energy)
            
        except FileNotFoundError:
            warnings.warn(
                "vasprun.xml file was not found. Trying OSZICAR to get the total energy.",
                category=UserWarning,
                stacklevel=2
            )
             
        except Exception as e:
            warnings.warn(
                f"Error reading vasprun.xml ({e}). Trying OSZICAR to get the total energy.",
                category=RuntimeWarning,
                stacklevel=2
            )

        # Since vasprun.xml didn't work, we try to extract final energy from OSZICAR
        try:
            oszicar = Oszicar(self._oszicar_path)
            return float(oszicar.ionic_steps[-1]['E0'])

        except FileNotFoundError:
            warnings.warn(
                "OSZICAR file was not found. Trying OUTCAR to get the total energy.",
                category=UserWarning,
                stacklevel=2
            )

        except Exception as e:
            warnings.warn(
                f"Error reading OSZICAR ({e}). Trying OUTCAR to get the total energy.",
                category=RuntimeWarning,
                stacklevel=2
            )

        # Instead of parsing OUTCAR with Pymatgen, we process it as plain text 
        # from the end backward to avoid memory overload
        total_energy = extract_outcar_property(
            outcar_path=self._outcar_path,
            target_patterns="energy(sigma->0)",
            property_name="total energy"
        )
        if total_energy is not None:
            return total_energy

        else:
            warnings.warn(
                "Total energy could not be recovered.",
                category=UserWarning,
                stacklevel=2
            )
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

        Raises:
            UserWarning/RuntimeWarning: Issued with the technical error message if 
                                        a stage fails.
        """
        # Extracting net magnetic moment from OSZICAR
        try:
            oszicar = Oszicar(self._oszicar_path)
            return oszicar.ionic_steps[-1].get('mag', None)

        except FileNotFoundError:
            warnings.warn(
                "OSZICAR file was not found. Trying OUTCAR to get the net magnetic moment.",
                category=UserWarning,
                stacklevel=2
            )

        except Exception as e:
            warnings.warn(
                f"Error reading OSZICAR ({e}). Trying OUTCAR to get the net magnetic moment.",
                category=RuntimeWarning,
                stacklevel=2
            )

        # Since OSZICAR file didn't work, we parse the OUTCAR file
        net_magmom = extract_outcar_property(
            outcar_path=self._outcar_path,
            target_patterns=["magnetization", "number of electron"],
            property_name="net magnetic moment"
        )
        if net_magmom is not None:
            return net_magmom

        else:
            warnings.warn(
                "Net magnetic moment could not be recovered.",
                category=UserWarning,
                stacklevel=2
            )
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

        Raises:
            UserWarning/RuntimeWarning: Issued with the technical error message if a 
                                        stage fails.
        """
        # 1. Extract geometry and create a Structure object (Pymatgen) 
        try:
            # Extracting geometry from vasprun.xml without full parsing
            vasprun = Vasprun(self._vasprun_path,
                                parse_dos=False,
                                parse_eigen=False,
                                parse_projected_eigen=False,
                                parse_potcar_file=False,
                                ionic_step_skip=None,
                                ionic_step_offset=-1
            )
            return vasprun.structures[-1]

        except FileNotFoundError:
            warnings.warn(
                "vasprun.xml file was not found. Trying POSCAR/CONTCAR to get the geometry.",
                category=UserWarning,
                stacklevel=2
            )
            
        except Exception as e:
            warnings.warn(
                f"Error reading vasprun.xml ({e}). Trying POSCAR/CONTCAR to get the geometry.",
                category=RuntimeWarning,
                stacklevel=2
            )

        # Since vasprun.xml didn't work, we try to extract geometry from POSCAR/CONTCAR
        try:
            return Structure.from_file(self._poscar_path)

        except Exception:
            pass
        
        try:
            return Structure.from_file(self._contcar_path)

        except FileNotFoundError:
            warnings.warn(
                "Neither the POSCAR nor the CONTCAR file was found. Trying OUTCAR to get the geometry.",
                category=UserWarning,
                stacklevel=2
            )

        except Exception as e:
            warnings.warn(
                f"Error processing POSCAR/CONTCAR ({e}). Trying OUTCAR to get the geometry.",
                category=RuntimeWarning,
                stacklevel=2
            )

        # Since POSCAR/CONTCAR didn't work, we try to extract geometry from OUTCAR
        structure = extract_outcar_structure(self._outcar_path)
        if structure is not None:
            return structure

        else:
            warnings.warn(
                "Geometry could not be recovered. Structure object cannot be created.",
                category=UserWarning,
                stacklevel=2
            )
            return None


class BaderParser:
    """
    A class to parse Bader charge and magnetic moment analysis results from a VASP 
    calculation directory. It provides methods to extract atomic charges and magnetic 
    moments from the corresponding ACF files generated by the Bader analysis.
    """

    # Dictionary that contains the expected ACF files for each property type
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
            raise ValueError(
                f"Invalid property '{property}'. Expected one of: {list(self.FILE_MAP.keys())}."
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