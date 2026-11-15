import warnings
from pathlib import Path
from functools import cached_property

import numpy as np
from collections import Counter

from pymatgen.io.vasp.outputs import Vasprun, Oszicar
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure

from .vasp_utils import extract_outcar_property, extract_outcar_structure, parse_acf_file


class VaspParser:
    """
    A class to parse the results of a VASP calculation. It provides methods to 
    extract key properties such as geometry, total energy and net magnetic moment 
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
        self._acf_charge_path: Path = self._directory / "ACF_charge.dat"
        self._acf_mag_path: Path = self._directory / "ACF_mag.dat"
    
    @cached_property
    def total_energy(self) -> float | None:
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
            self._total_energy = float(vasprun.final_energy)
            return self._total_energy
            
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
            self._total_energy = float(oszicar.ionic_steps[-1]['E0'])
            return self._total_energy

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
        tot_energy = extract_outcar_property(
            outcar_path=self._outcar_path,
            target_patterns="energy(sigma->0)",
            property_name="total energy"
        )
        if tot_energy is not None:
            self._total_energy = tot_energy

        else:
            warnings.warn(
                "Total energy could not be determined.",
                category=UserWarning,
                stacklevel=2
            )
            return None

        return self._total_energy

    @cached_property
    def net_magmom(self) -> float | None:
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
            self._net_magmom = oszicar.ionic_steps[-1].get('mag', None)
            return self._net_magmom

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
        magmom = extract_outcar_property(
            outcar_path=self._outcar_path,
            target_patterns=["magnetization", "number of electron"],
            property_name="net magnetic moment"
        )
        if magmom is not None:
            self._net_magmom = magmom

        else:
            warnings.warn(
                "Net magnetic moment could not be determined.",
                category=UserWarning,
                stacklevel=2
            )
            return None

        return self._net_magmom

    @cached_property
    def structure(self) -> Structure | None:
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

        Then, attempts to recover the Bader charges and magnetic moments from the ACF files:
            1. ACF_charge.dat: Parses the Bader charge output file and adds them as a site 
               property to the Structure object.
            2. ACF_mag.dat: Parses the Bader magnetic moment output file and adds them as a 
               site property to the Structure object.

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
            self._structure = vasprun.structures[-1]

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
            self._structure = Structure.from_file(self._poscar_path)

        except Exception:
            pass
        
        try:
            self._structure = Structure.from_file(self._contcar_path)

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
            self._structure = structure

        else:
            warnings.warn(
                "Geometry could not be recovered. Structure object cannot be created.",
                category=UserWarning,
                stacklevel=2
            )
            return None

        # 2. Extract Bader charges and introduce them as a property of the Structure object
        q_bader = parse_acf_file(
            acf_path=self._acf_charge_path
        )
        if q_bader is not None:
            self._structure.add_site_property("bader_charge", q_bader)

        else:
            warnings.warn(
                "Bader charge data could not be recovered. 'bader_charge' property "
                "will not be added to the Structure object.",
                category=UserWarning,
                stacklevel=2
            )

        # 3. Extract Bader magnetic moments and introduce them as a property of the Structure object
        mag_bader = parse_acf_file(
            acf_path=self._acf_mag_path
        )
        if mag_bader is not None:
            self._structure.add_site_property("bader_magmom", mag_bader)

        else:
            warnings.warn(
                "Bader magnetic moment data could not be recovered. 'bader_magmom' property "
                "will not be added to the Structure object.",
                category=UserWarning,
                stacklevel=2
            )

        return self._structure

    @cached_property
    def bader_charge(self) -> np.ndarray:
        """
        Extracts the atomic charges calculated via Bader charge analysis.

        Utilizes the generalized 'parse_acf_file' function to read the standard 
        Bader charge output file (typically 'ACF.dat'). The extracted values represent 
        the total number of electrons assigned to each atom's Bader volume.

        Returns:
            np.ndarray: A 1D NumPy array containing the Bader charge for each atom.
            None: If the ACF charge file is missing, empty, or corrupted.

        Raises:
            UserWarning/RuntimeWarning: Issued by the underlying parser if the file 
                                        is missing or malformed.
        """
        q_bader = parse_acf_file(
            acf_path=self._acf_charge_path
        )
        if q_bader is not None:
            self._bader_charge = q_bader

        return getattr(self, '_bader_charge', None)

    @cached_property
    def bader_magmom(self) -> np.ndarray:
        """
        Extracts the local atomic magnetic moments calculated via Bader analysis.

        Utilizes the generalized 'parse_acf_file' function to read the spin-polarized 
        Bader output file (typically 'ACF_mag.dat'). The extracted values represent 
        the net spin (spin-up minus spin-down electrons) within each atom's Bader volume.

        Returns:
            np.ndarray: A 1D NumPy array containing the local magnetic moment for each atom.
            None: If the ACF magnetic file is missing, empty, or corrupted.

        Raises:
            UserWarning/RuntimeWarning: Issued by the underlying parser if the file 
                                        is missing or malformed.
        """
        mag_bader = parse_acf_file(
            acf_path=self._acf_mag_path
        )
        if mag_bader is not None:
            self._bader_magmom = mag_bader

        return getattr(self, '_bader_magmom', None)

    @property
    def composition(self) -> dict[str, int]:
        """
        Get the chemical composition of the system.

        Returns:
            dict[str, int]: A dictionary where keys are chemical symbols (strings) and
                            values are their respective counts (integers).
        """
        return dict(Counter(self.geometry.get_chemical_symbols()))