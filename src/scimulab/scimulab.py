import os
import warnings
from pathlib import Path
from functools import cached_property

import numpy as np

from ase import Atoms
from ase.io import read, write
from pymatgen.io.vasp.outputs import Vasprun, Oszicar
from pymatgen.io.ase import AseAtomsAdaptor

from src.scimulab.vasp_utils import extract_outcar_property, parse_acf_file


class VaspCalculation:

    def __init__(self, directory: str):
        self.directory = directory

        if not self.directory.is_dir():
            raise FileNotFoundError(
                f"The specified directory '{self.directory}' does not exist or is invalid"
            )
        
        self._vasprun_path: Path = Path(os.path.join(directory, "vasprun.xml"))
        self._poscar_path: Path = Path(os.path.join(directory, "POSCAR"))
        self._contcar_path: Path = Path(os.path.join(directory, "CONTCAR"))
        self._outcar_path: Path = Path(os.path.join(directory, "OUTCAR"))
        self._oszicar_path: Path = Path(os.path.join(directory, "OSZICAR"))
        self._doscar_path: Path = Path(os.path.join(directory, "DOSCAR"))
        self._acf_charge_path: Path = Path(os.path.join(directory, "ACF_charge.dat"))
        self._acf_mag_path: Path = Path(os.path.join(directory, "ACF_mag.dat"))

    @cached_property
    def geometry(self) -> Atoms:
        """
        Extracts the final atomic geometry from a VASP calculation directory.

        Attempts to recover the geometry as an ASE Atoms object across three sequential stages:
            1. vasprun.xml: Parses the structured XML file via Pymatgen and converts the 
               last recorded structure into an ASE Atoms object.
            2. POSCAR: Falls back to reading the initial geometry file using ASE.
            3. CONTCAR: Falls back to reading the final geometry file using ASE.

        Returns:
            Atoms: The final geometry as an ASE Atoms object if successfully parsed.
            None: If all stages fail due to missing, empty, or corrupted files.

        Raises:
            UserWarning/RuntimeWarning: Issued with the technical error message if a 
                                        stage fails.
        """
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
            self.geometry = AseAtomsAdaptor.get_atoms(vasprun.structures[-1])
            return self.geometry

        except FileNotFoundError:
            warnings.warn(
                f"vasprun.xml file was not found. Trying POSCAR/CONTCAR.",
                category=UserWarning,
                stacklevel=2
            )
            
        except Exception as e:
            warnings.warn(
                f"Error reading vasprun.xml ({e}). Trying POSCAR/CONTCAR.",
                category=RuntimeWarning,
                stacklevel=2
            )

        try:
            self.geometry = read(self._poscar_path)
            return self.geometry

        except Exception:
            pass
        
        try:
            self.geometry = read(self._contcar_path)
            return self.geometry

        except FileNotFoundError:
            warnings.warn(
                f"Neither the POSCAR nor the CONTCAR file was found. No geometry could be recovered.",
                category=UserWarning,
                stacklevel=2
            )
            self.geometry = None
            return None

        except Exception as e:
            warnings.warn(
                f"Critical error processing POSCAR/CONTCAR ({e}). No geometry could be recovered.",
                category=RuntimeWarning,
                stacklevel=2
            )
            self.geometry = None
            return None

    @cached_property
    def total_energy(self) -> float:
        """
        Extracts the total energy from a VASP calculation directory.

        Attempts to recover the total energy (in eV) across three sequential stages:
            1. vasprun.xml: Parses the structured XML file via Pymatgen.
            2. OSZICAR: Extracts the 'E0' key from the last recorded ionic step.
            3. OUTCAR: Scans backward from the end of the file in binary mode to find 
               'energy(sigma->0)', avoiding high RAM usage on giant files.

        Returns:
            float: The final total energy if successfully parsed, updates 'self.total_energy'.
            None: If all stages fail due to missing, empty, or corrupted files.

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
                                ionic_step_skip=None,
                                ionic_step_offset=-1
            )
            self.total_energy = float(vasprun.final_energy)
            return self.total_energy
            
        except FileNotFoundError:
            warnings.warn(
                f"vasprun.xml file was not found. Trying OSZICAR.",
                category=UserWarning,
                stacklevel=2
            )
             
        except Exception as e:
            warnings.warn(
                f"Error reading vasprun.xml ({e}). Trying OSZICAR.",
                category=RuntimeWarning,
                stacklevel=2
            )

        # Since vasprun.xml didn't work, we try to extract final energy from OSZICAR
        try:
            oszicar = Oszicar(self._oszicar_path)
            self.total_energy = float(oszicar.ionic_steps[-1]['E0'])
            return self.total_energy

        except FileNotFoundError:
            warnings.warn(
                f"OSZICAR file was not found. Trying OUTCAR.",
                category=UserWarning,
                stacklevel=2
            )

        except Exception as e:
            warnings.warn(
                f"Error reading OSZICAR ({e}). Trying OUTCAR.",
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
            self.total_energy = tot_energy

        return getattr(self, 'total_energy', None)

    @cached_property
    def net_magmom(self) -> float:
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
            float: The final net magnetic moment if successfully parsed.
            None: If all extraction stages fail due to missing or corrupted files.

        Raises:
            UserWarning/RuntimeWarning: Issued with the technical error message if 
                                        a stage fails.
        """
        # Extracting net magnetic moment from OSZICAR
        try:
            oszicar = Oszicar(self._oszicar_path)
            self.net_magmom = oszicar.ionic_steps[-1].get('mag', None)
            return self.net_magmom

        except FileNotFoundError:
            warnings.warn(
                f"OSZICAR file was not found. Trying OUTCAR.",
                category=UserWarning,
                stacklevel=2
            )

        except Exception as e:
            warnings.warn(
                f"Error reading OSZICAR ({e}). Trying OUTCAR.",
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
            self.net_magmom = magmom

        return getattr(self, 'net_magmom', None)

    @cached_property
    def bader_charge(self) -> np.ndarray:
        """
        Extracts the atomic charges calculated via Bader charge analysis.

        Utilizes the generalized `parse_acf_file` function to read the standard 
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
            self.bader_charge = q_bader

        return getattr(self, 'bader_charge', None)

    @cached_property
    def bader_magmom(self) -> np.ndarray:
        """
        Extracts the local atomic magnetic moments calculated via Bader analysis.

        Utilizes the generalized `parse_acf_file` function to read the spin-polarized 
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
            self.bader_magmom = mag_bader

        return getattr(self, 'bader_magmom', None)
