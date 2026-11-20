from pathlib import Path

from math import isclose
import numpy as np

from pymatgen.core import Lattice, Structure

from scimulib.abinitio_parser import VaspParser


# Path of the test output files from VASP (Au7Co6 cluster)
TEST_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = TEST_DIRECTORY / "test_data"

vasp_parser = VaspParser(DATA_DIRECTORY)

# Reference energy, magnetic moment and geometry
with open (DATA_DIRECTORY / 'energy.txt', 'r') as f:
    energy_txt = float(f.read().strip())

with open (DATA_DIRECTORY / 'magmom.txt', 'r') as f:
    magmom_txt = float(f.read().strip())




def test_vasp_parser_energy_vasprun():
    assert isclose(vasp_parser.get_total_energy(), energy_txt, abs_tol=0.0001)

def test_vasp_parser_magmom_vasprun():

    assert isclose(vasp_parser.get_net_magmom(), magmom_txt, abs_tol=0.0001)

def test_vasp_parser_structure_vasprun():
    assert isclose(vasp_parser.get_structure(), energy_txt, abs_tol=0.0001)


def test_vasp_calculation_oszicar():
    
    vasp_parser._vasprun_path = None

    assert isclose(vasp_parser.get_total_energy(), energy_txt, abs_tol=0.0001)
    assert isclose(vasp_parser.get_net_magmom(), magmom_txt, abs_tol=0.0001)

def test_vasp_calculation_outcar():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    with open (os.path.join(path, 'energy.txt'), 'r') as f:
        energy_txt = float(f.read().strip())

    with open (os.path.join(path, 'magmom.txt'), 'r') as f:
        magmom_txt = float(f.read().strip())
    
    vasp_calc = VaspCalculation(path)
    vasp_calc._vasprun_path = None
    vasp_calc._oszicar_path = None

    assert isclose(vasp_calc.total_energy, energy_txt, abs_tol=0.0001)
    assert isclose(vasp_calc.net_magmom, magmom_txt, abs_tol=0.0001)

def test_vasp_calculation_geometry_vasprun():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    vasp_calc = VaspCalculation(path)

    assert len(vasp_calc.geometry) == 13
    assert vasp_calc.geometry.get_chemical_symbols() == ['Au', 'Au', 'Au', 'Au', 'Au', 'Au', 'Au',
                                                         'Co', 'Co', 'Co', 'Co', 'Co', 'Co']

def test_vasp_calculation_geometry_poscar():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    vasp_calc = VaspCalculation(path)
    vasp_calc._vasprun_path = None

    assert len(vasp_calc.geometry) == 13
    assert vasp_calc.geometry.get_chemical_symbols() == ['Au', 'Au', 'Au', 'Au', 'Au', 'Au', 'Au',
                                                         'Co', 'Co', 'Co', 'Co', 'Co', 'Co']

def test_vasp_calculation_geometry_contcar():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    vasp_calc = VaspCalculation(path)
    vasp_calc._vasprun_path = None
    vasp_calc._poscar_path = None

    assert len(vasp_calc.geometry) == 13
    assert vasp_calc.geometry.get_chemical_symbols() == ['Au', 'Au', 'Au', 'Au', 'Au', 'Au', 'Au',
                                                         'Co', 'Co', 'Co', 'Co', 'Co', 'Co']

def test_vasp_calculation_acf_charge():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    vasp_calc = VaspCalculation(path)

    assert isclose(np.sum(vasp_calc.bader_charge), 131, abs_tol=0.1)

def test_vasp_calculation_acf_mag():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    vasp_calc = VaspCalculation(path)

    assert isclose(np.sum(vasp_calc.bader_magmom), 13, abs_tol=0.1)

def test_vasp_calculation_composition():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    vasp_calc = VaspCalculation(path)

    assert vasp_calc.composition == {'Au':7, 'Co':6}