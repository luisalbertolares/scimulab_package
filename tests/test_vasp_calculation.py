import os
import pytest

from math import isclose
import numpy as np

from scimulab.scimulab import VaspCalculation


def test_vasp_calculation_vasprun():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    with open (os.path.join(path, 'energy.txt'), 'r') as f:
        energy_txt = float(f.read().strip())

    with open (os.path.join(path, 'magmom.txt'), 'r') as f:
        magmom_txt = float(f.read().strip())
    
    vasp_calc = VaspCalculation(path)

    assert isclose(vasp_calc.total_energy, energy_txt, abs_tol=0.0001)
    assert isclose(vasp_calc.net_magmom, magmom_txt, abs_tol=0.0001)

def test_vasp_calculation_oszicar():
    path = '/mnt/g/Unidades compartidas/Screening_AuMTiO2/data/sistemas/AuCo/cumulo/Au7Co6'

    with open (os.path.join(path, 'energy.txt'), 'r') as f:
        energy_txt = float(f.read().strip())

    with open (os.path.join(path, 'magmom.txt'), 'r') as f:
        magmom_txt = float(f.read().strip())
    
    vasp_calc = VaspCalculation(path)
    vasp_calc._vasprun_path = None

    assert isclose(vasp_calc.total_energy, energy_txt, abs_tol=0.0001)
    assert isclose(vasp_calc.net_magmom, magmom_txt, abs_tol=0.0001)

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