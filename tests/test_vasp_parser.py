from pathlib import Path
import logging

from math import isclose
import numpy as np
import pytest

from pymatgen.core import Lattice, Structure

from scimulib.abinitio_parser import VaspParser
from scimulib.exceptions import OszicarError, VaspExtractionError


# Path of the test output files from VASP (Au7Co6 cluster)
TEST_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = TEST_DIRECTORY / "test_data"

# Reference energy, magnetic moment and geometry
with open (DATA_DIRECTORY / 'energy.txt', 'r') as f:
    energy_txt = float(f.read().strip())

with open (DATA_DIRECTORY / 'magmom.txt', 'r') as f:
    magmom_txt = float(f.read().strip())

# Reference structure for the tests
lattice = Lattice.cubic(21)
species = ['Au'] * 7 + ['Co'] * 6
coords = [
    [0.5012406651094983,  0.4073275107542340,  0.4104206126129512],
    [0.5022071923580069,  0.4168117137338751,  0.5939111034924919],
    [0.5939111034924919,  0.4168117137338751,  0.5022071923580069],
    [0.4104206126129512,  0.4073275107542340,  0.5012406651094983],
    [0.4139187530975761,  0.5061287488693690,  0.4139187530975761],
    [0.4063587383186778,  0.5088486782956219,  0.5839164616914249],
    [0.5839164616914249,  0.5088486782956219,  0.4063587383186778],
    [0.5000000000000000,  0.5000000000000000,  0.5000000000000000],
    [0.4997179022681417,  0.5926953036560114,  0.4278038193796933],
    [0.5035821153555964,  0.5797114286814797,  0.5775458069905356],
    [0.4278038193796933,  0.5926953036560114,  0.4997179022681417],
    [0.5775458069905356,  0.5797114286814797,  0.5035821153555964],
    [0.5829062494204733,  0.5082438485277284,  0.5829062494204733]
]
test_structure = Structure(lattice, species, coords)

# These tests validate the functionality of the VaspParser class for extracting 
# energy, magnetic moment, and structure from vasprun.xml.
def test_vasp_parser_energy_vasprun():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    assert isclose(vasp_parser.get_total_energy(), energy_txt, abs_tol=0.0001)

def test_vasp_parser_magmom_vasprun():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    assert isclose(vasp_parser.get_net_magmom(), magmom_txt, abs_tol=0.0001)

def test_vasp_parser_structure_vasprun():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    structure = vasp_parser.get_structure()
    assert isinstance(structure, Structure)
    assert structure.num_sites == test_structure.num_sites
    assert structure.frac_coords.shape == test_structure.frac_coords.shape
    assert np.allclose(structure.frac_coords, test_structure.frac_coords, atol=1e-6)
    assert structure.species == test_structure.species
    assert np.allclose(structure.lattice.matrix, test_structure.lattice.matrix, atol=1e-6)

# Test VaspParser behavior when vasprun.xml is not available and it 
# should fallback to OSZICAR.
def test_vasp_parser_energy_oszicar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    vasp_parser._vasprun_path = None
    assert isclose(vasp_parser.get_total_energy(), energy_txt, abs_tol=0.0001)

def test_vasp_parser_magmom_oszicar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    vasp_parser._vasprun_path = None
    assert isclose(vasp_parser.get_net_magmom(), magmom_txt, abs_tol=0.0001)

# Test VaspParser behavior when OSZICAR is not available and it should fallback to OUTCAR.
# For get_structure, it should fallback to POSCAR, CONTCAR, and finally OUTCAR.
def test_vasp_parser_energy_outcar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    vasp_parser._vasprun_path = None
    vasp_parser._oszicar_path = None
    assert isclose(vasp_parser.get_total_energy(), energy_txt, abs_tol=0.0001)

def test_vasp_parser_magmom_outcar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    vasp_parser._vasprun_path = None
    vasp_parser._oszicar_path = None
    assert isclose(vasp_parser.get_net_magmom(), magmom_txt, abs_tol=0.0001)

def test_missing_oszicar_falls_back_to_outcar_without_warning(tmp_path, caplog):
    parser = VaspParser(DATA_DIRECTORY)
    parser._oszicar_path = tmp_path / "OSZICAR"
    caplog.set_level(logging.DEBUG)

    assert isclose(parser.get_net_magmom(), magmom_txt, abs_tol=0.0001)
    records = [
        record for record in caplog.records
        if "OSZICAR failed to provide net magnetic moment" in record.message
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG

def test_invalid_oszicar_still_warns_when_using_outcar(monkeypatch, caplog):
    parser = VaspParser(DATA_DIRECTORY)

    def fail_oszicar(**kwargs):
        raise OszicarError("invalid OSZICAR")

    monkeypatch.setattr(parser, "extract_oszicar_property", fail_oszicar)
    caplog.set_level(logging.DEBUG)

    assert isclose(parser.get_net_magmom(), magmom_txt, abs_tol=0.0001)
    records = [
        record for record in caplog.records
        if "OSZICAR failed to provide net magnetic moment" in record.message
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING

@pytest.mark.parametrize(
    ("method", "expected_sources"),
    [
        ("get_total_energy", ["vasprun.xml", "OSZICAR", "OUTCAR"]),
        ("get_net_magmom", ["OSZICAR", "OUTCAR"]),
        ("get_structure", ["vasprun.xml", "POSCAR", "CONTCAR", "OUTCAR"]),
        ("get_valence_electrons", ["OUTCAR", "POTCAR"]),
    ],
)
def test_missing_vasp_files_are_debug_logged(tmp_path, caplog, method, expected_sources):
    parser = VaspParser(tmp_path)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(VaspExtractionError):
        getattr(parser, method)()

    assert len(caplog.records) == len(expected_sources)
    for record, source in zip(caplog.records, expected_sources):
        assert source in record.message
        assert record.levelno == logging.DEBUG

@pytest.mark.parametrize(
    ("method", "extractor", "source"),
    [
        ("get_total_energy", "extract_vasprun_property", "vasprun.xml"),
        ("get_structure", "extract_vasprun_property", "vasprun.xml"),
        ("get_valence_electrons", "extract_outcar_valence_electrons", "OUTCAR"),
    ],
)
def test_vasp_extraction_errors_still_warn(tmp_path, monkeypatch, caplog, method, extractor, source):
    parser = VaspParser(tmp_path)

    def fail(*args, **kwargs):
        raise ValueError("invalid output")

    monkeypatch.setattr(parser, extractor, fail)
    caplog.set_level(logging.DEBUG)

    with pytest.raises(VaspExtractionError):
        getattr(parser, method)()

    assert source in caplog.records[0].message
    assert caplog.records[0].levelno == logging.WARNING

def test_vasp_parser_valence_electrons_outcar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    assert vasp_parser.get_valence_electrons() == {"Au": 11.0, "Co": 9.0}

def test_vasp_parser_structure_poscar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    vasp_parser._vasprun_path = None
    vasp_parser._oszicar_path = None
    structure = vasp_parser.get_structure()
    assert isinstance(structure, Structure)
    assert structure.num_sites == test_structure.num_sites
    assert structure.frac_coords.shape == test_structure.frac_coords.shape
    assert np.allclose(structure.frac_coords, test_structure.frac_coords, atol=1e-6)
    assert structure.species == test_structure.species
    assert np.allclose(structure.lattice.matrix, test_structure.lattice.matrix, atol=1e-6)

def test_vasp_parser_structure_contcar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    vasp_parser._vasprun_path = None
    vasp_parser._oszicar_path = None
    vasp_parser._poscar_path = None
    structure = vasp_parser.get_structure()
    assert isinstance(structure, Structure)
    assert structure.num_sites == test_structure.num_sites
    assert structure.frac_coords.shape == test_structure.frac_coords.shape
    assert np.allclose(structure.frac_coords, test_structure.frac_coords, atol=1e-6)
    assert structure.species == test_structure.species
    assert np.allclose(structure.lattice.matrix, test_structure.lattice.matrix, atol=1e-6)

def test_vasp_parser_structure_outcar():
    vasp_parser = VaspParser(DATA_DIRECTORY)
    vasp_parser._vasprun_path = None
    vasp_parser._oszicar_path = None
    vasp_parser._poscar_path = None
    vasp_parser._contcar_path = None
    structure = vasp_parser.get_structure()
    assert isinstance(structure, Structure)
    assert structure.num_sites == test_structure.num_sites
    assert structure.frac_coords.shape == test_structure.frac_coords.shape
    assert np.allclose(structure.frac_coords, test_structure.frac_coords, atol=1e-6)
    assert structure.species == test_structure.species
    assert np.allclose(structure.lattice.matrix, test_structure.lattice.matrix, atol=1e-6)