from pathlib import Path

from math import isclose
import numpy as np

from pymatgen.core import Lattice, Structure

from scimulib.scimulib import ChemicalEntity


# Path of the test output files from VASP (Au7Co6 cluster)
TEST_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = TEST_DIRECTORY / "test_data"

# Reference energy, magnetic moment and geometry
with open (DATA_DIRECTORY / 'energy.txt', 'r') as f:
    energy_txt = float(f.read().strip())

with open (DATA_DIRECTORY / 'magmom.txt', 'r') as f:
    magmom_txt = float(f.read().strip())

bader_charges = np.array([
    11.155531,
    11.188453,
    11.144737,
    11.110426,
    11.172038,
    11.312841,
    11.312844,
    8.226238,
    8.905799,
    8.921268,
    8.846117,
    8.867026,
    8.836679
])

bader_magnetic_moments = np.array([
    0.024969,
    0.078398,
    0.078382,
    0.025017,
    0.053099,
    0.159327,
    0.159328,
    1.900855,
    2.104810,
    2.099021,
    2.109633,
    2.102573,
    2.104588
])

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
test_structure.add_site_property("bader_charge", bader_charges)
test_structure.add_site_property("bader_magmom", bader_magnetic_moments)

test_entity = ChemicalEntity(
    name="Au7Co6",
    total_energy=energy_txt,
    net_magmom=magmom_txt,
    geometry=test_structure
)

def test_chemical_entity_properties():
    assert test_entity.name == "Au7Co6"
    assert isclose(test_entity.total_energy, energy_txt, rel_tol=1e-6)
    assert isclose(test_entity.net_magmom, magmom_txt, rel_tol=1e-6)
    assert np.allclose(test_entity.bader_charges, bader_charges)
    assert np.allclose(test_entity.bader_magnetic_moments, bader_magnetic_moments)
    assert np.allclose(test_entity.fractional_coords, test_structure.frac_coords)
    assert np.allclose(test_entity.cartesian_coords, test_structure.cart_coords)
    assert test_entity.composition == test_structure.composition

def test_chemical_entity_from_vasp():
    vasp_entity = ChemicalEntity.from_vasp(
        name="Au7Co6",
        directory=DATA_DIRECTORY
    )
    assert vasp_entity.name == "Au7Co6"
    assert isclose(vasp_entity.total_energy, energy_txt, rel_tol=1e-6)
    assert isclose(vasp_entity.net_magmom, magmom_txt, rel_tol=1e-6)
    assert np.allclose(vasp_entity.bader_charges, bader_charges)
    assert np.allclose(vasp_entity.bader_magnetic_moments, bader_magnetic_moments)
    assert np.allclose(vasp_entity.fractional_coords, test_structure.frac_coords)
    assert np.allclose(vasp_entity.cartesian_coords, test_structure.cart_coords)
    assert vasp_entity.composition == test_structure.composition
