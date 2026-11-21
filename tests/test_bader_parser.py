from pathlib import Path

import numpy as np

from scimulib.abinitio_parser import BaderParser


# Path of the test output files from VASP (Au7Co6 cluster)
TEST_DIRECTORY = Path(__file__).parent
DATA_DIRECTORY = TEST_DIRECTORY / "test_data"

# np array for expected charges and magnetic moments
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

def test_bader_charges_and_magnetic_moments():
    parser = BaderParser(DATA_DIRECTORY, expected_atoms=13)
    charges = parser.get_charges()
    magnetic_moments = parser.get_magnetic_moments()

    assert np.allclose(charges, bader_charges, atol=1e-6)
    assert np.allclose(magnetic_moments, bader_magnetic_moments, atol=1e-6)