#!/bin/bash
# Exit immediately if a command exits with a non-zero status
set -e

echo "1. Installing system compilers and mathematical libraries (requires sudo)..."
# Requirements: C compiler (build-essential/gcc), Fortran (gfortran), CMake, LAPACK, and BLAS
sudo apt-get update
sudo apt-get install -y build-essential gfortran cmake liblapack-dev libblas-dev

echo "2. Installing dependencies from PyPI..."
# Installing build requirements (meson, ninja), math dependencies,
# cosymlib requirements, and your analysis and testing libraries.
pip install \
    meson ninja "numpy>=2.0" setuptools scipy \
    matplotlib "PyYAML>=5.1" "huckelpy>=0.2" "pointgroup>=0.4" \
    pytest ase pymatgen

echo "3. Preparing temporary build environment..."
mkdir -p build_libs_tmp
cd build_libs_tmp

echo "4. Cloning and compiling symgroup (symgroupy)..."
git clone https://github.com/abelcarreras/symgroup.git
cd symgroup/python
python setup.py install
cd ../..

echo "5. Cloning and compiling WFNSYM (wfnsympy)..."
git clone https://github.com/abelcarreras/WFNSYM.git
cd WFNSYM/python
python setup.py install
cd ../..

echo "6. Cloning and compiling cosymlib..."
git clone https://github.com/GrupEstructuraElectronicaSimetria/cosymlib.git
cd cosymlib
python setup.py install
cd ../..

echo "7. Cleaning up temporary directory..."
rm -rf build_libs_tmp

echo "8. Generating final requirements.txt file..."
# Generate a complete, unfiltered snapshot of the newly built environment
python3 -m pip freeze > requirements.txt

echo "Installation completed successfully!"