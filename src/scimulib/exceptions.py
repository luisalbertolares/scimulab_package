class VaspExtractionError(Exception):
    """
    Base exception for all errors related to the extraction of data from VASP 
    output files.
    
    This class allows for catching any VASP-related parsing or extraction failure 
    using a single general exception.
    """
    pass


class VasprunError(VaspExtractionError):
    """
    Exception raised for errors occurring during the parsing or extraction 
    of data from the 'vasprun.xml' file.
    """
    pass


class OszicarError(VaspExtractionError):
    """
    Exception raised for errors occurring during the parsing or extraction of 
    data from the 'OSZICAR' file.
    """
    pass


class OutcarError(VaspExtractionError):
    """
    Exception raised for errors occurring during the parsing or extraction of 
    data from the 'OUTCAR' file.
    """
    pass


class PropertyNotFoundError(OutcarError):
    """Exception raised when the target property pattern is not found in the OUTCAR file."""
    pass


class StructureNotFoundError(OutcarError):
    """Exception raised when essential structural data (species, lattice, coords) is missing."""
    pass


class BaderExtractionError(Exception):
    """
    Base exception for all errors related to the extraction of data from Bader analysis output files.
    """
    pass


class AcfError(BaderExtractionError):
    """
    Exception raised for errors occurring during the parsing or extraction of data from Bader ACF files.
    """
    pass