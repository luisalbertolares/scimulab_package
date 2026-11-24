# Custom exceptions for ChemicalEntity class in scimulib.
class MissingDataError(ValueError):
    """Raised when accessing an attribute that was not provided at initialization."""
    pass


class MissingGeometryError(ValueError):
    """Raised when a geometric property is accessed without an assigned geometry."""
    pass


class NonPeriodicGeometryError(TypeError):
    """Raised when a periodic property is accessed on a non-periodic Molecule."""
    pass


class MissingSitePropertyError(ValueError):
    """Raised when a required site property is not present in the structure."""
    pass


# The following classes define custom exceptions for handling errors related to VASP output file extraction.
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


class PotcarError(VaspExtractionError):
    """
    Exception raised for errors occurring during the parsing or extraction of 
    data from the 'POTCAR' file.
    """
    pass


# The following classes define custom exceptions for handling errors related to Bader analysis output files.
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


# The following classes define custom exceptions for handling errors related to energy calculations.
class EnergyCalculationError(Exception):
    """Base exception for errors occurring during thermodynamic or energy calculations."""
    pass


class MissingReferenceError(EnergyCalculationError):
    """Raised when one or more required reference energies are missing."""

    def __init__(self, message: str, missing_species: set[str] | list[str] | None = None) -> None:
        super().__init__(message)
        self.missing_species = sorted(missing_species) if missing_species is not None else []


class StoichiometryError(EnergyCalculationError):
    """Raised when atomic fractions, counts, or stoichiometric coefficients are invalid."""
    pass


# Custom exceptions for Bader calculations
class BaderCalculationError(Exception):
    """Base exception for errors raised during Bader net charge calculations."""
    pass


class SizeMismatchError(BaderCalculationError):
    """Raised when the geometry and the Bader charges array have different lengths."""
    pass


class MissingValenceElementError(BaderCalculationError):
    """Raised when one or more elements in the geometry are missing from the valence dictionary."""
    pass


class InvalidValenceValueError(BaderCalculationError):
    """Raised when a valence dictionary value is not a valid finite real number."""
    pass


class ChargeNeutralityError(BaderCalculationError):
    """Raised when the sum of net Bader charges deviates from zero beyond tolerance."""
    pass


class MagneticMomentMismatchError(BaderCalculationError):
    """Raised when the sum of Bader magnetic moments deviates from the total magnetic moment."""
    pass