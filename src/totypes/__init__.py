"""totypes - Custom datatypes useful in a topology optimization context."""

__version__ = "0.0.0"
__author__ = "Martin Schubert <mfschubert@gmail.com>"

from totypes.types import BoundedArray
from totypes.types import Density2DArray
from totypes.types import extract_lower_bound as extract_lower_bound
from totypes.types import extract_upper_bound as extract_upper_bound

CUSTOM_TYPES = (BoundedArray, Density2DArray)
