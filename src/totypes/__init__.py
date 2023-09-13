"""totypes - Custom datatypes useful in a topology optimization context."""

__version__ = "0.0.0"
__author__ = "Martin Schubert <mfschubert@gmail.com>"
__all__ = ["BoundedArray", "Density2D"]

from totypes.types import BoundedArray as BoundedArray
from totypes.types import Density2D as Density2D

CUSTOM_TYPES = (BoundedArray, Density2D)
