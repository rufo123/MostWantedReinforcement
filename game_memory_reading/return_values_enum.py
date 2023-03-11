"""
This module contains the ReturnValuesEnum class, which defines a set of enumerated values for
    use in returning different data types.

Example usage:
"""
from enum import Enum


class ReturnValuesEnum(Enum):
    """
    A class that defines a set of enumerated values for use in returning different data types.

    Attributes:
        INT (int): An enumerated value representing an integer.
        FLOAT (float): An enumerated value representing a float.
    """
    INT = 0
    FLOAT = 1
