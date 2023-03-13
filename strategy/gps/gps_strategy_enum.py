"""
This module provides an enumeration for GPS strategy options.

This module provides an enumeration for GPS strategy options, allowing for
    the selection of either a CPU or GPU-based GPS strategy.
"""

from enum import Enum


class GPSStrategyEnum(Enum):
    """
    An enumeration for GPS strategy options.

    This enumeration allows for the selection of either a CPU or GPU-based GPS strategy.

    Values:
    - CPU: A CPU-based GPS strategy.
    - GPU: A GPU-based GPS strategy.
    """
    CPU = 0
    GPU = 1
