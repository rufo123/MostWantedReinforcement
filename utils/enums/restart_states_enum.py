"""
This module defines an enumeration for restart states.

The RestartStateEnum enumeration includes the following values:
- RESTART_STATE: indicates that the race is in a restart state.
- STANDINGS_STATE: indicates that the race is in a standings state.
- UNKNOWN_STATE: indicates that the race is in an unknown state.
"""

from enum import Enum


class RestartStateEnum(Enum):
    """
    An enumeration representing the possible restart states.
    """
    RESTART_STATE = 1
    STANDINGS_STATE = 2
    UNKNOWN_STATE = 3
