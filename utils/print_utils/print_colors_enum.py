"""
A module that defines an enumeration of ANSI color codes for use in console output.
"""
from enum import Enum


class PrintColorsEnum(Enum):
    """
    An enumeration of ANSI color codes for use in console output.

    Attributes:
        RED: The ANSI escape code for red text.
        YELLOW: The ANSI escape code for yellow text.
        BLUE: The ANSI escape code for blue text.
        ORANGE: The ANSI escape code for orange text.
        GREEN: The ANSI escape code for green text.
        VIOLET: The ANSI escape code for violet text.
        RED_ORANGE: The ANSI escape code for red-orange text.
        YELLOW_ORANGE: The ANSI escape code for yellow-orange text.
        YELLOW_GREEN: The ANSI escape code for yellow-green text.
        BLUE_GREEN: The ANSI escape code for blue-green text.
        BLUE_VIOLET: The ANSI escape code for blue-violet text.
        RED_VIOLET: The ANSI escape code for red-violet text.
        RESET: The ANSI escape code to reset the color to the default color.
    """
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ORANGE = '\033[38;5;208m'
    GREEN = '\033[92m'
    VIOLET = '\033[95m'
    RED_ORANGE = '\033[38;5;202m'
    YELLOW_ORANGE = '\033[38;5;214m'
    YELLOW_GREEN = '\033[38;5;112m'
    BLUE_GREEN = '\033[38;5;27m'
    BLUE_VIOLET = '\033[38;5;99m'
    RED_VIOLET = '\033[38;5;162m'
    RESET = '\033[0m'
