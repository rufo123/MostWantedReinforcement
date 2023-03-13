"""
This module provides a `LapProgress` class that reads memory from a game and 
    returns the percentage of the lap completed.
It inherits from the `MemoryReader` class and uses the `ReturnValuesEnum` enumeration for
    the type of value to be returned.
"""

from game_memory_reading.memory_reader import MemoryReader
from game_memory_reading.return_values_enum import ReturnValuesEnum


class LapProgress(MemoryReader):
    """
    A class that extends MemoryReader and reads the lap progress of the game.
    """

    def __init__(self):
        """
        Constructs an instance of LapProgress by calling MemoryReader constructor.
        """
        super().__init__(ReturnValuesEnum.FLOAT, [0x3C], 0x0052FDC4)

    def return_lap_completed_percent(self) -> float:
        """
        Returns the lap progress percentage by calling the return_value_from_pointer_address
            method of MemoryReader.

        Returns
        -------
        float
            The lap progress percentage.
        """
        return super().return_value_from_pointer_address()
