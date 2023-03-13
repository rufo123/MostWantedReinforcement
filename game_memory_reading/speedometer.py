"""
This module provides a `Speedometer` class that reads memory from a game and 
    returns the current speed of a car in mph.
It inherits from the `MemoryReader` class and uses the `ReturnValuesEnum` enumeration for
    the type of value to be returned.
"""

from game_memory_reading.memory_reader import MemoryReader
from game_memory_reading.return_values_enum import ReturnValuesEnum


class Speedometer(MemoryReader):
    """
       A class that extends MemoryReader and reads the Speedometer of the game.
       """

    def __init__(self):
        """
        Constructs an instance of Speedometer by calling MemoryReader constructor.
        """
        super().__init__(par_return_value_type=ReturnValuesEnum.INT,
                         par_offsets=[0xC, 0xC, 0x38, 0xC, 0x54],
                         par_module_base_address=0x00514824)

    def return_speed_mph(self) -> int:
        """
        Returns the current speed of car from Speedometer by calling the
            return_value_from_pointer_address method of MemoryReader.

        Returns
        -------
        int
            Current speed of car from Speedometer.
        """
        return super().return_value_from_pointer_address()
