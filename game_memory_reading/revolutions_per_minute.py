"""
This module provides a `RevolutionsPerMinute` class that reads memory from a game and 
    returns the revolutions per minute of the car.
It inherits from the `MemoryReader` class and uses the `ReturnValuesEnum` enumeration for
    the type of value to be returned.
"""

from game_memory_reading.memory_reader import MemoryReader
from game_memory_reading.return_values_enum import ReturnValuesEnum


class RevolutionsPerMinute(MemoryReader):
    """
    A class that extends MemoryReader and reads the revolutions per minute (RPM) of the car from 
        the game.
    """

    def __init__(self):
        """
        Constructs an instance of RevolutionsPerMinute by calling MemoryReader constructor.
        """
        super().__init__(par_return_value_type=ReturnValuesEnum.FLOAT,
                         par_offsets=[0xA4, 0x54, 0x4, 0x50, 0x660],
                         par_module_base_address=0x005382EC)

    def return_revolutions_per_minute(self) -> float:
        """
        Returns the revolutions per minute of the car by calling the 
            return_value_from_pointer_address method of MemoryReader.

        Returns
        -------
        float
            Revolutions per minute of the car (RPM).
        """
        return super().return_value_from_pointer_address()
