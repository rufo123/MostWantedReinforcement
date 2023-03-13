"""
This module provides a `LapTime` class that reads memory from a game and returns 
    the time of the lap currently in.
It inherits from the `MemoryReader` class and uses the `ReturnValuesEnum` enumeration
    for the type of value to be returned.
"""
from game_memory_reading.memory_reader import MemoryReader
from game_memory_reading.return_values_enum import ReturnValuesEnum


class LapTime(MemoryReader):
    """
    A class that extends MemoryReader and reads the lap time of the game.
    """

    def __init__(self):
        """
        Constructs an instance of LapTime by calling MemoryReader constructor.
        """
        super().__init__(par_return_value_type=ReturnValuesEnum.FLOAT,
                         par_offsets=[0x2E8, 0xC, 0xC, 0x18, 0x2C, 0x2F0, 0x48],
                         par_module_base_address=0x00520E6C)

    def return_lap_time(self) -> float:
        """
        Returns the lap time by calling the return_value_from_pointer_address
            method of MemoryReader.

        Returns
        -------
        float
            The lap progress percentage.
        """
        return super().return_value_from_pointer_address()
