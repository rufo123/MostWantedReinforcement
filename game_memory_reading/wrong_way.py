"""
This module provides a `WrongWay` class that reads memory from a game and 
    returns boolean if car is going in a wrong way.
It inherits from the `MemoryReader` class and uses the `ReturnValuesEnum` enumeration for
    the type of value to be returned.
"""

from game_memory_reading.memory_reader import MemoryReader
from game_memory_reading.return_values_enum import ReturnValuesEnum


class WrongWay(MemoryReader):
    """
    A class that extends MemoryReader and reads the wrong way boolean of the car from 
        the game.
    """

    def __init__(self):
        """
        Constructs an instance of WrongWay by calling MemoryReader constructor.
        """
        super().__init__(par_return_value_type=ReturnValuesEnum.BOOL,
                         par_offsets=[0xF8, 0x84, 0x4, 0x14, 0x79C],
                         par_module_base_address=0x0052C514)

    def return_is_wrong_way(self) -> bool:
        """
        Returns the wrong way boolean of the car by calling the 
            return_value_from_pointer_address method of MemoryReader.

        Returns
        -------
        bool
            Wrong way of the car.
        """
        return super().return_value_from_pointer_address()
