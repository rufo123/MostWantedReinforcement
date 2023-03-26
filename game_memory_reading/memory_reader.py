"""
Module for reading a game memory
"""
from multiprocessing import Lock, Value
from typing import Union

import pymem
from pymem import Pymem
from pymem.process import module_from_name

from game_memory_reading.return_values_enum import ReturnValuesEnum


# pylint: disable=too-many-instance-attributes
class MemoryReader:
    """
      A class for reading memory values from a game.
    """
    mem: Pymem
    module: pymem.process
    returned_value_from_pointer_address: Value
    lock: Lock

    # offsets = [0x34, 0xC, 0x2C, 0x4, 0x9C, 0x38, 0x18]
    offsets: list[int]
    module_base_address: int
    game_exec_name: str
    return_value_type: ReturnValuesEnum

    def __init__(self, par_return_value_type: ReturnValuesEnum, par_offsets: list[int],
                 par_module_base_address: int):
        """
        Initializes a new instance of the MemoryReader class.

        Args:
        - par_return_value_type: The type of the return value. An instance of ReturnValuesEnum.
        """
        self.lock = Lock()
        self.returned_value_from_pointer_address = Value('i', 0)
        self.offsets = par_offsets
        self.module_base_address = par_module_base_address
        self.game_exec_name = "speed.exe"
        self.return_value_type = par_return_value_type

    def construct(self):
        """
        Initializes the PyMem object and gets the base address of the game's module.
        """
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, self.game_exec_name).lpBaseOfDll

    def return_value_from_pointer_address(self) -> Union[int, float, bool]:
        """
        Returns the value read from a memory address that is calculated using the module's
            base address and offsets.

        Returns:
        - The value read from the memory address. The type of the value is determined
             by the return_value_type field.
        """

        if self.return_value_type == ReturnValuesEnum.FLOAT:
            self.returned_value_from_pointer_address = self.mem.read_float(
                self.get_pointer_address(self.module + self.module_base_address, self.offsets))
        elif self.return_value_type == ReturnValuesEnum.INT:
            self.returned_value_from_pointer_address = self.mem.read_int(
                self.get_pointer_address(self.module + self.module_base_address, self.offsets))
        else:
            self.returned_value_from_pointer_address = self.mem.read_bool(
                self.get_pointer_address(self.module + self.module_base_address, self.offsets))

        return self.returned_value_from_pointer_address

    def get_pointer_address(self, base: int, offsets: list[int]) -> int:
        """
        Calculates and returns the memory address from the given base address and offsets.

        Args:
        - base: The base address from which to start the calculation.
        - offsets: The list of offsets used to calculate the memory address.

        Returns:
        - The calculated memory address as an integer.
        """
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr
