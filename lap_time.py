"""" Module providing Multiprocessing support, imports Lock and Value"""
from multiprocessing import Lock, Value

import pymem
from pymem import Pymem
from pymem.process import module_from_name


class LapTime:
    """
    Class that reads lap time from game memory
    """
    mem: Pymem
    module: pymem.process
    shared_lap_time: Value
    lock: Lock

    offsets = [0x2E8, 0xC, 0xC, 0x18, 0x2C, 0x2F0, 0x48]

    def __init__(self):
        """
        Initialize LapTime class.

        Initializes a lock for shared memory access and a shared memory object for storing lap time.
        """
        self.lock = Lock()
        self.shared_lap_time = Value('i', 0)

    def construct(self):
        """
        Constructs a Pymem object for the game process and gets the base address of the game module.
        """
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, "speed.exe").lpBaseOfDll

    def return_lap_time(self) -> float:
        """
        Reads the lap time from game memory.

        Uses a lock for shared memory access and returns the lap time as a float.
        """
        self.lock.acquire()
        self.shared_lap_time = self.mem.read_float(
            self.get_pointer_address(self.module + 0x00520E6C, self.offsets))
        self.lock.release()
        return self.shared_lap_time

    def get_pointer_address(self, base: int, offsets: list[int]) -> int:
        """
        Calculates the pointer address for reading lap time from game memory.

        Uses the base address and a list of offsets to traverse the game's memory structure 
        and return the pointer address of the lap time value.
        """
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr
