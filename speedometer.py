"""" Module providing Multiprocessing support, imports Lock and Value"""
from multiprocessing import Lock, Value

import pymem
from pymem import Pymem
from pymem.process import module_from_name


class Speedometer:
    """
    A class to represent the speedometer of the game.

    Attributes:
    -----------
    mem : Pymem
        An instance of the Pymem class to interact with the game's memory.
    module : 'pymem.process'
        The base module of the game.
    shared_speed : multiprocessing.Value
        A shared memory value for interprocess communication between threads.
    lock : multiprocessing.Lock
        A lock for synchronizing access to shared_speed.
    offsets : List[int]
        The offsets required to find the speed in the game's memory.
    """
    mem: Pymem
    module: pymem.process
    shared_speed: Value
    lock: Lock

    offsets = [0xC, 0xC, 0x38, 0xC, 0x54]

    def __init__(self):
        self.lock = Lock()
        self.shared_speed = Value('i', 0)

    def construct(self):
        """
        Initializes the Pymem instance and sets the module to the base module
        of the game.
        """
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, "speed.exe").lpBaseOfDll

    def return_speed_mph(self) -> int:
        """
        Returns the speed of the car in miles per hour.

        Returns:
        --------
        int
            The speed of the car in miles per hour.
        """
        with self.lock:
            result = self.mem.read_int(
                self.get_pointer_address(self.module + 0x00514824, self.offsets))
        return result

    def get_pointer_address(self, base: int, offsets: list[int]) -> int:
        """
        Returns the memory address of the desired value.

        Parameters:
        -----------
        base : int
            The base address to start the search from.
        offsets : List[int]
            The list of offsets to follow to reach the desired value.

        Returns:
        --------
        int
            The memory address of the desired value.
        """
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr
