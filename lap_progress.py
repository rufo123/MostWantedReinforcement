"""" Module providing Multiprocessing support, imports Lock and Value"""
from multiprocessing import Lock, Value

import pymem
from pymem import Pymem
from pymem.process import module_from_name


class LapProgress:
    """
       A class to retrieve lap progress information from a game's memory using Pymem library.

       Attributes:
       -----------
       mem : pymem.Pymem
           An instance of the Pymem library for accessing memory.
       module : 'pymem.process'
           The module name to retrieve lap progress information from.
       shared_lap_progress : multiprocessing.Value
           A shared value between processes to store the lap progress percentage.
       lock : multiprocessing.Lock
           A lock object to ensure thread-safety while accessing shared_lap_progress.
    """
    mem: Pymem
    module: pymem.process
    shared_lap_progress: Value
    lock: Lock

    # offsets = [0x34, 0xC, 0x2C, 0x4, 0x9C, 0x38, 0x18]
    offsets = [0x3C]

    def __init__(self):
        self.lock = Lock()
        self.shared_lap_progress = Value('i', 0)

    def construct(self):
        """
        Initializes the Pymem instance and sets the module name.
        """
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, "speed.exe").lpBaseOfDll

    def return_lap_completed_percent(self) -> float:
        """
        Retrieves the lap progress percentage from memory using shared_lap_progress.
        """
        self.lock.acquire()
        self.shared_lap_progress = self.mem.read_float(
            self.get_pointer_address(self.module + 0x0052FDC4, self.offsets))
        self.lock.release()
        return self.shared_lap_progress

    def get_pointer_address(self, base: int, offsets: list[int]) -> int:
        """
        Returns the address of a memory pointer using a base address and a list of offsets.
        """
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr
