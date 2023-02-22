import time

import pymem.memory
import win32api
from pymem import *
from pymem.process import *

from multiprocessing import Lock, Value


class Speedometer:
    mem: Pymem
    module: pymem.process
    shared_speed: Value
    lock: Lock

    offsets = [0xC, 0xC, 0x38, 0xC, 0x54]

    def __init__(self):
        self.lock = Lock()
        self.shared_speed = Value('i', 0)

    def construct(self):
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, "speed.exe").lpBaseOfDll

    def return_speed_mph(self):
        with self.lock:
            result = self.mem.read_int(self.get_pointer_address(self.module + 0x00514824, self.offsets))
        return result

    def get_pointer_address(self, base, offsets):
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr
