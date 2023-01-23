from multiprocessing import Lock, Value

import pymem
from pymem import Pymem
from pymem.process import module_from_name


class LapTime:
    mem: Pymem
    module: pymem.process
    shared_lap_time: Value
    lock: Lock

    offsets = [0x2E8, 0xC, 0xC, 0x18, 0x2C, 0x2F0, 0x48]

    def __init__(self):
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, "speed.exe").lpBaseOfDll
        self.lock = Lock()
        self.shared_lap_time = Value('i', 0)

    def return_lap_time(self):
        self.lock.acquire()
        self.shared_lap_time = self.mem.read_float(self.get_pointer_address(self.module + 0x00520E6C, self.offsets))
        self.lock.release()
        return self.shared_lap_time

    def get_pointer_address(self, base, offsets):
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr
