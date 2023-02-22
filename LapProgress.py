from multiprocessing import Lock, Value
import pymem
from pymem import Pymem
from pymem.process import module_from_name


class LapProgress:
    mem: Pymem
    module: pymem.process
    shared_lap_progress: Value
    lock: Lock

    # offsets = [0x34, 0xC, 0x2C, 0x4, 0x9C, 0x38, 0x18]
    offsets = [0x3C]

    def __init__(self):
        self.lock = Lock()
        self.shared_lap_progress = Value('i', 0)
        pass

    def construct(self):
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, "speed.exe").lpBaseOfDll

    def return_lap_completed_percent(self):
        self.lock.acquire()
        self.shared_lap_progress = self.mem.read_float(self.get_pointer_address(self.module + 0x0052FDC4, self.offsets))
        self.lock.release()
        # return self.mem.read_float(self.get_pointer_address(self.module + 0x0052C514, self.offsets))
        return self.shared_lap_progress

    def get_pointer_address(self, base, offsets):
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr
