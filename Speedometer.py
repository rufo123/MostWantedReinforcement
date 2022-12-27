import pymem.memory
from pymem import *
from pymem.process import *

class Speedometer:

    mem: Pymem
    module: pymem.process

    offsets = [0xC, 0xC, 0x38, 0xC, 0x54]

    def __init__(self):
        self.mem = Pymem("speed.exe")
        self.module = module_from_name(self.mem.process_handle, "speed.exe").lpBaseOfDll
        pass

    def return_speed_mph(self):
        return self.mem.read_int(self.get_pointer_address(self.module + 0x00514824, self.offsets))

    def get_pointer_address(self, base, offsets):
        addr = self.mem.read_int(base)
        for offset in offsets:
            if offset != offsets[-1]:
                addr = self.mem.read_int(addr + offset)
        addr = addr + offsets[-1]
        return addr



