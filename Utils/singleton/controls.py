import ctypes
import threading
import time
import torch.multiprocessing

from torch.multiprocessing import Lock
# Direct Input Scan Codes:8
# https://gist.github.com/tracend/912308
from singleton.ThreadSafeSingleton import ThreadSafeSingleton


class Controls(metaclass=ThreadSafeSingleton):

    UP_KEY: int = 0xC8
    LEFT_KEY: int = 0XCB
    RIGHT_KEY: int = 0XCD
    DOWN_KEY: int = 0xD0
    HAND_BRAKE: int = 0x39

    ENTER: int = 0x1C
    ESCAPE: int = 0x01

    a_lock: torch.multiprocessing.Lock()
    a_is_executing_critical_action: bool

    def __init__(self,):
        self.a_lock = Lock()
        a_is_executing_critical_action = False

    def PressKey(self, par_hex_key_code: int) -> None:
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, par_hex_key_code, 0x0008, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def ReleaseKey(self, par_hex_key_code: int) -> None:
        extra = ctypes.c_ulong(0)
        ii_ = Input_I()
        ii_.ki = KeyBdInput(0, par_hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

    def PressAndReleaseKey(self, par_hex_key_code: int, par_sleep_time: float = 1, par_can_bypass_critical_check: bool = False) -> None:
        self.CheckCriticalAction(par_can_bypass_critical_check)
        with self.a_lock:
            self.ReleaseAllKeys()
            self.PressKey(par_hex_key_code)
            time.sleep(par_sleep_time)
            self.ReleaseKey(par_hex_key_code)
            #time.sleep(par_sleep_time)

    def PressAndReleaseTwoKeys(self, par_hex_key_code_first: int, par_hex_key_code_second: int, par_sleep_time: float = 1, par_can_bypass_critical_check: bool = False):
        self.CheckCriticalAction(par_can_bypass_critical_check)
        with self.a_lock:
            self.ReleaseAllKeys()
            self.PressKey(par_hex_key_code_first)
            self.PressKey(par_hex_key_code_second)
            time.sleep(par_sleep_time / 2)
            self.ReleaseKey(par_hex_key_code_first)
            self.ReleaseKey(par_hex_key_code_second)
            #time.sleep(par_sleep_time/ 2)

    def CheckCriticalAction(self, par_can_bypass: False):
        while (not par_can_bypass) and self.a_is_executing_critical_action:
            print("Critical Action Is Being Executed (Waiting)")
            time.sleep(1)
            raise Exception("Critical Action Exception (Debug Delete Later)")

    def CheckKeyPressed(self, par_hex_key_code: int) -> bool:
        return bool(ctypes.windll.user32.GetKeyState(par_hex_key_code) & 0x8000)

    def ReleaseAllKeys(self):
        keys_to_release = [self.UP_KEY, self.RIGHT_KEY, self.DOWN_KEY, self.LEFT_KEY, self.HAND_BRAKE, self.ENTER,
                           self.ESCAPE]
        for key in keys_to_release:
            if self.CheckKeyPressed(key):
                self.ReleaseKey(key)

    def Forward(self, par_sleep_time: float = 1):
        self.PressAndReleaseKey(self.UP_KEY, par_sleep_time)

    def Backward(self, par_sleep_time: float = 1):
        self.PressAndReleaseKey(self.DOWN_KEY, par_sleep_time)

    def Left(self, par_sleep_time: float = 1):
        self.PressAndReleaseKey(self.LEFT_KEY, par_sleep_time)

    def Right(self, par_sleep_time: float = 1):
        self.PressAndReleaseKey(self.RIGHT_KEY, par_sleep_time)

    def ForwardRight(self, par_sleep_time: float = 1):
        self.PressAndReleaseTwoKeys(self.UP_KEY, self.RIGHT_KEY, par_sleep_time)

    def ForwardLeft(self, par_sleep_time: float = 1):
        self.PressAndReleaseTwoKeys(self.UP_KEY, self.LEFT_KEY, par_sleep_time)

    def BackwardLeft(self, par_sleep_time: float = 1):
        self.PressAndReleaseTwoKeys(self.DOWN_KEY, self.LEFT_KEY, par_sleep_time)

    def BackwardRight(self, par_sleep_time: float = 1):
        self.PressAndReleaseTwoKeys(self.DOWN_KEY, self.RIGHT_KEY, par_sleep_time)

    def HandBrake(self, par_sleep_time: float = 1):
        self.PressAndReleaseKey(self.HAND_BRAKE, par_sleep_time)

PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]