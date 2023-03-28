"""
Module containing Controls class implementing ThreadSafeSingleton

The Class Integrates Game Controls via DirectInput
Direct Input HexCodes taken From
https://gist.github.com/tracend/912308
"""
import ctypes
import time

import torch.multiprocessing
from torch.multiprocessing import Lock

# Direct Input Scan Codes:8
# https://gist.github.com/tracend/912308
from singleton.thread_safe_singleton import ThreadSafeSingleton


class Controls(metaclass=ThreadSafeSingleton):
    """
    Class that Integrates Game Controls via DirectInput
    Direct Input HexCodes taken From
    https://gist.github.com/tracend/912308
    """
    W_KEY: int = 0x11
    S_KEY: int = 0x1F
    A_KEY: int = 0x1E
    D_KEY: int = 0x20

    UP_KEY: int = 0xC8
    LEFT_KEY: int = 0XCB
    RIGHT_KEY: int = 0XCD
    DOWN_KEY: int = 0xD0
    HAND_BRAKE: int = 0x39

    ENTER: int = 0x1C
    ESCAPE: int = 0x01

    a_lock: torch.multiprocessing.Lock()
    a_is_executing_critical_action: bool

    def __init__(self, ):
        self.a_lock = Lock()

    def press_key(self, par_hex_key_code: int) -> None:
        """
        Presses key via Direct Input
        :param par_hex_key_code: Hexadecimal code of key
        """
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        # pylint: disable=attribute-defined-outside-init, invalid-name
        ii_.ki = KeyBdInput(0, par_hex_key_code, 0x0008, 0, ctypes.pointer(extra))
        tmp_x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(tmp_x), ctypes.sizeof(tmp_x))

    def release_key(self, par_hex_key_code: int) -> None:
        """
        Releases key via Direct Input
        :param par_hex_key_code: Hexadecimal code of key
        """
        extra = ctypes.c_ulong(0)
        ii_ = InputI()
        # pylint: disable=attribute-defined-outside-init, invalid-name
        ii_.ki = KeyBdInput(0, par_hex_key_code, 0x0008 | 0x0002, 0, ctypes.pointer(extra))
        tmp_x = Input(ctypes.c_ulong(1), ii_)
        ctypes.windll.user32.SendInput(1, ctypes.pointer(tmp_x), ctypes.sizeof(tmp_x))

    def press_and_release_key(self, par_hex_key_code: int, par_sleep_time: float = 1,
                              par_can_bypass_critical_check: bool = False) -> None:
        """
        Presses and Releases Key Via Direct Input
        Waits if Critical Action Execution is set to True
        Locks if same function is already being executed
        :param par_hex_key_code: Hexadecimal code of key
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        :param par_can_bypass_critical_check: Skips Waiting for Critical Action Execution 
        to be set to False
        """
        self.check_critical_action(par_can_bypass_critical_check)
        with self.a_lock:
            self.release_all_keys()
            self.press_key(par_hex_key_code)
            time.sleep(par_sleep_time)
            self.release_key(par_hex_key_code)
            # time.sleep(par_sleep_time)

    def press_and_release_two_keys(self, par_hex_key_code_first: int, par_hex_key_code_second: int,
                                   par_sleep_time: float = 1,
                                   par_can_bypass_critical_check: bool = False):
        """
          Presses and Releases Key Via Direct Input
          Waits if Critical Action Execution is set to True
          Locks if same function is already being executed
          :param par_hex_key_code_first: Hexadecimal code of First key
          :param par_hex_key_code_second: Hexadecimal code of Second key
          :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
          :param par_can_bypass_critical_check: Skips Waiting for Critical Action Execution to be
           set to False
          """
        self.check_critical_action(par_can_bypass_critical_check)
        with self.a_lock:
            self.release_all_keys()
            self.press_key(par_hex_key_code_first)
            self.press_key(par_hex_key_code_second)
            time.sleep(par_sleep_time / 2)
            self.release_key(par_hex_key_code_first)
            self.release_key(par_hex_key_code_second)
            # time.sleep(par_sleep_time/ 2)

    def check_critical_action(self, par_can_bypass: False):
        """
        Check if Critical Action Execution is set to True
        If True, wait one second
        :param par_can_bypass: If set to True, Skips Waiting for Critical Action Execution
        """
        while (not par_can_bypass) and self.a_is_executing_critical_action:
            print("Critical Action Is Being Executed (Waiting)")
            time.sleep(1)

    def check_key_pressed(self, par_hex_key_code: int) -> bool:
        """
        Check if Key is already pressed via Direct Input
        :param par_hex_key_code: Hexadecimal code of Key to Check
        :return: Returns True if Key is Already Pressed otherwise False
        """
        return bool(ctypes.windll.user32.GetKeyState(par_hex_key_code) & 0x8000)

    def release_all_keys(self) -> None:
        """
        Releases All Specified Keys if They are Still Pressed
        Keys: [UP_KEY, RIGHT_KEY, DOWN_KEY, LEFT_KEY, HAND_BRAKE, ENTER, ESCAPE]
        """
        keys_to_release = [self.W_KEY, self.D_KEY, self.S_KEY, self.A_KEY,
                           self.HAND_BRAKE, self.ENTER,
                           self.ESCAPE]
        for key in keys_to_release:
            if self.check_key_pressed(key):
                self.release_key(key)

    def forward(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Forward
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_key(self.W_KEY, par_sleep_time)

    def backward(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Backward
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_key(self.S_KEY, par_sleep_time)

    def left(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Left
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_key(self.A_KEY, par_sleep_time)

    def right(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Right
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_key(self.D_KEY, par_sleep_time)

    def forward_right(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Forward and Right
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_two_keys(self.W_KEY, self.D_KEY, par_sleep_time)

    def forward_left(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Forward and Left
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_two_keys(self.W_KEY, self.A_KEY, par_sleep_time)

    def backward_left(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Backward and Left
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_two_keys(self.S_KEY, self.A_KEY, par_sleep_time)

    def backward_right(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to Go Backward and Right
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_two_keys(self.S_KEY, self.D_KEY, par_sleep_time)

    def hand_brake(self, par_sleep_time: float = 1) -> None:
        """
        Execute Action to HandBrake
        :param par_sleep_time: Sleep Time Before Pressing and Releasing Key in Seconds
        """
        self.press_and_release_key(self.HAND_BRAKE, par_sleep_time)


PUL = ctypes.POINTER(ctypes.c_ulong)


class KeyBdInput(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Class Implementing KeyBdInput
    """
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class HardwareInput(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Class Implementing HardwareInput
    """
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]


class MouseInput(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Class Implementing MouseInput
    """
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]


class InputI(ctypes.Union):  # pylint: disable=too-few-public-methods
    """
    Class Implementing Input_I
    """
    _fields_ = [("ki", KeyBdInput),
                ("mi", MouseInput),
                ("hi", HardwareInput)]


class Input(ctypes.Structure):  # pylint: disable=too-few-public-methods
    """
    Class Implementing Input
    """
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", InputI)]
