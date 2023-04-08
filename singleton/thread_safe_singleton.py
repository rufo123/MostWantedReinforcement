"""
This module provides a thread-safe singleton metaclass.
"""
from multiprocessing import Lock


class ThreadSafeSingleton(type):
    """
    A thread-safe singleton metaclass.
    """
    _instances = {}
    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        """
        Return the instance of the class if it exists, otherwise create a new one.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            An instance of the class.

        """
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]
