"""
Mutex module.

MicroPython module: https://docs.micropython.org/en/preview/library/mutex.html

The ``mutex`` module is used for creating mutexes.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/mutex.rst
from __future__ import annotations
from _typeshed import Incomplete

class Mutex:
    """
    Creates an unlocked mutex object.

    Methods
    ~~~~~~~
    """

    def __init__(self) -> None: ...
    def release(self) -> Incomplete:
        """
        Unlock the mutex.
        """
        ...

    def test(self) -> bool:
        """
        Try to acquire the mutex in a non-blocking way. Return ``True`` on success and ``False`` on failure.

        You may also acquire the mutex in a blocking way by using ``with``.
        """
        ...
