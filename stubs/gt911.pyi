"""
Touch Screen Driver.

MicroPython module: https://docs.micropython.org/en/preview/library/gt911.html

Basic polling mode example usage::

    import time
    from gt911 import GT911
    from machine import I2C
    # Note use pin numbers or names not Pin objects because the
    # driver needs to change pin directions to reset the controller.
    touch = GT911(I2C(1, freq=400_000), reset_pin="P1", irq_pin="P2", touch_points=5)
    while True:
       n, points = touch.read_points()
       for i in range(0, n):
          print(f"id {points[i][3]} x {points[i][0]} y {points[i][1]} size {points[i][2]}")
       time.sleep_ms(100)
"""

# source version: preview
# origin module:: repos/micropython/docs/library/gt911.rst
from __future__ import annotations
from typing import IO, Any, Callable, Coroutine, Dict, Generator, Iterator, List, NoReturn, Optional, Tuple, Union, NamedTuple, TypeVar
from _typeshed import Incomplete
class GT911():
    """
       Creates a touch screen controller object. You should initialize it according to the example above.
    """
    def __init__(self, bus, reset_pin, irq_pin, address=0x5D, width=800, height=480, touch_points=1, reserve_x=False, reserve_y=False, reverse_axis=True, stio=True, refresh_rate=240, touch_callback=None) -> None:
        ...
    def _read_reg(self, reg, size=1, buf=None) -> Incomplete:
        """
           Reads a register value.
        """
        ...
    def _write_reg(self, reg, val, size=1: Optional[Any]=None) -> None:
        """
           Writes a register value.
        """
        ...
    def read_id(self) -> Incomplete:
        """
           Returns the ID of the gt911 chip.
        """
        ...
    def read_points(self) -> Tuple:
        """
           Returns a tuple containing the count of points an array of point tuples. Each point tuple has
           an x[0], y[1], size[2], and id[3]. x/y are the position on screen. Size is the amount of pressure
           applied. And id is a unique id per point which should correlate to the same point over reads.
        """
        ...
    def reset(self) -> None:
        """
           Resets the gt911 chip.
        """
        ...
