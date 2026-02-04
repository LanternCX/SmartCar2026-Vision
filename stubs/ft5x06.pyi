"""
Touch Screen Driver.

MicroPython module: https://docs.micropython.org/en/preview/library/ft5x06.html

Touch Screen Driver for the OpenMV Pure Thermal.

.. note::

   This will be refactored to be under the display module soon.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/ft5x06.rst
from __future__ import annotations
from typing import IO, Any, Callable, Coroutine, Dict, Generator, Iterator, List, NoReturn, Optional, Tuple, Union, NamedTuple, TypeVar
from _typeshed import Incomplete
LCD_GESTURE_MOVE_UP: Incomplete
"""Touch screen move up gesture."""
LCD_GESTURE_MOVE_LEFT: Incomplete
"""Touch screen move left gesture."""
LCD_GESTURE_MOVE_DOWN: Incomplete
"""Touch screen move down gesture."""
LCD_GESTURE_MOVE_RIGHT: Incomplete
"""Touch screen move right gesture."""
LCD_GESTURE_ZOOM_IN: Incomplete
"""Touch screen zoom in gesture."""
LCD_GESTURE_ZOOM_OUT: Incomplete
"""Touch screen zoom out gesture."""
LCD_GESTURE_NONE: Incomplete
"""Touch screen no gesture."""
LCD_FLAG_PRESSED: Incomplete
"""Touch point is pressed."""
LCD_FLAG_RELEASED: Incomplete
"""Touch point is released."""
LCD_FLAG_MOVED: Incomplete
"""Touch point is moved."""
class FT5X06():
    """
       Creates a touch screen controller object
    """
    def __init__(self, i2c_addr=0x38: Optional[Any]=None) -> None:
        ...
    def get_gesture(self) -> Incomplete:
        """
           This is one of LCD_GESTURE_*.
        
           When a callback is enabled for the touch screen this method should not be called anymore except
           inside of the callback.
        """
        ...
    def get_points(self) -> int:
        """
           This returns the current number of touch points (0-5).
        
           When a callback is enabled for the touch screen this method should not be called anymore except
           inside of the callback.
        """
        ...
    def get_point_flag(self, index) -> int:
        """
           This returns the current touch point state of the point at ``index``.
        
           This is one of LCD_FLAG_*.
        
           When a callback is enabled for the touch screen this method should not be called anymore except
           inside of the callback.
        """
        ...
    def get_point_id(self, index) -> int:
        """
           This returns the current touch point ``id`` of the point at ``index``.
        
           The touch point ``id`` is a numeric value that allows you to track a touch point as it may move
           around in list of touch points returned as points are added and removed.
        
           When a callback is enabled for the touch screen this method should not be called anymore except
           inside of the callback.
        """
        ...
    def get_point_x(self, index) -> int:
        """
           This returns the current touch point x position of the point at ``index``.
        
           This is the x pixel position of the touch point on the screen.
        
           When a callback is enabled for the touch screen this method should not be called anymore except
           inside of the callback.
        """
        ...
    def get_point_y(self, index) -> int:
        """
           This returns the current touch point y position of the point at ``index``.
        
           This is the y pixel position of the touch point on the screen.
        
           When a callback is enabled for the touch screen this method should not be called anymore except
           inside of the callback.
        """
        ...
    def touch_callback(self, callback) -> Incomplete:
        """
           This method registers a callback which will receive the number of touch
           points (0-5) when a touch event happens.
        
           If you use this method do not call `FT5X06.update_points()` anymore until the callback is
           disabled by pass ``None`` as the callback for this method.
        """
        ...
    def update_points(self) -> int:
        """
           This function reads the touch screen state and returns the number of touch points (0-5).
        """
        ...
