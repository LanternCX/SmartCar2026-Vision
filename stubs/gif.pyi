"""
Gif recording.

MicroPython module: https://docs.micropython.org/en/preview/library/gif.html

The ``gif`` module is used for gif recording.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/gif.rst
from __future__ import annotations
from typing import IO, Any, Callable, Coroutine, Dict, Generator, Iterator, List, NoReturn, Optional, Tuple, Union, NamedTuple, TypeVar
from _typeshed import Incomplete
class Gif():
    """
       Create a Gif object which you can add frames to. ``filename`` is the path to
       save the gif recording to.
    
       ``width`` is automatically set equal to the image sensor horizontal resolution
       unless explicitly overridden.
    
       ``height`` is automatically set equal to the image sensor vertical resolution
       unless explicitly overridden.
    
       ``color`` is automatically set equal to the image sensor color mode
       unless explicitly overridden:
    
         - False for color results in a `sensor.GRAYSCALE` 7-bit per pixel gif.
         - True for color results in a `sensor.RGB565` 7-bit per pixel gif.
    
       ``loop`` when True results in the gif automatically looping on playback.
    
       Methods
       ~~~~~~~
    """
    def __init__(self, filename, width, height, color, loop=True) -> None:
        ...
    def width(self) -> gif:
        """
              Returns the width (horizontal resolution) for the gif object.
        """
        ...
    def height(self) -> gif:
        """
              Returns the height (vertical resolution) for the gif object.
        """
        ...
    def format(self) -> Incomplete:
        """
              Returns `sensor.RGB565` if color is True or `sensor.GRAYSCALE` if not.
        """
        ...
    def size(self) -> int:
        """
              Returns the file size of the gif so far. This value is updated after adding frames.
        """
        ...
    def loop(self) -> gif:
        """
              Returns if the gif object had loop set in its constructor.
        """
        ...
    def add_frame(self, image, delay=10: Optional[Any]=None) -> Incomplete:
        """
              Add an image to the gif recording. The image width, height, and color mode,
              must be equal to the same width, height, and color modes used in the constructor
              for the gif.
        
              ``delay`` is the number of centi-seconds to wait before displaying this frame
              after the previous frame (if not the first frame).
        """
        ...
    def close(self) -> Incomplete:
        """
              Finalizes the gif recording. This method must be called once the recording
              is complete to make the file viewable.
        """
        ...
