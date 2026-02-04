"""
DVI/HDMI Controller.

MicroPython module: https://docs.micropython.org/en/preview/library/TFP410.html

DVI/HDMI Controller for the OpenMV Pure Thermal.

.. note::

   This will be refactored to be under the display module soon.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/tfp410.rst
from __future__ import annotations
from typing import IO, Any, Callable, Coroutine, Dict, Generator, Iterator, List, NoReturn, Optional, Tuple, Union, NamedTuple, TypeVar
from _typeshed import Incomplete
class tfp410.TFP410():
    """
       Initializes the TFP410 DVI/HDMI controller chip to drive an external DVI/HDMI display via
       a 24-bit parallel LCD bus. You just need to create this object to initialize the display.
    """
    def __init__(self, i2c_addr=0x3F: Optional[Any]=None) -> None:
        ...
class TFP410():
    """ """
    def isconnected(self) -> Incomplete:
        """
           Returns if an external display is connected.
        """
        ...
    def hotplug_callback(self, callback) -> Incomplete:
        """
           Registers a ``callback`` function that be called whenever the state
           of an external display being connected changes. The new state will be passed as an argument.
        
           If you use this method do not call `TFP410.isconnected()` anymore until the callback is
           disabled by pass ``None`` as the callback for this method.
        """
        ...
