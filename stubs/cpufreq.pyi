"""
CPU Frequency Control.

MicroPython module: https://docs.micropython.org/en/preview/library/cpufreq.html

The ``cpufreq`` module is used to get/set the CPU frequency to save power.

.. note::

   This module is not supported on the OpenMV Cam M4 because the CPU frequency,
   for various reasons, cannot be set independently of peripherals.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/cpufreq.rst
from __future__ import annotations
from _typeshed import Incomplete

def set_frequency(supported_frequency) -> None:
    """
    Sets the CPU frequency to a supported frequency in MHz. Peripherals
    frequencies are not changed. Only the CPU performance.
    """
    ...

def get_current_frequencies() -> Incomplete:
    """
    Returns (cpu_clk_in_mhz, hclk_in_mhz, pclk1_in_mhz, pclk2_in_mhz).
    """
    ...

def get_supported_frequencies() -> Incomplete:
    """
    Returns the supported CPU frequencies [120, 144, 168, 192, 216] on the
    OpenMV Cam M7 and [60/50, 120/100, 240/200, 480/400] on the OpenMV Cam H7 Rev V/XY silicon in MHz.
    """
    ...
