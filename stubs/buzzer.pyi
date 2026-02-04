"""
Buzzer driver.

MicroPython module: https://docs.micropython.org/en/preview/library/buzzer.html

The ``buzzer`` module is used to control the amplitude and frequency of a buzzer onboard your OpenMV Cam.

.. note::

   The buzzer (and this module) is not present on all OpenMV Cam models.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/buzzer.rst
from __future__ import annotations
from _typeshed import Incomplete

RESONANT_FREQ: Incomplete
"""Constant definting the highest volume frequency of the buzzer (typically 4000 Hz)."""

def freq(freq) -> None:
    """
    Sets the buzzer frequency independently of the volume.

    ``freq`` any frequency to drive the buzzer at.
    """
    ...

def duty(duty) -> None:
    """
    Sets the buzzer duty cycle independently of the frequency.

    ``duty`` any PWM duty cycle percentage (0-255 for 0-100%).
    """
    ...
