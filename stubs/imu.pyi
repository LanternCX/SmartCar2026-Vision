"""
Imu sensor.

MicroPython module: https://docs.micropython.org/en/preview/library/imu.html

The ``imu`` module is used for reading the 6-DOF `LSM6DS3 <https://www.st.com/en/mems-and-sensors/lsm6ds3.html>`_
IMU sensor under the camera sensor.

.. note::

   The IMU sensor (and this module) is not present on all OpenMV Cam models.
"""

# source version: preview
# origin module:: repos/micropython/docs/library/imu.rst
from __future__ import annotations
from _typeshed import Incomplete

def acceleration_mg() -> float:
    """
    Returns the acceleration for (x, y, z) in a float tuple in milli-g's.

    For when the camera board is lying on a table face up:

    X points to the right of the camera sensor
    Y points down below the camera sensor (towards the bottom on the board)
    Z points in the reverse direction of the camera sensor (into the table)
    """
    ...

def angular_rate_mdps() -> float:
    """
    Returns the angular rate for (x, y, z) in a float tuple in milli-degrees-per-second.

    For when the camera board is lying on a table face up:

    X points to the right of the camera sensor
    Y points down below the camera sensor (towards the bottom on the board)
    Z points in the reverse direction of the camera sensor (into the table)
    """
    ...

def temperature_c() -> Incomplete:
    """
    Returns the temperature in celsius (float).
    """
    ...

def roll() -> Incomplete:
    """
    Returns the rotation angle in degrees (float) of the camera module.

       * 0 -> Camera is standing up.
       * 90 -> Camera is roated left.
       * 180 -> Camera is upside down.
       * 270 -> Camera is rotated right.
    """
    ...

def pitch() -> Incomplete:
    """
    Returns the rotation angle in degrees (float) of the camera module.

       * 0 -> Camera is standing up.
       * 90 -> Camera is pointing down.
       * 180 -> Camera is upside down.
       * 270 -> Camera is pointing up.
    """
    ...

def sleep(enable) -> Incomplete:
    """
    Pass ``True`` to put the IMU sensor to sleep. ``False`` to wake it back up (the default).
    """
    ...

def __write_reg(addr, val) -> Incomplete:
    """
    Set 8-bit LSM6DS3 register ``addr`` to 8-bit ``val``.
    """
    ...

def __read_reg(addr) -> Incomplete:
    """
    Get 8-bit LSM6DS3 register ``addr``.
    """
    ...
