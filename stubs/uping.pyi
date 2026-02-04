"""
Ping another computer.

MicroPython module: https://docs.micropython.org/en/preview/library/uping.html
"""

# source version: preview
# origin module:: repos/micropython/docs/library/uping.rst
from __future__ import annotations
from _typeshed import Incomplete

def ping(host, count=4, timeout=5000, interval=10, quiet=False, size=64) -> int:
    """
    Ping the ``host`` with ``count`` packets each having ``timeout`` at a
    rate between packets of ``interval``. If ``quiet`` is ``True`` do not print
    stats on return. Packet sizes are ``size`` bytes each.

    Returns a tuple containing the number of transmitted packets and the number received.
    """
    ...
