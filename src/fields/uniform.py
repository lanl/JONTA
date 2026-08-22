"""Uniform 0-D field model."""

from __future__ import annotations

from typing import NamedTuple


class UniformField(NamedTuple):
    """Uniform normalized fields for 0-D momentum-space calculations.

    ``e_parallel`` uses the RAMc sign convention. ``b`` is B/B0 and is only
    needed by models that explicitly depend on magnetic-field magnitude.
    """

    e_parallel: float
    b: float = 1.0
