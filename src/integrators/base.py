"""Integrator construction helpers.

Integrators are ordinary functions with signature ``step(rhs, y, t, dt)``.
This avoids runtime polymorphism in compiled kernels while keeping the
algorithm swappable at simulation construction time.
"""

from __future__ import annotations

from typing import Callable, Protocol, TypeVar

Y = TypeVar("Y")


class RHS(Protocol[Y]):
    def __call__(self, y: Y, t: float) -> Y: ...


Integrator = Callable[[RHS, Y, float, float], Y]
