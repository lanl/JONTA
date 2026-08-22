"""Collision operator contracts.

Operators are selected when a simulation kernel is constructed. Hot code
uses ordinary functions rather than runtime Python dispatch.
"""

from __future__ import annotations

from typing import Protocol

from core.state import ParticleState


class CollisionOperator(Protocol):
    def __call__(self, particles: ParticleState, *args, **kwargs) -> ParticleState: ...
