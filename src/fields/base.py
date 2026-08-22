"""Field-model protocols.

The hot orbit kernels should consume arrays/NamedTuples, not Python virtual
methods. These protocols document construction-time contracts only.
"""

from __future__ import annotations

from typing import Protocol

from core.state import KinematicState


class FieldModel(Protocol):
    """Construction-time interface for a field representation."""

    def sample(self, kin: KinematicState, time: float):
        """Return local field quantities for all markers."""
        ...
