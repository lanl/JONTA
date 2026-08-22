"""Process-wide numerical precision used by JONTA kernels.

Precision is a static build option: configure it before creating JAX arrays or
building JIT kernels.  FP64 remains the reference mode; FP32 is intended for
experiments and backends with no FP64 support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TypeAlias

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class PrecisionConfig:
    """Static floating-point configuration for one JAX process."""

    bits: int = 64

    def __post_init__(self):
        if self.bits not in (32, 64):
            raise ValueError("precision bits must be 32 or 64")

    @property
    def real_dtype(self):
        return jnp.float32 if self.bits == 32 else jnp.float64

    @property
    def index_dtype(self):
        # Particle counts and scan indices do not need 64-bit range.  Keeping
        # indices 32-bit in FP32 mode avoids another unsupported Metal dtype.
        return jnp.int32 if self.bits == 32 else jnp.int64

    @property
    def x64_enabled(self) -> bool:
        return self.bits == 64


PrecisionValue: TypeAlias = PrecisionConfig | int | str | None


_ACTIVE: PrecisionConfig | None = None


def _coerce(value: PrecisionValue) -> PrecisionConfig:
    if value is None:
        value = os.environ.get("JONTA_PRECISION", "64")
    if isinstance(value, PrecisionConfig):
        return value
    if isinstance(value, str):
        value = value.lower().removeprefix("float")
    try:
        return PrecisionConfig(int(value))
    except (TypeError, ValueError) as exc:
        raise ValueError("precision must be 32, 64, 'float32', or 'float64'") from exc


def configure_precision(value: PrecisionValue = None) -> PrecisionConfig:
    """Select process precision before array creation and JIT compilation."""

    global _ACTIVE
    if value is None and _ACTIVE is not None:
        return _ACTIVE
    if value is None:
        value = os.environ.get("JONTA_PRECISION", "64")
    selected = _coerce(value)
    jax.config.update("jax_enable_x64", selected.x64_enabled)
    _ACTIVE = selected
    return selected


def current_precision() -> PrecisionConfig:
    if _ACTIVE is None:
        return configure_precision()
    return _ACTIVE


def real_dtype():
    """Return active real dtype; call while constructing a kernel."""

    return current_precision().real_dtype


def index_dtype():
    """Return active integer index dtype."""

    return current_precision().index_dtype


# Configure from JONTA_PRECISION before core imports create any arrays.
configure_precision()
