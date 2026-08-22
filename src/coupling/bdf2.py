"""Generic first- and second-order implicit stepping helpers."""

from __future__ import annotations

import jax.numpy as jnp

from core.precision import real_dtype


def backward_euler_linear(A, y_n, source_np1, dt):
    """Solve y' = A y + source using backward Euler."""

    n = y_n.shape[-1]
    lhs = jnp.eye(n, dtype=real_dtype()) / dt - A
    rhs = y_n / dt + source_np1
    return jnp.linalg.solve(lhs, rhs)


def bdf2_linear(A, y_n, y_nm1, source_np1, dt):
    """Solve y' = A y + source using constant-step BDF2."""

    n = y_n.shape[-1]
    lhs = (3.0 / (2.0 * dt)) * jnp.eye(n, dtype=real_dtype()) - A
    rhs = (4.0 * y_n - y_nm1) / (2.0 * dt) + source_np1
    return jnp.linalg.solve(lhs, rhs)
