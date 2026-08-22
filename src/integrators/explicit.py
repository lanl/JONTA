"""Swappable fixed-step explicit integrators."""

from __future__ import annotations

import jax


def _axpy(y, a, k):
    return jax.tree.map(lambda yi, ki: yi + a * ki, y, k)


def euler_step(rhs, y, t, dt):
    return _axpy(y, dt, rhs(y, t))


def midpoint_step(rhs, y, t, dt):
    k1 = rhs(y, t)
    ymid = _axpy(y, 0.5 * dt, k1)
    k2 = rhs(ymid, t + 0.5 * dt)
    return _axpy(y, dt, k2)


def rk4_step(rhs, y, t, dt):
    k1 = rhs(y, t)
    k2 = rhs(_axpy(y, 0.5 * dt, k1), t + 0.5 * dt)
    k3 = rhs(_axpy(y, 0.5 * dt, k2), t + 0.5 * dt)
    k4 = rhs(_axpy(y, dt, k3), t + dt)
    return jax.tree.map(
        lambda yi, a, b, c, d: yi + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        y,
        k1,
        k2,
        k3,
        k4,
    )
