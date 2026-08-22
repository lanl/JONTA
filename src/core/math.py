"""Numerically safe scalar/vector helpers used in hot kernels."""

from __future__ import annotations

import jax.numpy as jnp


def momentum_from_gamma(gamma):
    """Return p/(m_e c) from gamma with a roundoff-safe floor."""

    return jnp.sqrt(jnp.maximum(gamma * gamma - 1.0, 0.0))


def speed_from_gamma(gamma):
    p = momentum_from_gamma(gamma)
    return p / jnp.maximum(gamma, 1.0)


def reflect_pitch(xi):
    """Reflect arbitrary values into [-1, 1] using a period-4 triangle map."""

    t = jnp.mod(xi + 1.0, 4.0)
    return jnp.where(t <= 2.0, t - 1.0, 3.0 - t)


def reflect_gamma(gamma, eps=1.0e-14):
    """Reflect through gamma=1, matching RAMc's low-energy Neumann boundary."""

    return jnp.where(gamma < 1.0, 2.0 - gamma + eps, gamma)


def safe_radius(x, y, floor=1.0e-12):
    r = jnp.sqrt(x * x + y * y)
    return r, jnp.maximum(r, floor)


def finite_difference_profile(y, x):
    """Second-order finite-difference derivative on a monotone 1-D grid."""

    dx_l = x[1:-1] - x[:-2]
    dx_r = x[2:] - x[1:-1]
    # Weighted second-order derivative for nonuniform spacing.
    slope_l = (y[1:-1] - y[:-2]) / dx_l
    slope_r = (y[2:] - y[1:-1]) / dx_r
    interior = (dx_r * slope_l + dx_l * slope_r) / (dx_l + dx_r)
    left = (y[1] - y[0]) / (x[1] - x[0])
    right = (y[-1] - y[-2]) / (x[-1] - x[-2])
    return jnp.concatenate([left[None], interior, right[None]])
