"""Diagnostics and Guo-2017 asymptotic models for the runaway vortex.

The formulas in this module are diagnostic/reference relations rather than
production evolution equations.  They are used by the 0D kinetic validation
benchmark in ``tests/convergence/test_runaway_vortex.py``.

Reference
---------
Z. Guo, C. J. McDevitt, and X.-Z. Tang, Plasma Phys. Control. Fusion 59,
044003 (2017), especially Eqs. (22)--(25).
"""

from __future__ import annotations

import jax.numpy as jnp


def guo_o_point_momentum(e_over_ec, alpha, z):
    """Large-p Guo model for the runaway-vortex O-point momentum (Eq. 22)."""

    e = jnp.asarray(e_over_ec)
    a = jnp.asarray(alpha)
    z = jnp.asarray(z)
    return jnp.sqrt(2.0) * (e + a) * (e - 1.0) / ((1.0 + z) * a)


def guo_x_point_momentum(e_over_ec, z):
    """Large-E Guo model for the runaway-vortex X-point momentum (Eq. 23)."""

    e = jnp.asarray(e_over_ec)
    z = jnp.asarray(z)
    return jnp.sqrt((1.0 + (1.0 + z) / (2.0 * jnp.sqrt(2.0))) / e)


def guo_bump_momentum(e_over_ec, alpha, z):
    """Empirical pitch-integrated bump location from Guo et al. Eq. (24)."""

    return guo_o_point_momentum(e_over_ec, alpha, z) / 1.55


def guo_bump_width(e_over_ec, alpha, z):
    """Guo model for the runaway-tail spread, ``(p_O-p_X)/1.8``."""

    return (
        guo_o_point_momentum(e_over_ec, alpha, z)
        - guo_x_point_momentum(e_over_ec, z)
    ) / 1.8


def guo_acceleration_channel_width(p, e_over_ec, alpha):
    """Large-p pitch-angle acceleration-channel width from Guo Eq. (16)."""

    p = jnp.asarray(p)
    gamma = jnp.sqrt(1.0 + p * p)
    e = jnp.asarray(e_over_ec)
    a = jnp.asarray(alpha)
    return (e - 1.0 - 1.0 / (p * p)) / (2.0 * a * p * gamma + e)
