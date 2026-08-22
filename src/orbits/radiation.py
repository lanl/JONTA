"""Radiation-reaction contributions to deterministic characteristics."""

from __future__ import annotations

import jax.numpy as jnp

from core.math import momentum_from_gamma


def synchrotron_low_beta(gamma, xi, alpha_syn, b_over_b0=1.0):
    """RAMc low-beta synchrotron terms in (gamma, xi).

    RAMc Eqs. (50)-(51) are written for momentum magnitude p. Since
    dgamma/dt = (p/gamma) dp/dt, the gamma contribution is

        dgamma/dt = -alpha * p^2 * (1-xi^2)

    with ``alpha`` multiplied by (B/B0)^2 when the local field varies.
    """

    p = momentum_from_gamma(gamma)
    alpha_local = alpha_syn * b_over_b0 * b_over_b0
    dgamma = -alpha_local * p * p * (1.0 - xi * xi)
    dxi = alpha_local * xi * (1.0 - xi * xi) / jnp.maximum(gamma, 1.0)
    return dgamma, dxi
