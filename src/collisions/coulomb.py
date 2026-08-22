"""Coulomb helper functions used by collision models."""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import erf

from core.constants import ME_C2_EV, PI


def thermal_coulomb_log(ne_cm3, te_ev):
    """Thermal Coulomb logarithm used in the supplied RAMc implementation."""

    ne = jnp.maximum(ne_cm3, 1.0)
    te = jnp.maximum(te_ev, 1.0e-12)
    return 14.9 - 0.5 * jnp.log(ne * 1.0e-14) + jnp.log(te * 1.0e-3)


def chandrasekhar(x):
    """Chandrasekhar function Psi(x), with a small-x expansion."""

    ax = jnp.abs(x)
    xs = jnp.maximum(ax, 1.0e-12)
    direct = 0.5 / (xs * xs) * (
        erf(xs) - (2.0 / jnp.sqrt(PI)) * xs * jnp.exp(-xs * xs)
    )
    series = (2.0 / (3.0 * jnp.sqrt(PI))) * xs - (
        2.0 / (5.0 * jnp.sqrt(PI))
    ) * xs**3
    return jnp.where(ax < 1.0e-3, series, direct)


def d_chandrasekhar_dx(x):
    ax = jnp.abs(x)
    xs = jnp.maximum(ax, 1.0e-12)
    psi = chandrasekhar(xs)
    direct = (2.0 / jnp.sqrt(PI)) * jnp.exp(-xs * xs) - 2.0 * psi / xs
    series = (2.0 / (3.0 * jnp.sqrt(PI))) - (
        6.0 / (5.0 * jnp.sqrt(PI))
    ) * xs**2
    return jnp.where(ax < 1.0e-3, series, direct)


def thermal_speed_over_c(te_ev):
    return jnp.sqrt(jnp.maximum(2.0 * te_ev / ME_C2_EV, 1.0e-30))
