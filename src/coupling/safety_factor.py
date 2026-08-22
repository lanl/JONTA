"""Circular-RAMc safety-factor update from the total parallel current."""

from __future__ import annotations

import jax.numpy as jnp

from core.constants import PI
from core.precision import real_dtype


def update_circular_q(
    r,
    e1,
    eta_bar,
    j_kin,
    epsilon: float,
    a_omega_ce_over_c: float,
    j_bootstrap=None,
):
    """Update q(r) using the simplified RAMc Ampere-law integral.

    This ports the active branch of ``ComputeSafetyFactor`` in the supplied
    RAMc source:

        Btheta integral = int_0^r r' [E1/eta + j_kin + j_boot] dr'
        q(r) = r^2 epsilon / [(4 pi / wce) * integral].
    """

    if j_bootstrap is None:
        j_bootstrap = jnp.zeros_like(r)
    integrand = r * (e1 / jnp.maximum(eta_bar, 1.0e-30) + j_kin + j_bootstrap)
    dr = r[1:] - r[:-1]
    cells = 0.5 * (integrand[1:] + integrand[:-1]) * dr
    integral = jnp.concatenate([jnp.zeros((1,), dtype=real_dtype()), jnp.cumsum(cells)])
    denom = (4.0 * PI / a_omega_ce_over_c) * integral
    q = jnp.where(r > 0.0, r * r * epsilon / jnp.maximum(denom, 1.0e-30), 0.0)
    q = q.at[0].set(q[1])
    return q
