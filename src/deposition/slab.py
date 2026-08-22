"""Geometry-neutral scalar moments for uniform 0-D/slab models."""

from __future__ import annotations

import jax.numpy as jnp

from core.constants import ALFVEN_CURRENT_A, ME_C2_EV
from core.math import momentum_from_gamma
from core.state import ParticleState

_RAMC_EC_ESU = 4.8032e-9


def deposit_slab_parallel_current(
    particles: ParticleState,
    a_minor_cm: float,
    *,
    current_scale: float = 1.0,
    a_runaway: float = 0.0,
    te_ev=None,
):
    """Deposit uniform-volume scalar parallel current.

    This is a 0-D/slab moment, not a radial profile. ``current_scale`` carries
    model-specific volume/normalization factors explicitly; no circular-q or
    flux-surface Jacobian is assumed. Runaway selection requires ``te_ev`` when
    ``a_runaway > 0``.
    """

    if a_runaway > 0.0 and te_ev is None:
        raise ValueError("te_ev required for runaway-threshold selection")
    kin = particles.kin
    p = momentum_from_gamma(kin.gamma)
    v = p / jnp.maximum(kin.gamma, 1.0)
    selected = particles.alive & (particles.weight > 0.0)
    if a_runaway > 0.0:
        gamma_ra = 1.0 + a_runaway * jnp.asarray(te_ev) / ME_C2_EV
        selected = selected & (kin.gamma >= gamma_ra)
    weight = jnp.where(selected, particles.weight, 0.0)
    prefactor = -(_RAMC_EC_ESU / a_minor_cm) / ALFVEN_CURRENT_A
    return current_scale * jnp.sum(prefactor * v * kin.xi * weight)


def deposit_slab_weight(particles: ParticleState):
    """Deposit represented marker weight as scalar 0-D density proxy."""

    return jnp.sum(jnp.where(particles.alive, particles.weight, 0.0))
