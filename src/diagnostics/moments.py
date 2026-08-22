"""Reduced particle diagnostics."""

from __future__ import annotations

import jax.numpy as jnp

from core.constants import ME_C2_EV
from core.math import momentum_from_gamma
from core.state import ParticleState


def total_weight(particles: ParticleState):
    return jnp.sum(jnp.where(particles.alive, particles.weight, 0.0))


def weighted_mean_gamma(particles: ParticleState):
    w = jnp.where(particles.alive, particles.weight, 0.0)
    return jnp.sum(w * particles.kin.gamma) / jnp.maximum(jnp.sum(w), 1.0e-300)


def total_kinetic_energy_ev(particles: ParticleState):
    w = jnp.where(particles.alive, particles.weight, 0.0)
    return jnp.sum(w * (particles.kin.gamma - 1.0) * ME_C2_EV)


def parallel_velocity_moment(particles: ParticleState):
    w = jnp.where(particles.alive, particles.weight, 0.0)
    p = momentum_from_gamma(particles.kin.gamma)
    vpar_over_c = p / jnp.maximum(particles.kin.gamma, 1.0) * particles.kin.xi
    return jnp.sum(w * vpar_over_c)
