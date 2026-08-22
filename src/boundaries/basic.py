"""Boundary/loss models for a fixed-size marker ensemble."""

from __future__ import annotations

import jax.numpy as jnp

from core.math import momentum_from_gamma, safe_radius
from core.state import ParticleState


def radial_absorbing_wall(particles: ParticleState, r_wall: float = 1.0):
    r, _ = safe_radius(particles.kin.x, particles.kin.y)
    alive = particles.alive & (r < r_wall)
    weight = jnp.where(alive, particles.weight, 0.0)
    return ParticleState(particles.kin, weight, alive, particles.pid)


def gamma_absorbing_boundary(particles: ParticleState, gamma_min: float):
    alive = particles.alive & (particles.kin.gamma >= gamma_min)
    weight = jnp.where(alive, particles.weight, 0.0)
    return ParticleState(particles.kin, weight, alive, particles.pid)


def momentum_reservoir_boundary(
    particles: ParticleState,
    p_min: float,
    p_max: float,
):
    """Fixed-N momentum window for tail / runaway benchmarks.

    Particles that diffuse below ``p_min`` are returned to ``p_min`` while
    retaining their pitch.  At ``p_max`` a zero-flux reflecting boundary is
    applied.  The low-energy clamp is useful as a particle analogue of a
    thermal reservoir boundary when the unresolved bulk lies below the
    simulated momentum window.  Pitch scattering at the lower boundary
    rapidly isotropizes returning markers.

    This boundary changes marker coordinates but never particle weights or
    array shapes, so it is compatible with JIT compilation and sharding.
    """

    if p_max <= p_min:
        raise ValueError("p_max must exceed p_min")

    kin = particles.kin
    p = momentum_from_gamma(kin.gamma)
    p = jnp.maximum(p, p_min)

    # Reflect arbitrary high-side overshoots into [p_min, p_max] with a
    # triangle map.  Because p was clamped first, the low side is a reservoir
    # boundary rather than a reflector.
    width = p_max - p_min
    u = jnp.mod(p - p_min, 2.0 * width)
    p_bounded = p_min + jnp.where(u <= width, u, 2.0 * width - u)
    gamma = jnp.sqrt(1.0 + p_bounded * p_bounded)
    bounded_kin = kin._replace(gamma=gamma)
    return ParticleState(bounded_kin, particles.weight, particles.alive, particles.pid)
