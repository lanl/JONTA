"""Geometry-neutral particle macrostep and moment-reduction contracts."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.state import ParticleState


class ParticleMoments(NamedTuple):
    """Reduced particle moments on geometry-specific grid/scalar layout."""

    parallel_current: jnp.ndarray
    weight_density: jnp.ndarray


class ParticleMacrostepResult(NamedTuple):
    """Particle state and moments after one fixed-field coupling interval."""

    particles: ParticleState
    moments: ParticleMoments
    max_large_angle_fraction: jnp.ndarray
    active_count: jnp.ndarray
    total_weight: jnp.ndarray


def reduce_particle_moments(particles: ParticleState, moment_depositor) -> ParticleMoments:
    """Apply geometry-specific depositor, reducing leading device axis.

    ``moment_depositor`` receives one flat ``ParticleState`` and returns
    ``ParticleMoments``. This keeps coupling independent of geometry: slab,
    circular RAMc, and future field models provide different depositors while
    serial and sharded execution share one reduction contract.
    """

    if particles.weight.ndim < 2:
        return moment_depositor(particles)
    local = jax.vmap(moment_depositor)(particles)
    return ParticleMoments(
        jnp.sum(local.parallel_current, axis=0),
        jnp.sum(local.weight_density, axis=0),
    )


def advance_particle_macrostep(
    particle_block,
    particles: ParticleState,
    field_state,
    background,
    time_n: float,
    particle_dt: float,
    base_key,
    global_step0: int,
    *,
    moment_depositor,
) -> ParticleMacrostepResult:
    """Push fixed fields, then deposit/reduce geometry-specific moments.

    ``particle_block`` owns static particle-step count and timestep hierarchy.
    ``moment_depositor`` must accept one flat particle pool and return
    ``ParticleMoments``. Geometry-specific normalization belongs in depositor,
    never in this shared path.
    """

    particles_np1, max_q = particle_block(
        particles,
        field_state,
        background,
        time_n,
        particle_dt,
        base_key,
        global_step0,
    )
    moments = reduce_particle_moments(particles_np1, moment_depositor)
    return ParticleMacrostepResult(
        particles_np1,
        moments,
        max_q,
        jnp.sum(particles_np1.alive),
        jnp.sum(particles_np1.weight),
    )
