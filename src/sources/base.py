"""Fixed-N source injection helpers."""

from __future__ import annotations

import jax.numpy as jnp

from core.precision import index_dtype, real_dtype
from core.rng import derive_particle_ids
from core.state import KinematicState, ParticleState
from population.capacity import capacity_control


def append_source_candidates(
    particles: ParticleState,
    source_kin: KinematicState,
    total_source_weight,
    *,
    global_step: int = 0,
):
    """Combine existing markers with equally weighted source candidates."""

    ns = source_kin.gamma.shape[0]
    ws = jnp.full((ns,), total_source_weight / ns, dtype=real_dtype())
    source = ParticleState(
        source_kin,
        ws,
        jnp.ones((ns,), dtype=jnp.bool_),
        jnp.arange(ns, dtype=index_dtype()),
    )

    def cat(a, b):
        return jnp.concatenate([a, b], axis=0)

    kin = KinematicState(
        cat(particles.kin.gamma, source.kin.gamma),
        cat(particles.kin.xi, source.kin.xi),
        cat(particles.kin.x, source.kin.x),
        cat(particles.kin.y, source.kin.y),
        cat(particles.kin.phi, source.kin.phi),
    )
    w = cat(jnp.where(particles.alive, particles.weight, 0.0), source.weight)
    source_pid = derive_particle_ids(
        jnp.arange(ns, dtype=index_dtype()),
        global_step,
        jnp.asarray(1, dtype=index_dtype()),
    )
    # Source slots receive deterministic IDs distinct from existing markers.
    pid = jnp.concatenate([particles.pid, source_pid])
    return ParticleState(kin, w, w > 0.0, pid)


def inject_and_thin(particles, source_kin, total_source_weight, key, *, global_step=0):
    """Inject source candidates, thinning only when local capacity overflows."""

    candidates = append_source_candidates(
        particles,
        source_kin,
        total_source_weight,
        global_step=global_step,
    )
    out, _diagnostics = capacity_control(
        candidates,
        particles.weight.shape[0],
        key,
        global_step=global_step,
    )
    return out
