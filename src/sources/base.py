"""Fixed-N source injection helpers."""

from __future__ import annotations

import jax.numpy as jnp

from core.precision import index_dtype, real_dtype
from core.state import KinematicState, ParticleState
from resampling.multinomial import multinomial_resample


def append_source_candidates(particles: ParticleState, source_kin: KinematicState, total_source_weight):
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
    return ParticleState(kin, w, w > 0.0, jnp.arange(w.shape[0], dtype=index_dtype()))


def inject_and_thin(particles, source_kin, total_source_weight, key):
    candidates = append_source_candidates(particles, source_kin, total_source_weight)
    return multinomial_resample(candidates, key, particles.weight.shape[0])
