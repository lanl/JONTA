"""Deterministic per-particle random streams.

Streams are keyed by global seed, global particle-step index, operator stream,
and stable particle slot ID. This makes stochastic results independent of how
particles are sharded across devices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .precision import real_dtype


def _particle_keys(base_key, pid, step: int, stream: int):
    key = jax.random.fold_in(base_key, jnp.asarray(step, dtype=jnp.uint32))
    key = jax.random.fold_in(key, jnp.asarray(stream, dtype=jnp.uint32))
    ids = jnp.asarray(pid, dtype=jnp.uint32)
    return jax.vmap(lambda i: jax.random.fold_in(key, i))(ids)


def normal_by_particle(base_key, pid, step: int, stream: int = 0):
    keys = _particle_keys(base_key, pid, step, stream)
    return jax.vmap(lambda k: jax.random.normal(k, dtype=real_dtype()))(keys)


def uniform_by_particle(base_key, pid, step: int, stream: int = 0):
    keys = _particle_keys(base_key, pid, step, stream)
    return jax.vmap(lambda k: jax.random.uniform(k, dtype=real_dtype()))(keys)
