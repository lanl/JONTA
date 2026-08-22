"""Deterministic per-particle random streams.

Streams are keyed by global seed, global particle-step index, operator stream,
and stable particle slot ID. This makes stochastic results independent of how
particles are sharded across devices.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .precision import real_dtype


def derive_particle_ids(pid, step, branch):
    """Derive deterministic child IDs without resetting marker identity.

    ``step`` and ``branch`` distinguish births from one parent.  IDs need only
    be unique within a live ensemble; the global particle step already makes
    random streams distinct across time.  Arithmetic intentionally follows
    the configured index width so FP32 experiments remain accelerator-safe.
    """

    dtype = jnp.asarray(pid).dtype
    if dtype == jnp.int32:
        multiplier = jnp.asarray(1103515245, dtype=dtype)
        step_multiplier = jnp.asarray(12345, dtype=dtype)
        mask = jnp.asarray(0x7FFFFFFF, dtype=dtype)
    else:
        multiplier = jnp.asarray(6364136223846793005, dtype=dtype)
        step_multiplier = jnp.asarray(1442695040888963407, dtype=dtype)
        mask = jnp.asarray(0x7FFFFFFFFFFFFFFF, dtype=dtype)
    value = (
        jnp.asarray(pid, dtype=dtype) * multiplier
        + jnp.asarray(step, dtype=dtype) * step_multiplier
        + jnp.asarray(branch, dtype=dtype)
    )
    return jnp.bitwise_and(value, mask)


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
