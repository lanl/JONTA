"""CPU multi-device correctness checks for deterministic particle evolution.

Run with logical CPU devices exposed before Python starts, for example::

    XLA_FLAGS=--xla_force_host_platform_device_count=2 \
      JAX_PLATFORMS=cpu pytest -q tests/integration/test_parallel_orbit.py

The test deliberately excludes Møller population control.  That operator
uses local fixed-capacity population control and is validated separately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.state import ParticleState
from integrators.explicit import rk4_step
from orbits.ramc_circular import ramc_circular_rhs
from parallel import merge_particle_partitions, partition_particles
from tests.validation.test_guiding_center_invariants import make_case

jax.config.update("jax_enable_x64", True)


def _advance(kin, profiles, norm):
    """Advance one local marker batch through a fixed deterministic orbit."""

    def rhs(state, time):
        return ramc_circular_rhs(state, time, profiles, norm)

    def body(step, state):
        return rk4_step(rhs, state, step * 1.0e-6, 1.0e-6)

    return jax.lax.fori_loop(0, 10, body, kin)


def test_cpu_particle_partition_matches_single_device_reference():
    """Particle sharding must preserve deterministic orbit trajectories."""

    if jax.local_device_count() < 2:
        pytest.skip("expose at least two CPU devices with XLA_FLAGS")

    n_devices = min(2, jax.local_device_count())
    kin, profiles, norm = make_case(n_particles=32, q2=0.0, e1=0.0)
    particles = ParticleState(
        kin,
        jnp.ones(kin.gamma.shape),
        jnp.ones(kin.gamma.shape, dtype=bool),
        jnp.arange(kin.gamma.shape[0], dtype=jnp.int64),
    )
    reference = _advance(kin, profiles, norm)

    partitioned = partition_particles(particles, n_devices)
    mapped = jax.pmap(
        _advance,
        in_axes=(0, None, None),
        devices=jax.local_devices()[:n_devices],
    )(partitioned.kin, profiles, norm)
    mapped_particles = ParticleState(
        mapped,
        partitioned.weight,
        partitioned.alive,
        partitioned.pid,
    )
    merged = merge_particle_partitions(mapped_particles)

    for expected, actual in zip(reference, merged.kin):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=5.0e-14)
