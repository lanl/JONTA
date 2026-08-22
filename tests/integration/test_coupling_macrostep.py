"""RAMc-style particle macrostep and moment-reduction contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.config import ExecutionConfig, OrbitNormalization
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles, CircularFieldProfiles, KinematicState
from coupling.macrostep import (
    ParticleMoments,
    advance_particle_macrostep,
    reduce_particle_moments,
)
from coupling.ramc1d import deposit_circular_particle_moments
from deposition.slab import deposit_slab_parallel_current, deposit_slab_weight
from fields.uniform import UniformField
from integrators import rk4_step
from parallel.sharding import partition_particles
from simulation import build_particle_block


def _background(grid):
    return BackgroundProfiles(
        grid,
        jnp.full_like(grid, 1.0e14),
        jnp.full_like(grid, 1.0e3),
        jnp.full_like(grid, 1.0e3),
        jnp.ones_like(grid),
        jnp.ones_like(grid),
    )


def _circular_depositor(grid, q, background, norm, a_minor_cm):
    def deposit(pool):
        return deposit_circular_particle_moments(
            pool, grid, q, background, norm, a_minor_cm
        )

    return deposit


def test_sharded_moment_reduction_matches_serial_deposition():
    grid = jnp.linspace(0.0, 1.0, 5)
    particles = particles_from_arrays(
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        [-0.8, -0.7, -0.6, 0.4, 0.5, 0.6],
        [0.1, 0.2, 0.4, 0.6, 0.8, 0.9],
        [0.0] * 6,
        [0.0] * 6,
        [1.0, 2.0, 1.5, 0.5, 3.0, 1.0],
        alive=[True, True, False, True, True, True],
    )
    background = _background(grid)
    norm = OrbitNormalization(0.1, 1.0, 1.0)
    depositor = _circular_depositor(grid, jnp.ones_like(grid), background, norm, 100.0)
    serial = depositor(particles)
    sharded = reduce_particle_moments(partition_particles(particles, 2), depositor)
    np.testing.assert_allclose(sharded.parallel_current, serial.parallel_current)
    np.testing.assert_allclose(sharded.weight_density, serial.weight_density)


def test_parallel_macrostep_preserves_local_partitions_for_reduction():
    if jax.local_device_count() < 2:
        pytest.skip("expose at least two CPU devices with XLA_FLAGS")

    grid = jnp.linspace(0.0, 1.0, 5)
    particles = particles_from_arrays(
        [2.0, 3.0, 4.0, 5.0],
        [-0.8, -0.7, 0.4, 0.5],
        [0.1, 0.2, 0.6, 0.8],
        [0.0] * 4,
        [0.0] * 4,
        [1.0] * 4,
    )
    background = _background(grid)
    norm = OrbitNormalization(0.1, 1.0, 1.0)
    depositor = _circular_depositor(grid, jnp.ones_like(grid), background, norm, 100.0)

    def orbit(kin, time, field):
        del time, field
        return KinematicState(*(jnp.zeros_like(value) for value in kin))

    block = build_particle_block(
        orbit,
        rk4_step,
        n_steps=1,
        execution=ExecutionConfig(mode="parallel", platform="cpu", n_devices=2),
        preserve_partitioning=True,
    )
    result = advance_particle_macrostep(
        block,
        particles,
        CircularFieldProfiles(grid, jnp.ones_like(grid), jnp.ones_like(grid)),
        background,
        time_n=0.0,
        particle_dt=1.0e-3,
        base_key=jax.random.key(0),
        global_step0=0,
        moment_depositor=depositor,
    )
    assert result.particles.weight.shape == (2, 2)
    serial = depositor(particles)
    np.testing.assert_allclose(result.moments.weight_density, serial.weight_density)


def test_macrostep_accepts_geometry_neutral_slab_depositor():
    particles = particles_from_arrays(
        [2.0, 3.0], [-0.8, 0.4], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0]
    )
    grid = jnp.array([0.0, 1.0])
    background = _background(grid)

    def orbit(kin, time, field):
        del time, field
        return KinematicState(*(jnp.zeros_like(value) for value in kin))

    def slab_moments(pool):
        return ParticleMoments(
            deposit_slab_parallel_current(pool, 100.0),
            deposit_slab_weight(pool),
        )

    block = build_particle_block(orbit, rk4_step, n_steps=1)
    result = advance_particle_macrostep(
        block,
        particles,
        UniformField(2.0),
        background,
        time_n=0.0,
        particle_dt=1.0e-3,
        base_key=jax.random.key(0),
        global_step0=0,
        moment_depositor=slab_moments,
    )
    assert result.moments.weight_density == 3.0
    assert jnp.isfinite(result.moments.parallel_current)
    reduced = reduce_particle_moments(partition_particles(particles, 2), slab_moments)
    assert reduced.weight_density == 3.0
    np.testing.assert_allclose(reduced.parallel_current, result.moments.parallel_current)
