"""Local-capacity population control under CPU device sharding."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from collisions.moller import apply_moller_source_only
from core.config import ExecutionConfig, MollerConfig
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles
from fields.uniform import UniformField
from integrators.explicit import rk4_step
from orbits.zero_d import zero_d_rhs
from simulation import build_particle_block


def test_parallel_moller_uses_local_capacity_and_preserves_weight():
    if jax.local_device_count() < 2:
        pytest.skip("expose at least two CPU devices with XLA_FLAGS")

    particles = particles_from_arrays(
        [5.0, 6.0, 7.0, 8.0],
        [-0.8, -0.8, -0.8, -0.8],
        [0.0] * 4,
        [0.0] * 4,
        [0.0] * 4,
        [1.0] * 4,
    )
    r = jnp.asarray([0.0, 1.0])
    background = BackgroundProfiles(
        r,
        jnp.full(2, 1.0e14),
        jnp.full(2, 1.0e3),
        jnp.full(2, 1.0e3),
        jnp.ones(2),
        jnp.ones(2),
    )
    config = MollerConfig(1.0e14, 15.0, 1.1, max_collision_fraction=0.2)

    def orbit(kin, time, field):
        return zero_d_rhs(kin, time, field, alpha_syn=0.0)

    def large_angle(state, bg, dt_large, key, step):
        return apply_moller_source_only(state, bg, dt_large, key, step, config)

    serial = build_particle_block(
        orbit,
        rk4_step,
        n_steps=1,
        large_angle_operator=large_angle,
        large_angle_every=1,
        execution=ExecutionConfig(mode="serial", platform="cpu"),
    )
    parallel = build_particle_block(
        orbit,
        rk4_step,
        n_steps=1,
        large_angle_operator=large_angle,
        large_angle_every=1,
        execution=ExecutionConfig(mode="parallel", platform="cpu", n_devices=2),
    )
    field = UniformField(2.0)
    serial_out, serial_q = serial(
        particles, field, background, 0.0, 1.0e-3, jax.random.key(2), 0
    )
    parallel_out, parallel_q = parallel(
        particles, field, background, 0.0, 1.0e-3, jax.random.key(2), 0
    )

    assert serial_out.weight.shape == parallel_out.weight.shape == (4,)
    assert jnp.all(jnp.isfinite(parallel_out.kin.gamma))
    assert jnp.all(parallel_out.weight >= 0.0)
    np.testing.assert_allclose(
        np.asarray(jnp.sum(parallel_out.weight)),
        np.asarray(jnp.sum(serial_out.weight)),
        rtol=0.0,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.asarray(parallel_q), np.asarray(serial_q), rtol=0.0, atol=1.0e-12)
