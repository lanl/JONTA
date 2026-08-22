import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.config import ExecutionConfig
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles
from fields.uniform import UniformField
from integrators import rk4_step
from orbits.zero_d import zero_d_rhs
from simulation import build_particle_block


def test_compiled_particle_block_advances_zero_d():
    p = particles_from_arrays([2.0, 2.5], [-0.8, -0.9], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0])
    r = jnp.array([0.0, 1.0])
    bg = BackgroundProfiles(r, jnp.full(2, 1e14), jnp.full(2, 1e3), jnp.full(2, 1e3), jnp.ones(2), jnp.ones(2))
    def orbit(kin, t, field):
        return zero_d_rhs(kin, t, field, alpha_syn=0.0)

    block = build_particle_block(orbit, rk4_step, n_steps=4)
    out, maxq = block(p, UniformField(2.0), bg, 0.0, 1e-3, jax.random.key(0), 0)
    assert bool(jnp.all(out.kin.gamma > p.kin.gamma))
    assert float(maxq) == 0.0


def test_parallel_particle_block_matches_serial_reference_on_cpu():
    if jax.local_device_count() < 2:
        pytest.skip("expose at least two CPU devices with XLA_FLAGS")

    p = particles_from_arrays(
        [2.0, 2.5, 3.0, 3.5],
        [-0.8, -0.9, -0.7, -0.6],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
    )
    r = jnp.array([0.0, 1.0])
    bg = BackgroundProfiles(
        r,
        jnp.full(2, 1e14),
        jnp.full(2, 1e3),
        jnp.full(2, 1e3),
        jnp.ones(2),
        jnp.ones(2),
    )
    def orbit(kin, t, field):
        return zero_d_rhs(kin, t, field, alpha_syn=0.0)

    serial = build_particle_block(
        orbit,
        rk4_step,
        n_steps=4,
        execution=ExecutionConfig(mode="serial", platform="cpu"),
    )
    parallel = build_particle_block(
        orbit,
        rk4_step,
        n_steps=4,
        execution=ExecutionConfig(mode="parallel", platform="cpu", n_devices=2),
    )

    serial_out, serial_q = serial(p, UniformField(2.0), bg, 0.0, 1e-3, jax.random.key(0), 0)
    parallel_out, parallel_q = parallel(
        p, UniformField(2.0), bg, 0.0, 1e-3, jax.random.key(0), 0
    )
    for expected, actual in zip(serial_out.kin, parallel_out.kin):
        np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=0.0, atol=1e-14)
    np.testing.assert_array_equal(np.asarray(parallel_out.pid), np.asarray(serial_out.pid))
    assert float(parallel_q) == float(serial_q)
