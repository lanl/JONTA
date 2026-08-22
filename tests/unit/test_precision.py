import jax
import jax.numpy as jnp

from core.initialization import particles_from_arrays
from core.precision import PrecisionConfig, configure_precision, current_precision
from core.state import BackgroundProfiles
from fields.uniform import UniformField
from integrators import rk4_step
from orbits.zero_d import zero_d_rhs
from simulation import build_particle_block


def test_precision_configs_select_real_and_index_dtypes():
    fp32 = PrecisionConfig(32)
    fp64 = PrecisionConfig(64)
    assert fp32.real_dtype is jnp.float32
    assert fp32.index_dtype is jnp.int32
    assert fp64.real_dtype is jnp.float64
    assert fp64.index_dtype is jnp.int64


def test_particle_initialization_accepts_explicit_float32_precision():
    previous = current_precision()
    try:
        particles = particles_from_arrays(
            [2.0],
            [-0.8],
            [0.0],
            [0.0],
            [0.0],
            [1.0],
            precision=32,
        )
        assert particles.kin.gamma.dtype == jnp.float32
        assert particles.weight.dtype == jnp.float32
        assert particles.pid.dtype == jnp.int32

        # Omitted precision must preserve process selection, not silently
        # re-read environment/default precision.
        retained = particles_from_arrays([2.0], [-0.8], [0.0], [0.0], [0.0], [1.0])
        assert retained.kin.gamma.dtype == jnp.float32
        assert retained.pid.dtype == jnp.int32
    finally:
        configure_precision(previous)


def test_particle_block_builds_with_float32_precision():
    previous = current_precision()
    try:
        particles = particles_from_arrays(
            [2.0], [-0.8], [0.0], [0.0], [0.0], [1.0], precision=32
        )
        background = BackgroundProfiles(
            jnp.asarray([0.0, 1.0], dtype=jnp.float32),
            jnp.full(2, 1.0e14, dtype=jnp.float32),
            jnp.full(2, 1.0e3, dtype=jnp.float32),
            jnp.full(2, 1.0e3, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32),
            jnp.ones(2, dtype=jnp.float32),
        )
        block = build_particle_block(
            lambda kin, t, field: zero_d_rhs(kin, t, field),
            rk4_step,
            n_steps=1,
            precision=32,
        )
        output, _ = block(
            particles,
            UniformField(2.0),
            background,
            0.0,
            1.0e-3,
            jax.random.key(0),
            0,
        )
        assert output.kin.gamma.dtype == jnp.float32
        assert output.pid.dtype == jnp.int32
    finally:
        configure_precision(previous)
