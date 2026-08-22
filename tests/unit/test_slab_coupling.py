"""Homogeneous slab coupling adapter tests."""

import jax.numpy as jnp

from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles
from coupling import picard_slab_step
from fields.uniform import UniformField


def test_slab_picard_updates_scalar_ohm_field():
    particles = particles_from_arrays(
        [2.0, 3.0], [-0.8, 0.4], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0]
    )
    r = jnp.array([0.0, 1.0])
    background = BackgroundProfiles(
        r,
        jnp.full(2, 1.0e14),
        jnp.full(2, 1.0e3),
        jnp.full(2, 1.0e3),
        jnp.ones(2),
        jnp.ones(2),
    )

    def particle_block(state, field, background, time, dt, key, step):
        del field, background, time, dt, key, step
        return state, jnp.asarray(0.0)

    result = picard_slab_step(
        particles,
        UniformField(0.0),
        background,
        particle_block,
        particle_dt=1.0e-3,
        base_key=jnp.array([0, 0], dtype=jnp.uint32),
        global_step0=0,
        a_minor_cm=100.0,
        dt_coupling=0.1,
        eta=2.0,
        j_total=1.0,
        max_iterations=4,
    )
    expected = 2.0 * (1.0 - result.history.jkin_n)
    assert result.converged
    assert result.field_state.b == 1.0
    assert jnp.allclose(result.field_state.e_parallel, expected)
