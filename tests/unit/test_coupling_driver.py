"""Geometry-neutral Picard driver contracts."""

import jax.numpy as jnp

from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles
from coupling import ParticleMoments, picard_particle_field_step
from deposition.slab import deposit_slab_parallel_current, deposit_slab_weight
from fields.uniform import UniformField


def test_generic_picard_driver_accepts_scalar_slab_backend():
    particles = particles_from_arrays(
        [2.0, 3.0], [-0.8, 0.4], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 2.0]
    )
    grid = jnp.array([0.0, 1.0])
    background = BackgroundProfiles(
        grid,
        jnp.full(2, 1.0e14),
        jnp.full(2, 1.0e3),
        jnp.full(2, 1.0e3),
        jnp.ones(2),
        jnp.ones(2),
    )

    def particle_block(state, field, background, time, dt, key, step):
        del field, background, time, dt, key, step
        return state, jnp.asarray(0.0)

    def slab_moments(pool):
        return ParticleMoments(
            deposit_slab_parallel_current(pool, 100.0),
            deposit_slab_weight(pool),
        )

    def make_particle_field(field_n, guess):
        del field_n
        return UniformField(guess)

    def solve_field(field_n, history, moments, background, dt):
        del history, moments, background, dt
        return field_n.e_parallel

    def finalize_field(field_n, guess, moments, background, dt):
        del field_n, moments, background, dt
        return UniformField(guess)

    def update_history(field_n, history, field, moments, background):
        del field_n, history, field, background
        return {"j": moments.parallel_current}

    result = picard_particle_field_step(
        particles,
        UniformField(2.0),
        background,
        {"j": jnp.asarray(0.0)},
        particle_block,
        particle_dt=1.0e-3,
        base_key=jnp.array([0, 0], dtype=jnp.uint32),
        global_step0=0,
        dt_coupling=0.1,
        initial_guess=jnp.asarray(2.0),
        make_particle_field=make_particle_field,
        solve_field=solve_field,
        finalize_field=finalize_field,
        update_history=update_history,
        moment_depositor=slab_moments,
    )

    assert result.iterations == 1
    assert result.converged
    assert float(result.field_state.e_parallel) == 2.0
    assert float(result.history["j"]) != 0.0
