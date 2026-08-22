"""Bounded manufactured tests for circular coupling contracts.

These tests exercise RAMc-specific field/coupling adapters without running a
long particle benchmark.  They verify source sign, BDF2 history flow, safety-
factor consistency, and serial moment deposition.
"""

import jax.numpy as jnp
import numpy as np

from core.config import OrbitNormalization
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles, CircularFieldProfiles, FieldHistory
from coupling import picard_ramc1d_step
from coupling.electric_field import ElectricFieldBoundary, bdf2_electric_field_step
from coupling.macrostep import advance_particle_macrostep
from coupling.ramc1d import deposit_circular_particle_moments
from coupling.safety_factor import update_circular_q


def _background(grid):
    return BackgroundProfiles(
        grid,
        jnp.full_like(grid, 1.0e14),
        jnp.full_like(grid, 1.0e3),
        jnp.full_like(grid, 1.0e3),
        jnp.ones_like(grid),
        jnp.ones_like(grid),
    )


def _particles():
    return particles_from_arrays(
        [3.0, 3.5, 4.0],
        [-0.8, -0.7, -0.6],
        [0.15, 0.45, 0.75],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    )


def test_bdf2_current_history_drives_negative_electric_field():
    """Positive increasing RAMc current produces expected source sign."""

    grid = jnp.linspace(0.0, 1.0, 17)
    zeros = jnp.zeros_like(grid)
    ones = jnp.ones_like(grid)
    out = bdf2_electric_field_step(
        grid,
        zeros,
        zeros,
        zeros,
        zeros,
        ones,
        ones,
        ones,
        ones,
        dt_coupling=0.1,
        epsilon=0.1,
        boundary=ElectricFieldBoundary("conducting", 1.0),
    )

    # dj/dt = (3*0 - 4*0 + 1)/(2*dt) > 0; field source is -eta*dj/dt.
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out[:-1] < 0.0)
    assert out[-1] == 0.0


def test_serial_circular_macrostep_matches_direct_moment_deposition():
    """Serial macrostep path must expose same moments as its depositor."""

    grid = jnp.linspace(0.0, 1.0, 9)
    particles = _particles()
    background = _background(grid)
    norm = OrbitNormalization(0.1, 1.0, 1.0)
    q = jnp.full_like(grid, 2.0)

    def no_op_block(state, field, background, time, dt, key, step):
        del field, background, time, dt, key, step
        return state, jnp.asarray(0.0)

    result = advance_particle_macrostep(
        no_op_block,
        particles,
        CircularFieldProfiles(grid, jnp.full_like(grid, 0.2), q),
        background,
        time_n=0.0,
        particle_dt=1.0e-3,
        base_key=jnp.array([0, 0], dtype=jnp.uint32),
        global_step0=0,
        moment_depositor=lambda pool: deposit_circular_particle_moments(
            pool, grid, q, background, norm, 1.0
        ),
    )
    expected = deposit_circular_particle_moments(
        particles, grid, q, background, norm, a_minor_cm=1.0
    )
    np.testing.assert_allclose(result.moments.parallel_current, expected.parallel_current)
    np.testing.assert_allclose(result.moments.weight_density, expected.weight_density)
    np.testing.assert_array_equal(result.particles.alive, particles.alive)


def test_ramc_picard_updates_q_and_history_from_same_final_moments():
    """Returned q/history must use final field and final particle current."""

    grid = jnp.linspace(0.0, 1.0, 9)
    background = _background(grid)
    particles = _particles()
    field = CircularFieldProfiles(
        grid, jnp.full_like(grid, 0.2), jnp.full_like(grid, 2.0)
    )
    history = FieldHistory(
        field.e1,
        field.e1,
        jnp.zeros_like(grid),
        jnp.zeros_like(grid),
        jnp.ones_like(grid),
        jnp.ones_like(grid),
    )

    def no_op_block(state, field, background, time, dt, key, step):
        del field, background, time, dt, key, step
        return state, jnp.asarray(0.0)

    result = picard_ramc1d_step(
        particles,
        field,
        background,
        history,
        no_op_block,
        particle_dt=1.0e-3,
        base_key=jnp.array([0, 0], dtype=jnp.uint32),
        global_step0=0,
        norm=OrbitNormalization(0.1, 1.0, 1.0),
        a_minor_cm=1.0,
        dt_coupling=0.1,
        max_iterations=1,
        evolve_q=True,
    )

    expected_moments = deposit_circular_particle_moments(
        particles,
        grid,
        field.q,
        background,
        OrbitNormalization(0.1, 1.0, 1.0),
        a_minor_cm=1.0,
    )
    expected_q = update_circular_q(
        grid,
        result.field_profiles.e1,
        background.eta_bar,
        expected_moments.parallel_current,
        epsilon=0.1,
        a_omega_ce_over_c=1.0,
    )
    np.testing.assert_allclose(result.field_profiles.q, expected_q)
    np.testing.assert_allclose(result.history.e1_n, result.field_profiles.e1)
    np.testing.assert_allclose(result.history.jkin_n, expected_moments.parallel_current)
    np.testing.assert_allclose(result.history.e1_nm1, history.e1_n)
    np.testing.assert_allclose(result.history.jkin_nm1, history.jkin_n)
