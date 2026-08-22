"""Fixed-field coupling invariants."""

import jax.numpy as jnp
import numpy as np

from core.config import OrbitNormalization
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles, CircularFieldProfiles, FieldHistory
from coupling import picard_ramc1d_step
from coupling.electric_field import ElectricFieldBoundary, bdf2_electric_field_step


def _background(grid):
    return BackgroundProfiles(
        grid,
        jnp.full_like(grid, 1.0e14),
        jnp.full_like(grid, 1.0e3),
        jnp.full_like(grid, 1.0e3),
        jnp.ones_like(grid),
        jnp.ones_like(grid),
    )


def test_bdf2_zero_field_remains_zero():
    r = jnp.linspace(0.0, 1.0, 32)
    z = jnp.zeros_like(r)
    eta = jnp.ones_like(r)
    out = bdf2_electric_field_step(
        r,
        z,
        z,
        z,
        z,
        z,
        eta,
        eta,
        eta,
        1.0e-3,
        epsilon=1.0 / 3.0,
        boundary=ElectricFieldBoundary("conducting", 1.0),
    )
    assert jnp.allclose(out, z)


def test_picard_returns_particles_pushed_with_returned_field():
    """Final particle state must use the final Picard electric field."""

    grid = jnp.linspace(0.0, 1.0, 5)
    particles = particles_from_arrays(
        [2.0, 2.5], [0.0, 0.0], [0.2, 0.8], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]
    )
    background = _background(grid)
    field = CircularFieldProfiles(grid, jnp.array([1.0, 0.8, 0.6, 0.3, 0.0]), jnp.ones(5))
    history = FieldHistory(
        field.e1, field.e1, jnp.zeros(5), jnp.zeros(5), jnp.ones(5), jnp.ones(5)
    )

    def field_sensitive_block(state, profiles, background, time, dt, key, step):
        gamma = 2.0 + jnp.interp(jnp.abs(state.kin.x), profiles.r, profiles.e1)
        return state._replace(kin=state.kin._replace(gamma=gamma)), jnp.array(0.0)

    result = picard_ramc1d_step(
        particles,
        field,
        background,
        history,
        field_sensitive_block,
        particle_dt=0.01,
        base_key=jnp.array([0, 0], dtype=jnp.uint32),
        global_step0=0,
        norm=OrbitNormalization(0.1, 1.0, 1.0),
        a_minor_cm=100.0,
        dt_coupling=0.1,
        max_iterations=1,
    )
    expected = 2.0 + jnp.interp(jnp.abs(particles.kin.x), result.field_profiles.r, result.field_profiles.e1)
    np.testing.assert_allclose(result.particles.kin.gamma, expected, rtol=1e-6, atol=1e-6)
    assert result.converged is False
