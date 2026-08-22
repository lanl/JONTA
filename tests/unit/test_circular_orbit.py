import jax
import jax.numpy as jnp

from core.config import OrbitNormalization
from core.state import CircularFieldProfiles, KinematicState
from fields.circular import sample_circular_field
from orbits.ramc_circular import ramc_circular_rhs


def test_magnetic_field_does_no_work():
    grid = jnp.linspace(0.0, 1.0, 65)
    profiles = CircularFieldProfiles(grid, jnp.zeros_like(grid), 2.1 + 2.0 * grid**2)
    norm = OrbitNormalization(1.0 / 3.0, 5.0e5, 6.22e3, 0.0)
    kin = KinematicState(
        jnp.array([2.0]), jnp.array([-0.7]), jnp.array([0.3]), jnp.array([0.1]), jnp.array([0.2])
    )
    rhs = ramc_circular_rhs(kin, 0.0, profiles, norm)
    assert abs(float(rhs.gamma[0])) < 1e-13
    assert bool(jnp.all(jnp.isfinite(jnp.stack(rhs))))


def test_magnetic_moment_rhs_is_small_without_electric_field():
    grid = jnp.linspace(0.0, 1.0, 129)
    profiles = CircularFieldProfiles(grid, jnp.zeros_like(grid), 2.1 + 2.0 * grid**2)
    norm = OrbitNormalization(1.0 / 3.0, 5.0e5, 6.22e3, 0.0)
    kin = KinematicState(
        jnp.array([1.8]), jnp.array([-0.6]), jnp.array([0.35]), jnp.array([0.12]), jnp.array([0.0])
    )
    rhs = ramc_circular_rhs(kin, 0.0, profiles, norm)

    def mu_of(k):
        fs = sample_circular_field(k, profiles, norm)
        p2 = k.gamma * k.gamma - 1.0
        return p2 * (1.0 - k.xi * k.xi) / (2.0 * fs.B_mag)

    _, dmu = jax.jvp(mu_of, (kin,), (rhs,))
    assert abs(float(dmu[0])) < 1e-4
