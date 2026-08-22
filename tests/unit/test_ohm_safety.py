import jax.numpy as jnp

from coupling.ohm import electric_field_from_total_current, ohmic_current
from coupling.safety_factor import update_circular_q


def test_algebraic_ohm_round_trip():
    eta = jnp.array([2.0, 4.0])
    jt = jnp.array([3.0, 2.0])
    jk = jnp.array([1.0, 0.5])
    e = electric_field_from_total_current(jt, jk, eta)
    assert jnp.allclose(ohmic_current(e, eta) + jk, jt)


def test_q_is_constant_for_uniform_total_current_density():
    r = jnp.linspace(0.0, 1.0, 128)
    epsilon = 1.0 / 3.0
    wce = 6000.0
    eta = jnp.ones_like(r)
    e = jnp.zeros_like(r)
    j = jnp.full_like(r, 2.0)
    q = update_circular_q(r, e, eta, j, epsilon, wce)
    # Integral r*j dr = j*r^2/2, so q is radially constant away from the axis.
    assert jnp.max(jnp.abs(q[2:] - q[2])) < 2e-12
