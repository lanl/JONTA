import jax.numpy as jnp

from integrators import midpoint_step, rk4_step


def test_rk4_scalar_accuracy():
    rhs = lambda y, t: y
    y1 = rk4_step(rhs, jnp.array(1.0), 0.0, 0.1)
    assert jnp.allclose(y1, jnp.exp(0.1), rtol=1e-6)


def test_midpoint_is_second_order_reasonable():
    rhs = lambda y, t: -2.0 * y
    y1 = midpoint_step(rhs, jnp.array(1.0), 0.0, 0.05)
    assert abs(float(y1 - jnp.exp(-0.1))) < 2e-4
