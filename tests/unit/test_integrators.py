import jax
import jax.numpy as jnp

from integrators import (
    adaptive_bogacki_shampine5_interval,
    bogacki_shampine5_step,
    midpoint_step,
    rk4_step,
)


def test_rk4_scalar_accuracy():
    def rhs(y, t):
        return y

    y1 = rk4_step(rhs, jnp.array(1.0), 0.0, 0.1)
    assert jnp.allclose(y1, jnp.exp(0.1), rtol=1e-6)


def test_midpoint_is_second_order_reasonable():
    def rhs(y, t):
        return -2.0 * y

    y1 = midpoint_step(rhs, jnp.array(1.0), 0.0, 0.05)
    assert abs(float(y1 - jnp.exp(-0.1))) < 2e-4


def test_bogacki_shampine5_has_fifth_order_fixed_step():
    def rhs(y, t):
        return y

    coarse, _, _ = bogacki_shampine5_step(rhs, jnp.array(1.0), 0.0, 0.2)
    fine, _, _ = bogacki_shampine5_step(rhs, jnp.array(1.0), 0.0, 0.1)
    exact_coarse = jnp.exp(0.2)
    exact_fine = jnp.exp(0.1)
    coarse_error = abs(coarse - exact_coarse)
    fine_error = abs(fine - exact_fine)
    assert float(coarse_error / fine_error) > 20.0


def test_bogacki_shampine5_embedded_error_is_consistent():
    def rhs(y, t):
        return y

    fifth, fourth, error = bogacki_shampine5_step(rhs, jnp.array(1.0), 0.0, 0.1)
    assert float(abs(fifth - jnp.exp(0.1))) < 1.0e-10
    assert float(abs(error - (fifth - fourth))) < 1.0e-15


def test_adaptive_bogacki_shampine5_reaches_interval():
    def rhs(y, t):
        return y

    result = adaptive_bogacki_shampine5_interval(
        rhs,
        jnp.array(1.0),
        0.0,
        0.5,
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    assert bool(result.converged)
    assert int(result.accepted_steps) > 0
    assert float(abs(result.state - jnp.exp(0.5))) < 1.0e-10


def test_adaptive_bogacki_shampine5_is_jit_vmap_safe():
    def rhs(y, t):
        return y

    advance = jax.jit(
        jax.vmap(
            lambda value: adaptive_bogacki_shampine5_interval(
                rhs, value, 0.0, 0.2, atol=1.0e-10, rtol=1.0e-10
            )
        )
    )
    result = advance(jnp.asarray([1.0, 2.0]))
    assert bool(jnp.all(result.converged))
    assert jnp.allclose(result.state, jnp.asarray([1.0, 2.0]) * jnp.exp(0.2), atol=1.0e-8)
