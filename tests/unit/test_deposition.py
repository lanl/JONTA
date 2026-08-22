import jax.numpy as jnp

from core.initialization import particles_from_arrays
from deposition.radial import deposit_weight_density, linear_bin_sum


def test_linear_bin_sum_preserves_total():
    grid = jnp.linspace(0.0, 1.0, 6)
    r = jnp.array([0.1, 0.37, 0.8])
    values = jnp.array([2.0, 3.0, 5.0])
    out = linear_bin_sum(r, values, grid)
    assert jnp.allclose(jnp.sum(out), jnp.sum(values), rtol=1e-14, atol=1e-14)


def test_weight_deposition_preserves_alive_weight():
    p = particles_from_arrays(
        gamma=jnp.array([2.0, 2.0]),
        xi=jnp.array([0.0, 0.0]),
        x=jnp.array([0.25, 0.75]),
        y=jnp.array([0.0, 0.0]),
        phi=jnp.array([0.0, 0.0]),
        weight=jnp.array([2.0, 3.0]),
        alive=jnp.array([True, False]),
    )
    out = deposit_weight_density(p, jnp.linspace(0.0, 1.0, 5))
    assert jnp.allclose(jnp.sum(out), 2.0)
