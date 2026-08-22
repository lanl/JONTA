import jax.numpy as jnp

from coupling.electric_field import ElectricFieldBoundary, bdf2_electric_field_step


def test_bdf2_zero_field_remains_zero():
    r = jnp.linspace(0.0, 1.0, 32)
    z = jnp.zeros_like(r)
    eta = jnp.ones_like(r)
    out = bdf2_electric_field_step(
        r, z, z, z, z, z, eta, eta, eta, 1e-3,
        epsilon=1/3,
        boundary=ElectricFieldBoundary("conducting", 1.0),
    )
    assert jnp.allclose(out, z)
