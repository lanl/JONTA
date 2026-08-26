import jax.numpy as jnp

from fields.interpolated import (
    AxisymmetricFieldTable,
    bilinear_regular_grid,
    sample_axisymmetric_table,
)


def test_bilinear_exact_for_affine_field():
    r = jnp.array([1.0, 2.0, 4.0])
    z = jnp.array([-1.0, 0.5, 2.0])
    rr, zz = jnp.meshgrid(r, z, indexing="ij")
    values = 2.0 * rr - 3.0 * zz + 5.0
    x = jnp.array([1.5, 3.0])
    y = jnp.array([-0.25, 1.25])
    got = bilinear_regular_grid(r, z, values, x, y)
    want = 2.0 * x - 3.0 * y + 5.0
    assert jnp.allclose(got, want, rtol=1e-13, atol=1e-13)


def test_axisymmetric_table_samples_all_components():
    r = jnp.array([1.0, 2.0])
    z = jnp.array([-1.0, 1.0])
    rr, zz = jnp.meshgrid(r, z, indexing="ij")
    base = rr + 2.0 * zz
    table = AxisymmetricFieldTable(r, z, base, 2 * base, 3 * base, 4 * base, 5 * base, 6 * base)
    out = sample_axisymmetric_table(table, jnp.array([1.5]), jnp.array([0.0]))
    assert jnp.allclose(out.B_R, 1.5)
    assert jnp.allclose(out.E_phi, 9.0)
