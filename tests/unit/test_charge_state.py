import jax.numpy as jnp

from plasma.charge_state import bdf2_charge_state_step


def test_charge_state_bdf2_conserves_species_density():
    n_n = jnp.array([1.0, 0.0, 0.0])
    n_nm1 = n_n
    ion = jnp.array([1e-14, 1e-14, 0.0])
    rec = jnp.array([0.0, 5e-15, 5e-15])
    out = bdf2_charge_state_step(n_n, n_nm1, jnp.array(1e19), ion, rec, 1e-6)
    assert jnp.allclose(jnp.sum(out), 1.0, rtol=1e-12)
    assert bool(jnp.all(out >= 0.0))
