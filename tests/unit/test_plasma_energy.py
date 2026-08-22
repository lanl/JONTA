import jax.numpy as jnp

from plasma.energy import EnergyPowers, bdf2_energy_step, electron_power, ion_power


def test_energy_power_sign_convention():
    p = EnergyPowers(
        ohmic_w_m3=jnp.array(10.0),
        runaway_collisional_w_m3=jnp.array(3.0),
        radiation_w_m3=jnp.array(2.0),
        electron_ion_w_m3=jnp.array(4.0),
        electron_transport_w_m3=jnp.array(1.0),
        ion_transport_w_m3=jnp.array(0.5),
        external_electron_w_m3=jnp.array(2.0),
        external_ion_w_m3=jnp.array(1.0),
    )
    assert jnp.allclose(electron_power(p), 8.0)
    assert jnp.allclose(ion_power(p), 4.5)


def test_bdf2_energy_constant_power_is_exact_for_linear_solution():
    dt = 0.2
    power = jnp.array(5.0)
    u_nm1 = jnp.array(1.0)
    u_n = u_nm1 + power * dt
    u_np1 = bdf2_energy_step(u_n, u_nm1, power, dt)
    assert jnp.allclose(u_np1, u_n + power * dt, rtol=1e-14, atol=1e-14)
