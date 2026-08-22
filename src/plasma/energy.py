"""Electron and ion thermal-energy bookkeeping."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from core.constants import E_CHARGE_C


class EnergyPowers(NamedTuple):
    ohmic_w_m3: jnp.ndarray
    runaway_collisional_w_m3: jnp.ndarray
    radiation_w_m3: jnp.ndarray
    electron_ion_w_m3: jnp.ndarray
    electron_transport_w_m3: jnp.ndarray
    ion_transport_w_m3: jnp.ndarray
    external_electron_w_m3: jnp.ndarray
    external_ion_w_m3: jnp.ndarray


def electron_power(p: EnergyPowers):
    """Positive values heat electrons; e-i term is positive e -> i transfer."""

    return (
        p.ohmic_w_m3
        + p.runaway_collisional_w_m3
        - p.radiation_w_m3
        - p.electron_ion_w_m3
        - p.electron_transport_w_m3
        + p.external_electron_w_m3
    )


def ion_power(p: EnergyPowers):
    return p.electron_ion_w_m3 - p.ion_transport_w_m3 + p.external_ion_w_m3


def thermal_energy_density_j_m3(n_m3, temperature_ev):
    return 1.5 * n_m3 * E_CHARGE_C * temperature_ev


def temperature_from_energy_ev(energy_j_m3, n_m3):
    return energy_j_m3 / jnp.maximum(1.5 * n_m3 * E_CHARGE_C, 1.0e-300)


def bdf2_energy_step(u_n, u_nm1, power_np1, dt_s):
    """BDF2 update when the n+1 power has been supplied by the coupling iteration."""

    return (4.0 * u_n - u_nm1 + 2.0 * dt_s * power_np1) / 3.0
