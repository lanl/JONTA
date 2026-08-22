"""Isotropic Compton source following Ekmark et al. (JPP 2024)."""

from __future__ import annotations

import jax.numpy as jnp

from core.constants import ME_C2_EV, R_E_M
from core.precision import real_dtype


def klein_nishina_dsigma_domega(photon_energy_ev, electron_gamma):
    """Klein-Nishina d sigma/d Omega in m^2 for a given outgoing electron."""

    wg = photon_energy_ev / ME_C2_EV
    w = electron_gamma - 1.0
    wgp = wg - w
    valid = wgp > 0.0
    cos_theta = 1.0 - w / jnp.maximum(wg * wgp, 1.0e-30)
    cos_theta = jnp.clip(cos_theta, -1.0, 1.0)
    sin2 = 1.0 - cos_theta * cos_theta
    ds = 0.5 * R_E_M**2 * (wgp / wg) ** 2 * (
        wg / jnp.maximum(wgp, 1.0e-30)
        + wgp / wg
        - sin2
    )
    return jnp.where(valid, ds, 0.0)


def compton_source_density_p(
    p,
    ne_total_m3,
    photon_energy_ev,
    gamma_flux_per_m2_s_ev,
):
    """Evaluate the isotropic Ekmark Compton source on a photon-energy grid.

    ``gamma_flux_per_m2_s_ev`` is the incident differential photon flux
    Gamma_gamma(W_gamma). The photon-energy grid must be ascending.
    """

    p = jnp.asarray(p, dtype=real_dtype())
    gamma = jnp.sqrt(1.0 + p * p)
    beta = p / gamma
    wg = photon_energy_ev / ME_C2_EV
    lower = 0.5 * (p + gamma - 1.0)

    # Broadcast particles x photon energies.
    wg2 = wg[None, :]
    g2 = gamma[..., None]
    ds = klein_nishina_dsigma_domega(photon_energy_ev[None, :], g2)
    denom = (wg2 + 1.0 - g2) ** 2
    integrand = (
        gamma_flux_per_m2_s_ev[None, :]
        * ds
        * beta[..., None]
        / jnp.maximum(denom, 1.0e-30)
    )
    integrand = jnp.where(wg2 >= lower[..., None], integrand, 0.0)
    integral = jnp.trapezoid(integrand, photon_energy_ev, axis=-1)
    return 0.5 * ne_total_m3 * integral / jnp.maximum(p * p, 1.0e-30)
