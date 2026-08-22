"""Reference resistivity closures."""

from __future__ import annotations

import jax.numpy as jnp

from core.constants import ALFVEN_CURRENT_A, ME_C2_EV, PI
from collisions.coulomb import thermal_coulomb_log


def ramc_spitzer_eta_bar(
    te_ev,
    zeff,
    ne0_cm3: float,
    a_minor_cm: float,
    coulog0: float,
    ne_cm3=None,
):
    """Dimensionless Spitzer resistivity used by RAMc Eq. (89).

    ``eta_bar = eta * I_A / (E_c a^2)``. The implementation follows
    ``EvaluateProfiles.cpp::Geteta`` with the ``UseSpitzer`` branch.
    """

    te = jnp.maximum(te_ev, 1.0e-8)
    if ne_cm3 is None:
        ne_cm3 = jnp.full_like(te, ne0_cm3)
    coulog = thermal_coulomb_log(ne_cm3, te)
    l11 = 0.58 * 32.0 / (3.0 * PI)
    tauc_over_tauei = (
        (1.0 / 3.0)
        * jnp.sqrt(2.0 / PI)
        * zeff
        * (ME_C2_EV / te) ** 1.5
    )
    ec_over_a = 4.8032e-9 / a_minor_cm
    return (
        tauc_over_tauei
        * (1.0 / (ne0_cm3 * a_minor_cm**3))
        * (ALFVEN_CURRENT_A / ec_over_a)
        * (1.0 / l11)
        * (coulog / coulog0)
    )
