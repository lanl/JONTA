"""Radial-profile sampling helpers."""

from __future__ import annotations

import jax.numpy as jnp

from core.math import finite_difference_profile
from core.state import BackgroundProfiles, CircularFieldProfiles


def interp1(r, grid, values):
    """Linear interpolation with endpoint clamping."""

    return jnp.interp(r, grid, values, left=values[0], right=values[-1])


def sample_background(r, profiles: BackgroundProfiles):
    return (
        interp1(r, profiles.r, profiles.ne_cm3),
        interp1(r, profiles.r, profiles.te_ev),
        interp1(r, profiles.r, profiles.ti_ev),
        interp1(r, profiles.r, profiles.zeff),
        interp1(r, profiles.r, profiles.eta_bar),
    )


def sample_circular_profiles(r, profiles: CircularFieldProfiles):
    de1 = finite_difference_profile(profiles.e1, profiles.r)
    dq = finite_difference_profile(profiles.q, profiles.r)
    e1 = interp1(r, profiles.r, profiles.e1)
    q = interp1(r, profiles.r, profiles.q)
    de1_dr = interp1(r, profiles.r, de1)
    dq_dr = interp1(r, profiles.r, dq)
    # Regularity at the magnetic axis.
    de1_dr = jnp.where(r <= profiles.r[1] * 0.5, 0.0, de1_dr)
    dq_dr = jnp.where(r <= profiles.r[1] * 0.5, 0.0, dq_dr)
    return e1, de1_dr, q, dq_dr
