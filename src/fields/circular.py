"""Unshifted circular-flux-surface field used by the legacy RAMc benchmarks."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from core.config import OrbitNormalization
from core.math import safe_radius
from core.state import CircularFieldProfiles, KinematicState, QuadraticCircularFieldProfiles
from .profiles import sample_circular_profiles


class CircularFieldSample(NamedTuple):
    r: jnp.ndarray
    r_safe: jnp.ndarray
    R: jnp.ndarray
    e1: jnp.ndarray
    de1_dr: jnp.ndarray
    q: jnp.ndarray
    dq_dr: jnp.ndarray
    B_R: jnp.ndarray
    B_Z: jnp.ndarray
    B_phi: jnp.ndarray
    B_mag: jnp.ndarray
    E_phi: jnp.ndarray


def sample_circular_field(
    kin: KinematicState,
    profiles: CircularFieldProfiles | QuadraticCircularFieldProfiles,
    norm: OrbitNormalization,
) -> CircularFieldSample:
    """Evaluate the axisymmetric circular equilibrium at marker positions.

    Coordinates follow RAMc: ``x=r cos(theta)``, ``y=r sin(theta)`` with
    ``r`` normalized to minor radius. The major-radius factor is
    ``R/R0 = 1 + epsilon*x``.
    """

    r, r_safe = safe_radius(kin.x, kin.y)
    if isinstance(profiles, QuadraticCircularFieldProfiles):
        e1 = jnp.broadcast_to(jnp.asarray(profiles.e1), r.shape)
        de1_dr = jnp.zeros_like(r)
        q = jnp.asarray(profiles.q0) + jnp.asarray(profiles.q2) * r * r
        dq_dr = 2.0 * jnp.asarray(profiles.q2) * r
    else:
        e1, de1_dr, q, dq_dr = sample_circular_profiles(r, profiles)
    eps = norm.epsilon
    R = 1.0 + eps * kin.x

    b_phi = 1.0 / R
    b_theta = (r * eps / q) / R
    sin_theta = kin.y / r_safe
    cos_theta = kin.x / r_safe
    b_R = -b_theta * sin_theta
    b_Z = b_theta * cos_theta
    b_mag = jnp.sqrt(b_phi * b_phi + b_theta * b_theta)
    e_phi = e1 / R

    return CircularFieldSample(
        r=r,
        r_safe=r_safe,
        R=R,
        e1=e1,
        de1_dr=de1_dr,
        q=q,
        dq_dr=dq_dr,
        B_R=b_R,
        B_Z=b_Z,
        B_phi=b_phi,
        B_mag=b_mag,
        E_phi=e_phi,
    )


def analytic_q_profile(r, q0: float, q2: float):
    return q0 + q2 * r * r
