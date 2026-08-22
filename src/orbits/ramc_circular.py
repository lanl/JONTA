"""Analytic unshifted-circular RAMc guiding-center equations.

This module ports the axisymmetric analytic RHS in RAMc's ``push.cpp`` to
batched JAX arrays. It intentionally remains a geometry-specific backend;
the integrator and collision modules do not depend on it.
"""

from __future__ import annotations

import jax.numpy as jnp

from core.config import OrbitNormalization
from core.math import momentum_from_gamma
from core.state import CircularFieldProfiles, KinematicState, QuadraticCircularFieldProfiles
from fields.circular import sample_circular_field


def ramc_circular_rhs(
    kin: KinematicState,
    time: float,
    profiles: CircularFieldProfiles | QuadraticCircularFieldProfiles,
    norm: OrbitNormalization,
) -> KinematicState:
    """Return the deterministic circular-RAMc characteristics.

    The implementation follows ``RHSFunctionDKEanalytic`` in the supplied
    RAMc source for the axisymmetric, unperturbed magnetic field. Collisional
    friction is excluded because collisions are handled by a separate
    operator. Synchrotron radiation is retained in the deterministic orbit.
    """

    del time
    sample = sample_circular_field(kin, profiles, norm)

    gamma = kin.gamma
    xi = kin.xi
    X = kin.x
    Y = kin.y
    r = sample.r
    rs = sample.r_safe
    R = sample.R
    e1 = sample.e1
    de1_dr = sample.de1_dr
    q = sample.q
    dq_dr = sample.dq_dr

    eps = norm.epsilon
    tauc = norm.c_tau_over_a
    alpha = norm.alpha_syn
    wce0 = norm.a_omega_ce_over_c

    p = momentum_from_gamma(gamma)
    p_safe = jnp.maximum(p, 1.0e-14)
    v = p / jnp.maximum(gamma, 1.0)

    Ephi = e1 / R
    BR = jnp.sqrt(1.0 + r * r * eps * eps / (q * q))
    bphi = 1.0 / BR
    wceB = wce0 * BR / R
    inv_wceB = 1.0 / jnp.maximum(wceB, 1.0e-30)

    dBRdr = (
        r
        * eps
        * eps
        / (BR * q * q)
        * (1.0 - (r / q) * dq_dr)
    )
    BthetaOverB = (r * eps / q) / BR

    dBsubRdY = eps / (R * q) - (Y * Y * eps / (R * rs)) * dq_dr / (q * q)
    dBsubYdR = -eps / (R * R * q) + (X * X * eps / (R * rs)) * dq_dr / (q * q)
    bhat_dot_curl_bhat = (R / BR) ** 2 * (1.0 / R) * (dBsubRdY - dBsubYdR)

    denom = 1.0 - xi * p * inv_wceB * bhat_dot_curl_bhat
    denom = jnp.where(
        jnp.abs(denom) < 1.0e-12,
        jnp.where(denom >= 0.0, 1.0e-12, -1.0e-12),
        denom,
    )
    B_over_Bstar = 1.0 / denom

    # Axisymmetric equilibrium: delta B = 0.
    bdotr = jnp.zeros_like(r)
    bdottheta = BthetaOverB
    bdotphi = 1.0 / BR

    bdot_nabla_lnB = (1.0 / BR) * (eps / q) * (eps / R) * Y

    xhat = X / rs
    yhat = Y / rs
    b_cross_gradlnB_r = -(eps / R) * (1.0 / BR) * yhat
    b_cross_gradlnB_theta = -(eps / R) * (1.0 / BR) * xhat + dBRdr / (BR * BR)
    b_cross_gradlnB_phi = (
        (eps / R) * (1.0 / BR) * X * (eps / q)
        - dBRdr * (r * eps / q) / (BR * BR)
    )

    # Time-dependent-bhat correction from the radial gradient of E1.
    time_bhat_term = jnp.where(
        r > 1.0e-10,
        xi * p * (1.0 / R) * inv_wceB * (q / (eps * rs)) * de1_dr,
        0.0,
    )
    Ephi_star = Ephi + time_bhat_term

    Exb_r = -(1.0 / BR) * (r * eps / q) * Ephi_star
    Exb_theta = jnp.zeros_like(r)
    Exb_phi = jnp.zeros_like(r)

    bdotE = bphi * Ephi
    gradlnB_dot_ExB = (
        ((eps / R) * xhat - dBRdr / BR)
        * (1.0 / BR)
        * (r * eps / q)
        * Ephi_star
    )

    dX_star = (
        -yhat
        * tauc
        * xi
        * v
        * B_over_Bstar
        * BthetaOverB
        * inv_wceB
        * xi
        * p
        * bhat_dot_curl_bhat
    )
    dY_star = (
        xhat
        * tauc
        * xi
        * v
        * B_over_Bstar
        * BthetaOverB
        * inv_wceB
        * xi
        * p
        * bhat_dot_curl_bhat
    )
    dphi_star = (
        -tauc
        * xi
        * v
        * B_over_Bstar
        * BthetaOverB
        * BthetaOverB
        * (1.0 / bphi)
        * inv_wceB
        * xi
        * p
        * bhat_dot_curl_bhat
        * (eps / R)
    )

    B_over_B0 = BR / R

    dgamma = v * (
        -xi * bdotE
        + B_over_Bstar
        * 0.5
        * inv_wceB
        * p
        * (1.0 + xi * xi)
        * gradlnB_dot_ExB
        - B_over_B0 * B_over_B0 * alpha * p * gamma * (1.0 - xi * xi)
        + xi
        * Ephi_star
        * B_over_Bstar
        * BthetaOverB
        * BthetaOverB
        * (1.0 / bphi)
        * inv_wceB
        * xi
        * p
        * bhat_dot_curl_bhat
    )

    dxi = (1.0 - xi * xi) * (
        -bdotE / p_safe
        - B_over_Bstar * 0.5 * tauc * v * bdot_nabla_lnB
        + B_over_Bstar * 0.5 * inv_wceB * xi * gradlnB_dot_ExB
        + B_over_B0 * B_over_B0 * alpha * xi / jnp.maximum(gamma, 1.0)
        + (Ephi_star / p_safe)
        * B_over_Bstar
        * BthetaOverB
        * BthetaOverB
        * (1.0 / bphi)
        * inv_wceB
        * xi
        * p
        * bhat_dot_curl_bhat
    )

    dX = (
        tauc
        * (
            xi * v * (bdotr * xhat - bdottheta * yhat)
            - B_over_Bstar
            * 0.5
            * inv_wceB
            * v
            * p
            * (1.0 + xi * xi)
            * (xhat * b_cross_gradlnB_r - yhat * b_cross_gradlnB_theta)
        )
        + B_over_Bstar * inv_wceB * (Exb_r * xhat - Exb_theta * yhat)
        + dX_star
    )

    dY = (
        tauc
        * (
            xi * v * (bdotr * yhat + bdottheta * xhat)
            - B_over_Bstar
            * 0.5
            * inv_wceB
            * v
            * p
            * (1.0 + xi * xi)
            * (yhat * b_cross_gradlnB_r + xhat * b_cross_gradlnB_theta)
        )
        + B_over_Bstar * inv_wceB * (Exb_r * yhat + Exb_theta * xhat)
        + dY_star
    )

    dphi = (
        tauc
        * (
            xi * v * bdotphi * eps / R
            - B_over_Bstar
            * 0.5
            * inv_wceB
            * v
            * p
            * (1.0 + xi * xi)
            * b_cross_gradlnB_phi
            * eps
            / R
        )
        + B_over_Bstar * inv_wceB * Exb_phi * eps / R
        + dphi_star
    )

    return KinematicState(dgamma, dxi, dX, dY, dphi)
