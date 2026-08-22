"""JAX pytree-compatible state containers.

NamedTuple is used deliberately: it is transparent to humans/LLM agents,
requires no custom pytree registration, and works naturally with JAX.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

Array = jnp.ndarray


class KinematicState(NamedTuple):
    """Per-marker phase-space coordinates.

    All fields are arrays with leading particle dimension ``N``.

    gamma
        Lorentz factor.
    xi
        Pitch ``p_parallel / p``. RAMc sign convention is retained:
        positive E_parallel accelerates electrons with xi < 0.
    x, y
        Poloidal-plane coordinates normalized to minor radius ``a``.
    phi
        Toroidal angle in radians.
    """

    gamma: Array
    xi: Array
    x: Array
    y: Array
    phi: Array


class ParticleState(NamedTuple):
    """Fixed-size weighted marker ensemble."""

    kin: KinematicState
    weight: Array
    alive: Array
    pid: Array


class BackgroundProfiles(NamedTuple):
    """Radial background-plasma profiles.

    ``r`` is normalized to minor radius. Density is in cm^-3 and
    temperatures are in eV to match the legacy RAMc collision formulas.
    ``eta_bar`` is the dimensionless resistivity used by the RAMc field
    equation.
    """

    r: Array
    ne_cm3: Array
    te_ev: Array
    ti_ev: Array
    zeff: Array
    eta_bar: Array


class CircularFieldProfiles(NamedTuple):
    """Tabulated radial profiles for the circular RAMc orbit model."""

    r: Array
    e1: Array
    q: Array


class QuadraticCircularFieldProfiles(NamedTuple):
    """Fast analytic circular-field profiles used by RAMc-style benchmarks.

    The safety factor is ``q(r)=q0+q2*r**2`` and the normalized inductive
    electric-field amplitude is spatially constant. This is the equilibrium
    used in the published RAMc circular-geometry benchmark suite and avoids
    introducing interpolation work/error when the underlying profiles are
    analytic.
    """

    e1: Array
    q0: Array
    q2: Array


class FieldHistory(NamedTuple):
    """Two-step history required by BDF2 electric-field coupling."""

    e1_n: Array
    e1_nm1: Array
    jkin_n: Array
    jkin_nm1: Array
    eta_n: Array
    eta_nm1: Array


class ChargeStateHistory(NamedTuple):
    """Charge-state populations for BDF2 evolution.

    Arrays use shape ``(..., Z + 1)`` with charge state on the last axis.
    """

    n_n: Array
    n_nm1: Array
