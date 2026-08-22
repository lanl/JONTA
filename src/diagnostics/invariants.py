"""Deterministic-orbit invariants used for verification and convergence tests.

The routines here are diagnostics only; they do not participate in the particle
push.  Keeping invariant evaluation independent from the integrator makes the
convergence tests useful for detecting errors in either the orbit equations or
the time integrator.
"""

from __future__ import annotations

import jax.numpy as jnp

from core.config import OrbitNormalization
from core.math import momentum_from_gamma
from core.state import CircularFieldProfiles, KinematicState
from fields.circular import sample_circular_field


def _cumulative_trapezoid(y, x):
    """Cumulative trapezoidal integral with zero at the first grid point."""

    increments = 0.5 * (y[1:] + y[:-1]) * (x[1:] - x[:-1])
    return jnp.concatenate((jnp.zeros((1,), dtype=y.dtype), jnp.cumsum(increments)))


def circular_poloidal_flux_integral(r, profiles: CircularFieldProfiles):
    r"""Return :math:`\int_0^r r'/q(r')\,dr'` for the circular equilibrium.

    The supplied profile grid is used directly.  This is exact for constant
    ``q`` and converges with profile resolution for a general tabulated ``q``.
    """

    grid = profiles.r
    integrand = grid / profiles.q
    primitive = _cumulative_trapezoid(integrand, grid)

    # Integrate the piecewise-linear representation through the partial cell
    # containing r.  Interpolating the already-integrated primitive would
    # introduce an avoidable O(dr_grid^2) diagnostic error (and would not be
    # exact even for constant q).
    rc = jnp.clip(r, grid[0], grid[-1])
    idx = jnp.searchsorted(grid, rc, side="right") - 1
    idx = jnp.clip(idx, 0, grid.size - 2)
    x0 = grid[idx]
    x1 = grid[idx + 1]
    f0 = integrand[idx]
    f1 = integrand[idx + 1]
    frac = (rc - x0) / (x1 - x0)
    fr = f0 + frac * (f1 - f0)
    return primitive[idx] + 0.5 * (f0 + fr) * (rc - x0)


def toroidal_canonical_momentum_circular(
    kin: KinematicState,
    time: float,
    profiles: CircularFieldProfiles,
    norm: OrbitNormalization,
):
    r"""Dimensionless toroidal canonical momentum for the RAMc circular model.

    This evaluates Eq. (17) of the supplied RAMc documentation in the
    axisymmetric, unperturbed limit, normalized by :math:`R_0 m_e c`:

    .. math::

       \bar P_\phi = \frac{R}{R_0}\frac{B_\phi}{B}\,\xi p
       + \frac{a\omega_{ce0}}{c}\frac{a}{R_0}
         \int_0^{r/a}\frac{r'}{q(r')}dr'
       + E_1\,t.

    Here ``p`` is normalized by ``m_e c``, ``time`` by ``tau_c``, and ``E1``
    by the Connor--Hastie field.  The final term is essential when a finite
    inductive electric field is present.
    """

    sample = sample_circular_field(kin, profiles, norm)
    p = momentum_from_gamma(kin.gamma)

    kinetic = sample.R * (sample.B_phi / sample.B_mag) * kin.xi * p
    equilibrium_flux = (
        norm.a_omega_ce_over_c
        * norm.epsilon
        * circular_poloidal_flux_integral(sample.r, profiles)
    )
    inductive_flux = sample.e1 * time
    return kinetic + equilibrium_flux + inductive_flux


def magnetic_moment_circular(
    kin: KinematicState,
    profiles: CircularFieldProfiles,
    norm: OrbitNormalization,
):
    r"""Dimensionless magnetic moment for the circular guiding-center model.

    RAMc normalizes the magnetic moment by :math:`m_e c^2/B_0`. In the
    :math:`(\gamma,\xi)` coordinates used by JONTA, the corresponding
    dimensionless invariant is

    .. math::

       \bar\mu = \frac{p^2(1-\xi^2)}{2\,\bar B},

    where :math:`p` is normalized by :math:`m_e c` and
    :math:`\bar B=B/B_0`. The guiding-center equations satisfy
    :math:`d\mu/dt=0`; any drift in this diagnostic therefore measures the
    deterministic-orbit integration error (with collisions and radiation
    disabled).
    """

    sample = sample_circular_field(kin, profiles, norm)
    p = momentum_from_gamma(kin.gamma)
    p_perp_sq = p * p * (1.0 - kin.xi * kin.xi)
    return 0.5 * p_perp_sq / jnp.maximum(sample.B_mag, 1.0e-30)
