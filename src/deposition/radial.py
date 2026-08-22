"""GPU-friendly radial binning and RAMc current deposition."""

from __future__ import annotations

import jax.numpy as jnp

from core.constants import ALFVEN_CURRENT_A, ME_C2_EV, PI
from core.math import momentum_from_gamma, safe_radius
from core.state import BackgroundProfiles, ParticleState
from fields.profiles import interp1

# Exact numerical value used in the supplied RAMc current-deposition routine.
_RAMC_EC_ESU = 4.8032e-9


def linear_bin_sum(r, values, grid):
    """Cloud-in-cell deposition onto a uniform radial grid."""

    dr = grid[1] - grid[0]
    u = (r - grid[0]) / dr
    i0 = jnp.floor(u).astype(jnp.int32)
    i0 = jnp.clip(i0, 0, grid.shape[0] - 2)
    frac = jnp.clip(u - i0, 0.0, 1.0)
    i1 = i0 + 1
    out0 = jnp.bincount(i0, weights=values * (1.0 - frac), length=grid.shape[0])
    out1 = jnp.bincount(i1, weights=values * frac, length=grid.shape[0])
    return out0 + out1


def deposit_weight_density(particles: ParticleState, grid):
    r, _ = safe_radius(particles.kin.x, particles.kin.y)
    w = jnp.where(particles.alive, particles.weight, 0.0)
    return linear_bin_sum(r, w, grid)


def deposit_ramc_parallel_current(
    particles: ParticleState,
    grid,
    q_profile,
    background: BackgroundProfiles,
    epsilon: float,
    a_minor_cm: float,
    a_runaway: float = 0.0,
):
    """Deposit RAMc's dimensionless parallel current ``a^2 j_parallel / I_A``.

    This ports the linear radial weighting in ``DepositCurrent`` from the
    supplied RAMc source. Marker ``weight`` is interpreted as represented
    physical electron number. If ``a_runaway > 0``, only markers satisfying
    ``gamma >= 1 + a_runaway*T_e/(m_e c^2)`` contribute.
    """

    kin = particles.kin
    r, _ = safe_radius(kin.x, kin.y)
    dr = grid[1] - grid[0]
    n = grid.shape[0]
    u = (r - grid[0]) / dr
    i0 = jnp.clip(jnp.floor(u).astype(jnp.int32), 0, n - 2)
    i1 = i0 + 1
    frac = jnp.clip(u - i0, 0.0, 1.0)

    ri = grid[i0]
    rip1 = grid[i1]
    qi = q_profile[i0]
    qip1 = q_profile[i1]
    BRi = jnp.sqrt(1.0 + ri * ri * epsilon * epsilon / (qi * qi))
    BRip1 = jnp.sqrt(1.0 + rip1 * rip1 * epsilon * epsilon / (qip1 * qip1))

    major_over_a = 1.0 / epsilon
    jac_i = 4.0 * PI * PI * major_over_a * jnp.maximum(ri, dr) * dr
    jac_ip1 = 4.0 * PI * PI * major_over_a * rip1 * dr
    jac_ip1 = jnp.where(i1 == n - 1, 0.5 * jac_ip1, jac_ip1)

    R = 1.0 + epsilon * kin.x
    v = momentum_from_gamma(kin.gamma) / jnp.maximum(kin.gamma, 1.0)
    te = interp1(r, background.r, background.te_ev)
    gamma_ra = 1.0 + a_runaway * te / ME_C2_EV
    selected = particles.alive & (particles.weight > 0.0)
    selected = selected & ((a_runaway <= 0.0) | (kin.gamma >= gamma_ra))
    w = jnp.where(selected, particles.weight, 0.0)

    pref = -(_RAMC_EC_ESU / a_minor_cm) / ALFVEN_CURRENT_A
    value = pref * v * kin.xi * w
    contrib_i = value * (R / BRi) * (1.0 - frac) / jac_i
    contrib_ip1 = value * (R / BRip1) * frac / jac_ip1
    # Legacy RAMc suppresses direct deposition to r=0 and imposes j(0)=j(1).
    contrib_i = jnp.where(i0 == 0, 0.0, contrib_i)

    out = jnp.bincount(i0, weights=contrib_i, length=n)
    out = out + jnp.bincount(i1, weights=contrib_ip1, length=n)
    out = out.at[0].set(out[1])
    return out
