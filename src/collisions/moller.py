"""Conservative linearized Moller large-angle collision operator.

The operator uses a weighted gain-loss realization. For a marker of weight w,
a fraction q of its weight undergoes a Moller event over the large-angle
interval. The candidate ensemble contains

    uncollided:       w (1-q) at the incoming state
    outgoing primary: w q     at state 3
    outgoing electron:w q     at state 4

which is the Monte Carlo representation of one loss and two gain terms. The
3N weighted candidates are then stratified-randomly thinned back to the fixed N slots.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.precision import index_dtype

from core.config import MollerConfig
from core.constants import PI
from core.math import momentum_from_gamma, safe_radius
from core.rng import uniform_by_particle
from core.state import BackgroundProfiles, KinematicState, ParticleState
from fields.profiles import sample_background
from resampling.multinomial import stratified_resample


def moller_dsigma_dgamma(gamma0, gamma_secondary):
    """Dimensionless differential Moller cross section d(sigma/r_e^2)/dgamma."""

    g0 = gamma0
    gs = gamma_secondary
    nu = (gs - 1.0) / jnp.maximum(g0 - 1.0, 1.0e-30)
    x = 1.0 / jnp.maximum(nu * (1.0 - nu), 1.0e-30)
    pref = 2.0 * PI * g0 * g0 / (
        jnp.maximum((g0 - 1.0) ** 3 * (g0 + 1.0), 1.0e-30)
    )
    return pref * (
        x * x - 3.0 * x + ((g0 - 1.0) / g0) ** 2 * (1.0 + x)
    )


def moller_tail_cross_section(gamma0, gamma_cut):
    """Integrated dimensionless cross section from gamma_cut to (g0+1)/2.

    This is the analytic integral used by RAMc. It is zero when the incoming
    electron does not have enough energy to produce both outgoing kinetic
    electrons above the selected large-angle cutoff.
    """

    g = gamma0
    gc = gamma_cut
    eligible = g > (2.0 * gc - 1.0)
    a = jnp.maximum(g - gc, 1.0e-30)
    b = jnp.maximum(gc - 1.0, 1.0e-30)
    expr = (
        0.5 * (g + 1.0)
        - gc
        - g * g * (1.0 / a - 1.0 / b)
        + ((2.0 * g - 1.0) / jnp.maximum(g - 1.0, 1.0e-30)) * jnp.log(b / a)
    )
    sigma = (2.0 * PI / jnp.maximum(g * g - 1.0, 1.0e-30)) * expr
    return jnp.where(eligible, jnp.maximum(sigma, 0.0), 0.0)


def _sample_secondary_gamma(gamma0, gamma_min, u, n_iter: int):
    """Invert the analytic Moller tail cross section by fixed-count bisection."""

    gmid = 0.5 * (gamma0 + 1.0)
    total = moller_tail_cross_section(gamma0, gamma_min)
    target = (1.0 - jnp.clip(u, 1.0e-12, 1.0 - 1.0e-12)) * total
    lo = jnp.full_like(gamma0, gamma_min)
    hi = gmid

    def body(_i, vals):
        lo_, hi_ = vals
        mid = 0.5 * (lo_ + hi_)
        tail = moller_tail_cross_section(gamma0, mid)
        # tail decreases monotonically with the lower integration limit.
        lo_new = jnp.where(tail > target, mid, lo_)
        hi_new = jnp.where(tail > target, hi_, mid)
        return lo_new, hi_new

    lo, hi = jax.lax.fori_loop(0, n_iter, body, (lo, hi))
    sampled = 0.5 * (lo + hi)
    return jnp.where(total > 0.0, sampled, gamma_min)


def moller_outgoing_pair(gamma0, xi0, gamma_secondary, azimuth_u):
    """Return the two outgoing states for a cold-target Moller event.

    The secondary scattering angle follows the same kinematics that produces
    RAMc's Pi(gamma0, xi0, gamma, xi) distribution. A random azimuth around
    the incident momentum is sampled explicitly. The outgoing primary pitch
    is then reconstructed from parallel momentum conservation.
    """

    gs = gamma_secondary
    gp = gamma0 + 1.0 - gs
    p0 = momentum_from_gamma(gamma0)
    ps = momentum_from_gamma(gs)
    pp = jnp.maximum(momentum_from_gamma(gp), 1.0e-14)

    cos_theta = jnp.sqrt(
        jnp.clip(
            (gamma0 + 1.0)
            * (gs - 1.0)
            / jnp.maximum((gamma0 - 1.0) * (gs + 1.0), 1.0e-30),
            0.0,
            1.0,
        )
    )
    sin_theta = jnp.sqrt(jnp.maximum(1.0 - cos_theta * cos_theta, 0.0))
    chi = 2.0 * PI * azimuth_u
    xi_s = (
        xi0 * cos_theta
        + jnp.sqrt(jnp.maximum(1.0 - xi0 * xi0, 0.0)) * sin_theta * jnp.cos(chi)
    )
    xi_s = jnp.clip(xi_s, -1.0, 1.0)

    xi_p = (p0 * xi0 - ps * xi_s) / pp
    xi_p = jnp.clip(xi_p, -1.0, 1.0)
    return gp, xi_p, gs, xi_s


def collision_fraction(
    particles: ParticleState,
    background: BackgroundProfiles,
    dt_large_angle: float,
    config: MollerConfig,
):
    r, _ = safe_radius(particles.kin.x, particles.kin.y)
    ne, _te, _ti, _zeff, _eta = sample_background(r, background)
    ne_norm = ne / config.ne0_cm3
    v = momentum_from_gamma(particles.kin.gamma) / jnp.maximum(particles.kin.gamma, 1.0)
    sigma = moller_tail_cross_section(particles.kin.gamma, config.gamma_min)
    q = (
        ne_norm
        * config.target_electron_factor
        * dt_large_angle
        * v
        * sigma
        / (4.0 * PI * config.coulog0)
    )
    q = jnp.where(particles.alive, q, 0.0)
    return q


def gain_loss_candidates(
    particles: ParticleState,
    background: BackgroundProfiles,
    dt_large_angle: float,
    base_key,
    global_step: int,
    config: MollerConfig,
):
    """Construct the fixed-shape 3N weighted gain-loss candidate ensemble."""

    q_raw = collision_fraction(particles, background, dt_large_angle, config)
    q = jnp.clip(q_raw, 0.0, config.max_collision_fraction)

    u_g = uniform_by_particle(base_key, particles.pid, global_step, stream=21)
    u_a = uniform_by_particle(base_key, particles.pid, global_step, stream=22)
    gs = _sample_secondary_gamma(
        particles.kin.gamma, config.gamma_min, u_g, config.bisection_steps
    )
    gp, xi_p, gs, xi_s = moller_outgoing_pair(
        particles.kin.gamma, particles.kin.xi, gs, u_a
    )

    kin0 = particles.kin
    kinp = KinematicState(gp, xi_p, kin0.x, kin0.y, kin0.phi)
    kins = KinematicState(gs, xi_s, kin0.x, kin0.y, kin0.phi)

    w = jnp.where(particles.alive, particles.weight, 0.0)
    w0 = w * (1.0 - q)
    wp = w * q
    ws = w * q

    def cat(a, b, c):
        return jnp.concatenate([a, b, c], axis=0)

    kin = KinematicState(
        cat(kin0.gamma, kinp.gamma, kins.gamma),
        cat(kin0.xi, kinp.xi, kins.xi),
        cat(kin0.x, kinp.x, kins.x),
        cat(kin0.y, kinp.y, kins.y),
        cat(kin0.phi, kinp.phi, kins.phi),
    )
    wc = cat(w0, wp, ws)
    alive = wc > 0.0
    pid = jnp.arange(wc.shape[0], dtype=index_dtype())
    candidates = ParticleState(kin, wc, alive, pid)
    return candidates, q_raw



def source_only_candidates(
    particles: ParticleState,
    background: BackgroundProfiles,
    dt_large_angle: float,
    base_key,
    global_step: int,
    config: MollerConfig,
):
    """Construct the fixed-shape 2N source-only Moller candidate ensemble.

    This is the conventional avalanche source approximation used for the
    Appendix-B benchmarks of McDevitt, Guo & Tang (PPCF 61, 054008, 2019).
    The incoming primary remains at its original state while a secondary of
    expected weight ``w*q`` is added.  The candidate ensemble is subsequently
    randomly thinned to the production fixed-N marker count.
    """

    q_raw = collision_fraction(particles, background, dt_large_angle, config)
    q = jnp.clip(q_raw, 0.0, config.max_collision_fraction)

    u_g = uniform_by_particle(base_key, particles.pid, global_step, stream=31)
    u_a = uniform_by_particle(base_key, particles.pid, global_step, stream=32)
    gs = _sample_secondary_gamma(
        particles.kin.gamma, config.gamma_min, u_g, config.bisection_steps
    )
    _gp, _xi_p, gs, xi_s = moller_outgoing_pair(
        particles.kin.gamma, particles.kin.xi, gs, u_a
    )

    kin0 = particles.kin
    kins = KinematicState(gs, xi_s, kin0.x, kin0.y, kin0.phi)
    w = jnp.where(particles.alive, particles.weight, 0.0)
    w0 = w
    ws = w * q

    def cat(a, b):
        return jnp.concatenate([a, b], axis=0)

    kin = KinematicState(
        cat(kin0.gamma, kins.gamma),
        cat(kin0.xi, kins.xi),
        cat(kin0.x, kins.x),
        cat(kin0.y, kins.y),
        cat(kin0.phi, kins.phi),
    )
    wc = cat(w0, ws)
    alive = wc > 0.0
    pid = jnp.arange(wc.shape[0], dtype=index_dtype())
    return ParticleState(kin, wc, alive, pid), q_raw


def apply_moller_source_only(
    particles: ParticleState,
    background: BackgroundProfiles,
    dt_large_angle: float,
    base_key,
    global_step: int,
    config: MollerConfig,
):
    """Apply the conventional source-only Moller model and thin to N."""

    candidates, q_raw = source_only_candidates(
        particles, background, dt_large_angle, base_key, global_step, config
    )
    n = particles.weight.shape[0]
    resample_key = jax.random.fold_in(base_key, jnp.asarray(global_step, dtype=jnp.uint32))
    resample_key = jax.random.fold_in(resample_key, jnp.uint32(33))
    out = stratified_resample(candidates, resample_key, n)
    return out, jnp.max(q_raw)

def apply_moller_gain_loss(
    particles: ParticleState,
    background: BackgroundProfiles,
    dt_large_angle: float,
    base_key,
    global_step: int,
    config: MollerConfig,
):
    """Apply the weighted conservative gain-loss model and thin back to N."""

    candidates, q_raw = gain_loss_candidates(
        particles, background, dt_large_angle, base_key, global_step, config
    )
    n = particles.weight.shape[0]
    resample_key = jax.random.fold_in(base_key, jnp.asarray(global_step, dtype=jnp.uint32))
    resample_key = jax.random.fold_in(resample_key, jnp.uint32(23))
    out = stratified_resample(candidates, resample_key, n)
    return out, jnp.max(q_raw)
