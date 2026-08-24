"""Maxwellian-background test-particle small-angle collisions.

The default implementation follows the fully ionized RAMc Monte Carlo model:
pitch-angle scattering, collisional friction, and energy diffusion are
applied to a fixed marker ensemble.  The pitch-angle operator can optionally
use the partially screened Hesslow/RAMc coefficients employed in McDevitt,
Guo & Tang, PPCF 61, 024004 (2019).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
from jax.scipy.special import erf

from core.config import SmallAngleConfig
from core.constants import ALPHA_FS, ME_C2_EV, PI
from core.math import momentum_from_gamma, reflect_gamma, reflect_pitch, safe_radius
from core.rng import normal_by_particle
from core.state import BackgroundProfiles, KinematicState, ParticleState
from fields.profiles import sample_background

from .coulomb import chandrasekhar, d_chandrasekhar_dx, thermal_coulomb_log, thermal_speed_over_c


def _log1p_power(base, exponent):
    """Stable ``log(1 + base**exponent)`` for positive ``base``."""

    log_power = exponent * jnp.log(jnp.maximum(base, 1.0e-300))
    return jnp.logaddexp(0.0, log_power)


def relativistic_coulomb_logs(gamma, te, coulog, k: float = 5.0):
    """Energy-dependent electron-electron/electron-ion Coulomb logs.

    These are Eqs. (13a,b) of McDevitt, Guo & Tang, PPCF 61, 054008
    (2019).  ``coulog`` is the thermal logarithm ``ln Lambda_0`` and
    ``vTe/c`` is inferred from ``te``.
    """

    p = momentum_from_gamma(gamma)
    vte = thermal_speed_over_c(te)
    log_ee = coulog + (1.0 / k) * _log1p_power(
        2.0 * (gamma - 1.0) / (vte * vte), k / 2.0
    )
    log_ei = coulog + (1.0 / k) * _log1p_power(2.0 * p / vte, k)
    return log_ee, log_ei


def reduced_large_angle_coulomb_log(te, coulog, gamma_min):
    """McDevitt-2019 Eq. (32) Coulomb log for the conservative split."""

    vte = thermal_speed_over_c(te)
    return coulog + jnp.log(
        jnp.maximum(2.0 * jnp.sqrt(jnp.maximum(gamma_min - 1.0, 1.0e-30)) / vte, 1.0e-30)
    )


def source_only_large_angle_coulomb_log(gamma, te, coulog):
    """McDevitt-2019 Eq. (33) Coulomb log for source-only avalanche runs."""

    vte = thermal_speed_over_c(te)
    return coulog + jnp.log(
        jnp.maximum(
            jnp.sqrt(2.0 * jnp.maximum(gamma - 1.0, 1.0e-30)) / vte,
            1.0e-30,
        )
    )




def partial_screening_friction_coefficient(
    gamma, ne_norm, te, coulog, config: SmallAngleConfig
):
    """RAMc/Hesslow partially screened friction coefficient ``C_F``.

    This ports ``ComputeFrictionHesslow`` from the supplied RAMc source.
    The impurity mean excitation energy is supplied in eV and internally
    normalized to ``m_e c^2``.
    """

    p = momentum_from_gamma(gamma)
    # All coefficient evaluations must use the same resolved lower momentum
    # as the timestep bound.  Using an independent machine-scale floor here
    # makes the energy-diffusion drift singular below ``config.p_min`` while
    # the host-side subcycle estimate assumes a finite resolved frequency.
    p_safe = jnp.maximum(p, config.p_min)
    v = p_safe / jnp.maximum(gamma, 1.0)
    vte = thermal_speed_over_c(te)
    x = v / vte
    psi = chandrasekhar(x)
    log_ee, _log_ei = relativistic_coulomb_logs(
        gamma, te, coulog, config.screening_k
    )

    n_bound = config.impurity_nuclear_charge - config.impurity_charge_state
    f_imp = config.impurity_fraction
    impurity_weight = f_imp / (1.0 + config.impurity_charge_state * f_imp)
    excitation = jnp.maximum(
        config.impurity_mean_excitation_ev / ME_C2_EV, 1.0e-30
    )
    h_i = p_safe * jnp.sqrt(jnp.maximum(gamma - 1.0, 0.0)) / excitation
    bound_drag = n_bound * impurity_weight * (
        _log1p_power(h_i, config.screening_k) / config.screening_k - v * v
    )
    return (
        ne_norm
        * 2.0
        * psi
        / (vte * vte)
        * (log_ee / config.coulog0)
        * (1.0 + bound_drag / jnp.maximum(log_ee, 1.0e-30))
    )

def partial_screening_g(p, config: SmallAngleConfig):
    """Return the smooth RAMc/Hesslow partial-screening function ``g_I(p)``.

    This is the form used in the supplied RAMc implementation.  At large
    momentum it asymptotes to the expression quoted as Eq. (6) of McDevitt
    et al. (2019).
    """

    z0 = config.impurity_nuclear_charge
    zi = config.impurity_charge_state
    n_bound = z0 - zi
    y = 2.0 * config.impurity_radius_abohr * p / ALPHA_FS
    y32 = jnp.power(jnp.maximum(y, 0.0), 1.5)
    return (
        (2.0 / 3.0) * (z0 * z0 - zi * zi) * jnp.log1p(y32)
        - (2.0 / 3.0) * n_bound * n_bound * y32 / (1.0 + y32)
    )


def pitch_scattering_frequency(
    gamma,
    zeff,
    ne_norm,
    te,
    coulog,
    config: SmallAngleConfig,
):
    """Dimensionless deflection frequency ``tau_c0 * nu_D``.

    The fully ionized branch reproduces the default RAMc operator.  With
    ``config.partial_screening`` enabled, the ion term, energy-dependent
    electron-ion/electron-electron Coulomb logarithms, and impurity screening
    correction follow the partially screened RAMc implementation used for
    the McDevitt et al. Figure-6 transport benchmark.
    """

    p = momentum_from_gamma(gamma)
    # The test-particle coefficients are singular at p=0.  The configured
    # lower momentum is the resolved kinetic-domain boundary; it must be used
    # consistently when estimating the collision frequency.
    p_safe = jnp.maximum(p, config.p_min)
    v = p / jnp.maximum(gamma, 1.0)
    v_safe = jnp.maximum(v, 1.0e-14)
    vte = thermal_speed_over_c(te)
    x = v / vte
    psi = chandrasekhar(x)
    electron_term = erf(x) - psi + 0.5 * vte**4 * x * x

    def reduced_log(_):
        # Only electron-electron collisions are split between the
        # Fokker-Planck and Moller operators. Electron-ion scattering retains
        # its full relativistic Coulomb logarithm.
        log_red = reduced_large_angle_coulomb_log(
            te, coulog, config.large_angle_gamma_min
        )
        _log_ee, log_ei = relativistic_coulomb_logs(
            gamma, te, coulog, config.screening_k
        )
        return ne_norm * gamma * (
            zeff * (log_ei / config.coulog0)
            + electron_term * (log_red / config.coulog0)
        ) / (p_safe**3)

    def source_log(_):
        log_src = source_only_large_angle_coulomb_log(gamma, te, coulog)
        _log_ee, log_ei = relativistic_coulomb_logs(
            gamma, te, coulog, config.screening_k
        )
        return ne_norm * gamma * (
            zeff * (log_ei / config.coulog0)
            + electron_term * (log_src / config.coulog0)
        ) / (p_safe**3)

    def relativistic_log(_):
        log_ee, log_ei = relativistic_coulomb_logs(
            gamma, te, coulog, config.screening_k
        )
        return ne_norm * gamma * (
            zeff * (log_ei / config.coulog0)
            + electron_term * (log_ee / config.coulog0)
        ) / (p_safe**3)

    def thermal_log(_):
        log_ratio = coulog / config.coulog0
        return ne_norm * log_ratio * gamma * (zeff + electron_term) / (p_safe**3)

    def fully_ionized(_):
        # Configuration switches can be tracers when the caller JIT-compiles
        # the whole collision loop, so branch only through JAX control flow.
        return jax.lax.cond(
            jnp.asarray(config.large_angle_reduced_coulog),
            reduced_log,
            lambda __: jax.lax.cond(
                jnp.asarray(config.large_angle_source_coulog),
                source_log,
                lambda ___: jax.lax.cond(
                    jnp.asarray(config.relativistic_coulog),
                    relativistic_log,
                    thermal_log,
                    operand=None,
                ),
                operand=None,
            ),
            operand=None,
        )

    def partially_screened(_):
        # f_I = n_I/n_D and f_I/(1+Z_I f_I) = n_I/n_e for a deuterium
        # main-ion background.
        k = config.screening_k
        log_ee, log_ei = relativistic_coulomb_logs(gamma, te, coulog, k)
        f_imp = config.impurity_fraction
        zi = config.impurity_charge_state
        screening = partial_screening_g(p_safe, config)
        impurity_weight = f_imp / (1.0 + zi * f_imp)
        ion_term = (log_ei / config.coulog0) * zeff * (
            1.0
            + impurity_weight
            * screening
            / (jnp.maximum(zeff, 1.0e-14) * log_ei)
        )
        ee_term = (log_ee / config.coulog0) * electron_term
        cb_screened = (ion_term + ee_term) / v_safe
        return ne_norm * cb_screened / (p_safe * p_safe)

    return jax.lax.cond(
        jnp.asarray(config.partial_screening),
        partially_screened,
        fully_ionized,
        operand=None,
    )


def small_angle_step(
    particles: ParticleState,
    background: BackgroundProfiles,
    dt: float,
    base_key,
    global_step: int,
    config: SmallAngleConfig,
) -> ParticleState:
    """Advance the default small-angle collision operator by ``dt/tau_c0``."""

    kin = particles.kin
    gamma = kin.gamma
    xi = reflect_pitch(kin.xi)
    r, _ = safe_radius(kin.x, kin.y)

    ne, te, _ti, zeff, _eta = sample_background(r, background)
    coulog = thermal_coulomb_log(ne, te)
    ne_norm = ne / config.ne0_cm3
    # Configuration objects are JAX pytrees when passed through a compiled
    # outer kernel, so their boolean switches may be tracers. Keep all branch
    # selection inside JAX control flow rather than Python ``if`` statements.
    log_friction = jax.lax.cond(
        jnp.asarray(config.large_angle_reduced_coulog),
        lambda _: reduced_large_angle_coulomb_log(
            te, coulog, config.large_angle_gamma_min
        ),
        lambda _: jax.lax.cond(
            jnp.asarray(config.large_angle_source_coulog),
            lambda _: source_only_large_angle_coulomb_log(gamma, te, coulog),
            lambda _: jax.lax.cond(
                jnp.asarray(config.relativistic_coulog),
                lambda _: relativistic_coulomb_logs(
                    gamma, te, coulog, config.screening_k
                )[0],
                lambda _: coulog,
                operand=None,
            ),
            operand=None,
        ),
        operand=None,
    )
    log_ratio = log_friction / config.coulog0

    p = momentum_from_gamma(gamma)
    p_safe = jnp.maximum(p, 1.0e-14)
    v = p / jnp.maximum(gamma, 1.0)
    vte = thermal_speed_over_c(te)
    x = v / vte
    psi = chandrasekhar(x)

    active = particles.alive & (particles.weight > 0.0)

    # Pitch-angle scattering frequency.  The optional partially screened
    # branch is used by the McDevitt Figure-6 transport benchmark.
    nu_d = pitch_scattering_frequency(
        gamma,
        zeff,
        ne_norm,
        te,
        coulog,
        config,
    )
    # Do not clip nu*dt.  Clipping changes the pitch operator while the
    # friction and energy-diffusion terms still use the requested dt.  Callers
    # must subcycle until this increment is below the configured N_SA target.
    nu_dt = nu_d * dt
    z_pitch = normal_by_particle(base_key, particles.pid, global_step, stream=11)
    sigma_xi = jnp.sqrt(jnp.maximum((1.0 - xi * xi) * nu_dt, 0.0))
    xi_scattered = reflect_pitch(xi * (1.0 - nu_dt) + z_pitch * sigma_xi)
    xi_new = jnp.where(config.pitch_scattering & active, xi_scattered, xi)

    # Friction.  The partially screened branch ports RAMc's
    # ComputeFrictionHesslow; the fully ionized branch is 2*Psi/vTe^2.
    friction = jax.lax.cond(
        jnp.asarray(config.partial_screening),
        lambda _: partial_screening_friction_coefficient(
            gamma, ne_norm, te, coulog, config
        ),
        lambda _: ne_norm * log_ratio * 2.0 * psi / (vte * vte),
        operand=None,
    )
    dgamma_drag = -v * friction * dt

    # Energy diffusion and Ito drift.  RAMc's Hesslow branch uses the
    # energy-dependent electron-electron Coulomb logarithm, without the
    # inelastic bound-electron drag correction.
    log_energy = jax.lax.cond(
        jnp.asarray(config.partial_screening),
        lambda _: relativistic_coulomb_logs(
            gamma, te, coulog, config.screening_k
        )[0],
        lambda _: log_friction,
        operand=None,
    )
    energy_log_ratio = log_energy / config.coulog0
    dpsi_dgamma = d_chandrasekhar_dx(x) / (vte * p_safe * gamma * gamma)
    gamma_conv = ne_norm * energy_log_ratio * (2.0 * psi / p_safe + v * dpsi_dgamma)
    sigma_gamma = jnp.sqrt(jnp.maximum(ne_norm * energy_log_ratio * 2.0 * v * psi * dt, 0.0))
    z_energy = normal_by_particle(base_key, particles.pid, global_step, stream=12)

    gamma_new = gamma
    gamma_new = jnp.where(config.friction & active, gamma_new + dgamma_drag, gamma_new)
    gamma_new = jnp.where(
        config.energy_scattering & active,
        gamma_new + gamma_conv * dt + z_energy * sigma_gamma,
        gamma_new,
    )
    # Default retains RAMc's reflecting gamma=1 boundary.  A configured
    # thermal floor instead clips into the reservoir cell; the caller then
    # performs Maxwellian re-entry without artificial energy reflection.
    gamma_new = jnp.where(
        jnp.asarray(config.gamma_floor) > 1.0,
        jnp.maximum(gamma_new, config.gamma_floor),
        reflect_gamma(gamma_new),
    )

    phi_new = jnp.mod(kin.phi, 2.0 * PI)
    new_kin = KinematicState(gamma_new, xi_new, kin.x, kin.y, phi_new)
    return ParticleState(new_kin, particles.weight, particles.alive, particles.pid)


def required_small_angle_substeps(
    dt: float,
    background: BackgroundProfiles,
    config: SmallAngleConfig,
) -> int:
    """Return a static substep count satisfying the N_SA collision target.

    The bound is evaluated at the configured lowest resolved momentum and at
    every supplied background profile point.  It is intentionally host-side:
    accelerator kernels receive the resulting fixed loop count.
    """

    if dt <= 0.0:
        raise ValueError("dt must be positive")
    p_floor = max(
        float(config.p_min),
        math.sqrt(max(float(config.gamma_floor) ** 2 - 1.0, 0.0)),
    )
    gamma_floor = math.sqrt(1.0 + p_floor * p_floor)
    gamma = jnp.full_like(background.te_ev, gamma_floor)
    ne_norm = background.ne_cm3 / config.ne0_cm3
    coulog = thermal_coulomb_log(background.ne_cm3, background.te_ev)
    nu = pitch_scattering_frequency(
        gamma,
        background.zeff,
        ne_norm,
        background.te_ev,
        coulog,
        config,
    )
    max_nu = float(jax.device_get(jnp.max(nu)))
    return max(1, int(math.ceil(float(dt) * max_nu * int(config.n_sa))))


def small_angle_subcycle(
    particles: ParticleState,
    background: BackgroundProfiles,
    dt: float,
    base_key,
    global_step: int,
    config: SmallAngleConfig,
    n_substeps: int,
) -> ParticleState:
    """Apply the small-angle operator with a fixed JAX substep count."""

    if isinstance(n_substeps, int) and n_substeps < 1:
        raise ValueError("n_substeps must be positive")
    dt_sub = dt / n_substeps

    def body(index, state):
        return small_angle_step(
            state,
            background,
            dt_sub,
            base_key,
            global_step * n_substeps + index,
            config,
        )

    return jax.lax.fori_loop(0, n_substeps, body, particles)
