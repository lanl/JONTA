import jax
import jax.numpy as jnp

from collisions.coulomb import thermal_coulomb_log
from collisions.small_angle import (
    partial_screening_friction_coefficient,
    pitch_scattering_frequency,
    reduced_large_angle_coulomb_log,
    relativistic_coulomb_logs,
    required_small_angle_substeps,
    small_angle_step,
    source_only_large_angle_coulomb_log,
)
from core.config import SmallAngleConfig
from core.constants import ME_C2_EV
from core.initialization import particles_from_arrays
from core.math import momentum_from_gamma
from core.state import BackgroundProfiles


def _background():
    r = jnp.linspace(0.0, 1.0, 8)
    return BackgroundProfiles(
        r,
        jnp.full_like(r, 1.0e14),
        jnp.full_like(r, 1.0e3),
        jnp.full_like(r, 1.0e3),
        jnp.ones_like(r),
        jnp.ones_like(r),
    )


def test_disabled_small_angle_is_identity_except_phi_wrap():
    p = particles_from_arrays([2.0], [0.3], [0.2], [0.1], [7.0], [1.0])
    cfg = SmallAngleConfig(1e14, 15.0, False, False, False)
    out = small_angle_step(p, _background(), 1e-4, jax.random.key(0), 0, cfg)
    assert jnp.allclose(out.kin.gamma, p.kin.gamma)
    assert jnp.allclose(out.kin.xi, p.kin.xi)


def test_small_angle_stays_in_physical_domain():
    p = particles_from_arrays([1.01, 2.0, 5.0], [0.99, -0.5, 0.2], [0.1]*3, [0.0]*3, [0.0]*3, [1.0]*3)
    cfg = SmallAngleConfig(1e14, 15.0)
    out = small_angle_step(p, _background(), 1e-5, jax.random.key(1), 4, cfg)
    assert bool(jnp.all(out.kin.gamma >= 1.0))
    assert bool(jnp.all(jnp.abs(out.kin.xi) <= 1.0))
    assert bool(jnp.all(jnp.isfinite(out.kin.gamma)))


def test_collision_resolution_uses_configured_lower_momentum():
    background = _background()
    cfg = SmallAngleConfig(1.0e14, 15.0, n_sa=100, p_min=1.0e-3)
    gamma = jnp.asarray([1.0], dtype=jnp.float64)
    nu = pitch_scattering_frequency(
        gamma,
        jnp.ones_like(gamma),
        jnp.ones_like(gamma),
        jnp.asarray([1.0e3]),
        jnp.asarray([15.0]),
        cfg,
    )
    assert bool(jnp.all(jnp.isfinite(nu)))
    substeps = required_small_angle_substeps(1.0e-4, background, cfg)
    assert substeps >= 1


def test_partial_screening_enhances_relativistic_pitch_scattering():
    te_ev = 0.5 * ME_C2_EV * 0.0063**2
    ne_cm3 = 9.707456085827e13
    coulog = float(thermal_coulomb_log(ne_cm3, te_ev))
    gamma = jnp.asarray([1.0 + 5.0e5 / ME_C2_EV, 1.0 + 1.5e6 / ME_C2_EV])
    common = dict(
        ne0_cm3=ne_cm3,
        coulog0=coulog,
        pitch_scattering=True,
        friction=False,
        energy_scattering=False,
    )
    base = SmallAngleConfig(**common)
    screened = SmallAngleConfig(
        **common,
        partial_screening=True,
        impurity_fraction=0.1,
        impurity_nuclear_charge=18.0,
        impurity_charge_state=1.0,
        impurity_radius_abohr=0.329,
        screening_k=5.0,
    )
    zeff = jnp.ones_like(gamma)
    ne_norm = jnp.ones_like(gamma)
    te = jnp.full_like(gamma, te_ev)
    log = jnp.full_like(gamma, coulog)
    nu_base = pitch_scattering_frequency(gamma, zeff, ne_norm, te, log, base)
    nu_screened = pitch_scattering_frequency(gamma, zeff, ne_norm, te, log, screened)
    assert bool(jnp.all(nu_screened > 5.0 * nu_base))


def test_large_angle_split_reduces_only_electron_electron_pitch_scattering():
    """The FP/Boltzmann split must not remove electron-ion scattering.

    McDevitt et al. decompose only the electron-electron operator between the
    Fokker-Planck and Moller pieces.  The increment in nu_D obtained by
    increasing Zeff by one must therefore retain the full relativistic
    electron-ion Coulomb logarithm for both conservative and source-only
    large-angle configurations.
    """

    vte = 0.1
    te_ev = 0.5 * ME_C2_EV * vte**2
    coulog = 15.0
    gamma = jnp.asarray([2.0, 5.0, 10.0], dtype=jnp.float64)
    ne_norm = jnp.ones_like(gamma)
    te = jnp.full_like(gamma, te_ev)
    log = jnp.full_like(gamma, coulog)
    p = momentum_from_gamma(gamma)
    _log_ee, log_ei = relativistic_coulomb_logs(gamma, te, log)
    expected_ion_increment = gamma * (log_ei / coulog) / (p**3)

    for cfg in (
        SmallAngleConfig(
            1.0e14,
            coulog,
            large_angle_reduced_coulog=True,
            large_angle_gamma_min=1.02,
        ),
        SmallAngleConfig(
            1.0e14,
            coulog,
            large_angle_source_coulog=True,
        ),
    ):
        nu_z1 = pitch_scattering_frequency(
            gamma, jnp.ones_like(gamma), ne_norm, te, log, cfg
        )
        nu_z2 = pitch_scattering_frequency(
            gamma, 2.0 * jnp.ones_like(gamma), ne_norm, te, log, cfg
        )
        assert jnp.allclose(
            nu_z2 - nu_z1,
            expected_ion_increment,
            rtol=5.0e-13,
            atol=5.0e-13,
        )

    # Also make the intended electron-electron modifications explicit here.
    reduced = reduced_large_angle_coulomb_log(te, log, 1.02)
    source = source_only_large_angle_coulomb_log(gamma, te, log)
    assert bool(jnp.all(jnp.isfinite(reduced)))
    assert bool(jnp.all(jnp.isfinite(source)))


def test_partial_screening_friction_matches_supplied_ramc_formula():
    """Port of RAMc ``ComputeFrictionHesslow`` for singly ionized argon."""

    te_ev = 10.0
    vte = (2.0 * te_ev / ME_C2_EV) ** 0.5
    coulog = 10.0
    gamma = jnp.asarray([1.5, 3.0, 10.0], dtype=jnp.float64)
    cfg = SmallAngleConfig(
        1.0e14,
        coulog,
        partial_screening=True,
        impurity_fraction=1.0,
        impurity_nuclear_charge=18.0,
        impurity_charge_state=1.0,
        impurity_radius_abohr=0.329,
        impurity_mean_excitation_ev=219.4,
        screening_k=5.0,
    )
    got = partial_screening_friction_coefficient(
        gamma, jnp.ones_like(gamma), jnp.full_like(gamma, te_ev),
        jnp.full_like(gamma, coulog), cfg
    )

    p0 = jnp.sqrt(gamma * gamma - 1.0)
    v = p0 / gamma
    x = v / vte
    from jax.scipy.special import erf
    psi = 0.5 / (x * x) * (erf(x) - 2.0 / jnp.sqrt(jnp.pi) * x * jnp.exp(-x * x))
    logee = coulog + 0.2 * jnp.log(1.0 + (2.0 * (gamma - 1.0) / vte**2) ** 2.5)
    h_i = p0 * jnp.sqrt(gamma - 1.0) / (219.4 / ME_C2_EV)
    bound = 17.0 * (1.0 / 2.0) * (0.2 * jnp.log(1.0 + h_i**5) - v * v)
    expected = 2.0 / vte**2 * psi * (logee / coulog) * (1.0 + bound / logee)
    assert jnp.allclose(got, expected, rtol=2e-11, atol=2e-11)
