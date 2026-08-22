"""McDevitt-2019 large-angle / runaway-avalanche validation.

This benchmark is particle Monte Carlo only. JONTA is compared directly with
published numerical values and analytical relations from McDevitt, Guo & Tang,
PPCF 61, 054008 (2019). No auxiliary continuum kinetic solver is used.

The benchmark family contains:

* Moller source-limit avalanche growth, Appendix Fig. B3(a);
* avalanche threshold scaling, Eq. (B15) / Fig. B3(b);
* conservative mixed FP--Boltzmann cutoff invariance, Fig. 13;
* conservative gain--loss versus conventional source-only avalanche physics,
  Fig. 14.

The hot particle loop uses fixed-shape arrays, fixed-N random thinning and JAX
control flow, so the same kernel is GPU-ready. The paper-quality configurations
are intentionally marked slow.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boundaries.basic import gamma_absorbing_boundary
from collisions.moller import apply_moller_gain_loss, apply_moller_source_only
from collisions.small_angle import (
    reduced_large_angle_coulomb_log,
    small_angle_step,
    source_only_large_angle_coulomb_log,
)
from core.config import MollerConfig, SmallAngleConfig
from core.constants import ALPHA_FS, ME_C2_EV
from core.initialization import (
    particles_from_arrays,
    rosenbluth_energy_scale,
    rosenbluth_legendre_markers,
)
from core.state import BackgroundProfiles, ParticleState
from diagnostics.avalanche import fit_exponential_growth, replicate_mean_sem
from fields.uniform import UniformField
from integrators.explicit import rk4_step
from orbits.zero_d import zero_d_rhs

jax.config.update("jax_enable_x64", True)

REFERENCE_DATA = Path(__file__).parent / "reference_data" / "mcdevitt_2019_ppcf_large_angle.csv"
B4_REFERENCE_DATA = Path(__file__).parent / "reference_data" / "mcdevitt_2019_ppcf_b4.csv"
DEFAULT_VTE = 0.1


B3_INV_ALPHA_GRID = (2.0, 2.5, 4.0, 8.0, 10.0, 20.0, 40.0, 80.0)
MCDEVITT_FIGURE_TARGETS = ("2", "3", "4", "5", "6", "7", "8", "10", "11", "13", "14", "B2", "B3", "B4")


def argon_mixture_parameters(impurity_fraction, charge_state, atomic_number=18.0):
    """Return ``(Zeff, target_electron_factor)`` for D + Ar^q+.

    ``impurity_fraction`` is ``n_Ar/n_D``.  The Moller target factor includes
    both free and weakly bound electrons, as in McDevitt Appendix B.2.
    """

    f = float(impurity_fraction)
    zi = float(charge_state)
    z0 = float(atomic_number)
    ne_over_nd = 1.0 + zi * f
    zeff = (1.0 + zi * zi * f) / ne_over_nd
    target_factor = (1.0 + z0 * f) / ne_over_nd
    return zeff, target_factor



# Hesslow partial-screening atomic inputs.  The Ar+ values are the exact values
# used by the supplied RAMc decks.  Ar3+ and Ar5+ are reconstructed from the
# rounded ln(a_bar) and ln(I^-1) values in Hesslow et al. PPCF 60, 074010,
# Table A1, using a_bar = 2*a_I/alpha_FS and I normalized to m_e c^2.
_ARGON_HESSLOW_TABLE = {
    1: (0.329, 219.4),
    3: (0.5 * ALPHA_FS * np.exp(4.4), ME_C2_EV * np.exp(-7.5)),
    5: (0.5 * ALPHA_FS * np.exp(4.2), ME_C2_EV * np.exp(-7.2)),
}


def argon_hesslow_parameters(charge_state):
    try:
        return _ARGON_HESSLOW_TABLE[int(charge_state)]
    except KeyError as exc:
        raise ValueError("B4 currently defines Ar charge states 1, 3, and 5") from exc

def psi10_current_ma(gamma0_tau_c):
    """Return ``2*pi*psi_10/(mu0*R0)`` in MA from dimensionless ``gamma0*tau_c``."""

    # psi_10 = ln(10) R0 Ec / gamma0 and Ec*tau_c = m_e*c/e.
    m_e = 9.1093837015e-31
    c = 299792458.0
    e = 1.602176634e-19
    mu0 = 1.25663706212e-6
    return 2.0 * np.pi * np.log(10.0) * m_e * c / (e * mu0 * gamma0_tau_c) / 1.0e6


def rosenbluth_putvinski_gamma0_tau_c(radius_over_a, aspect_ratio, zeff, coulog):
    """McDevitt Eqs. (1)-(2) high-field avalanche coefficient."""

    eps = float(radius_over_a) * float(aspect_ratio)
    phi = 1.0 / (1.0 + 1.46 * np.sqrt(max(eps, 0.0)) + 1.72 * eps)
    return np.sqrt(np.pi * phi / (3.0 * (float(zeff) + 5.0))) / float(coulog)


def mcdevitt_threshold_fit(alpha, zeff):
    """McDevitt Appendix Eq. (B15), inherited from Ref. [15]."""

    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    zeff = jnp.asarray(zeff, dtype=jnp.float64)
    return 1.0 + 1.0906 * (alpha**0.6 * (zeff + 1.0)) ** 0.6801


def load_reference_data(path: Path = REFERENCE_DATA):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "figure": row["figure"],
                    "series": row["series"],
                    "e_over_ec": float(row["e_over_ec"]),
                    "gamma_min_minus_1": (
                        float(row["gamma_min_minus_1"])
                        if row["gamma_min_minus_1"].strip()
                        else np.nan
                    ),
                    "value": float(row["value"]),
                    "source_note": row["source_note"],
                }
            )
    return rows


def b3_growth_reference():
    return [r for r in load_reference_data() if r["figure"] == "B3a"]


def fig13_reference(e_over_ec=None):
    rows = [r for r in load_reference_data() if r["figure"] == "13"]
    if e_over_ec is not None:
        rows = [r for r in rows if np.isclose(r["e_over_ec"], e_over_ec)]
    return rows



def b4_reference(path: Path = B4_REFERENCE_DATA):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "charge_state": int(row["charge_state"]),
                    "alpha": float(row["alpha"]),
                    "n_ar_over_n_d": float(row["n_ar_over_n_d"]),
                    "eav_over_ec": float(row["eav_over_ec"]),
                    "source_note": row["source_note"],
                }
            )
    return rows


def _b4_reference_value(charge_state, alpha, impurity_fraction):
    matches = [
        r
        for r in b4_reference()
        if r["charge_state"] == int(charge_state)
        and np.isclose(r["alpha"], alpha)
        and np.isclose(r["n_ar_over_n_d"], impurity_fraction)
    ]
    if len(matches) != 1:
        raise ValueError(
            f"no unique Fig. B4 reference for Ar{charge_state}+, "
            f"alpha={alpha}, nAr/nD={impurity_fraction}"
        )
    return matches[0]["eav_over_ec"]

def _te_from_vte(vte_over_c):
    return 0.5 * ME_C2_EV * vte_over_c**2


def _ne_for_coulog(coulog0, te_ev):
    """Density making JONTA's thermal Coulomb log exactly ``coulog0``."""

    return 1.0e14 * np.exp(2.0 * (14.9 + np.log(te_ev * 1.0e-3) - coulog0))


def _background(vte_over_c, zeff, coulog0):
    te_ev = _te_from_vte(vte_over_c)
    ne_cm3 = _ne_for_coulog(coulog0, te_ev)
    r = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    bg = BackgroundProfiles(
        r,
        jnp.full_like(r, ne_cm3),
        jnp.full_like(r, te_ev),
        jnp.full_like(r, te_ev),
        jnp.full_like(r, zeff),
        jnp.ones_like(r),
    )
    return bg, ne_cm3, te_ev


def _initialize_seed(
    n_markers,
    seed,
    gamma_range=(10.0, 20.0),
    xi_range=(-1.0, -0.8),
):
    rng = np.random.default_rng(seed)
    gamma = rng.uniform(gamma_range[0], gamma_range[1], n_markers)
    xi = rng.uniform(xi_range[0], xi_range[1], n_markers)
    z = np.zeros(n_markers)
    return particles_from_arrays(
        gamma,
        xi,
        z,
        z,
        z,
        np.full(n_markers, 1.0 / n_markers),
        pid=np.arange(n_markers),
    )


@pytest.fixture(autouse=True)
def _release_cached_kernels_after_test():
    """Keep sequential CPU test runs from retaining large compiled executables."""

    yield
    _build_kernel.cache_clear()
    jax.clear_caches()


@lru_cache(maxsize=128)
def _build_kernel(
    *,
    zeff,
    coulog0,
    dt,
    n_steps,
    large_angle_every,
    sample_every,
    conservative,
    source_relativistic_log=False,
    reduced_log=False,
    vte_over_c=DEFAULT_VTE,
    partial_screening=False,
    impurity_fraction=0.0,
    impurity_nuclear_charge=18.0,
    impurity_charge_state=1.0,
    impurity_radius_abohr=0.329,
    impurity_mean_excitation_ev=219.4,
    target_electron_factor=1.0,
    relativistic_coulog=False,
    energy_scattering=False,
    small_angle_substeps=1,
    max_nu_dt=0.25,
):
    bg, ne_cm3, _te_ev = _background(vte_over_c, zeff, coulog0)
    sa_cfg = SmallAngleConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog0,
        pitch_scattering=True,
        friction=True,
        energy_scattering=energy_scattering,
        max_nu_dt=max_nu_dt,
        partial_screening=partial_screening,
        impurity_fraction=impurity_fraction,
        impurity_nuclear_charge=impurity_nuclear_charge,
        impurity_charge_state=impurity_charge_state,
        impurity_radius_abohr=impurity_radius_abohr,
        impurity_mean_excitation_ev=impurity_mean_excitation_ev,
        relativistic_coulog=relativistic_coulog,
        large_angle_reduced_coulog=reduced_log,
        large_angle_source_coulog=source_relativistic_log,
        large_angle_gamma_min=1.02,
    )
    la_cfg = MollerConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog0,
        gamma_min=1.02,
        max_collision_fraction=0.20,
        bisection_steps=24,
        target_electron_factor=target_electron_factor,
    )
    dt_la = dt * large_angle_every
    n_sa = int(small_angle_substeps)
    if n_sa < 1:
        raise ValueError("small_angle_substeps must be positive")
    dt_sa_half = 0.5 * dt / n_sa
    n_samples = n_steps // sample_every + 1

    @jax.jit
    def run(particles, base_key, e_over_ec, gamma_min, alpha_syn):
        field = UniformField(e_over_ec)
        sa_runtime = sa_cfg._replace(large_angle_gamma_min=gamma_min)
        la_runtime = la_cfg._replace(gamma_min=gamma_min)
        times = jnp.zeros((n_samples,), dtype=jnp.float64).at[0].set(0.0)
        weights = jnp.zeros((n_samples,), dtype=jnp.float64).at[0].set(
            jnp.sum(particles.weight)
        )
        sample_index = jnp.asarray(1, dtype=jnp.int32)
        max_q = jnp.asarray(0.0, dtype=jnp.float64)

        def body(i, carry):
            p, times_, weights_, idx_, max_q_ = carry
            # Strang split stochastic small-angle physics around the RK4
            # deterministic electric + synchrotron push.
            def collide_first(j, pp):
                step_id = (2 * i) * n_sa + j
                return small_angle_step(
                    pp, bg, dt_sa_half, base_key, step_id, sa_runtime
                )

            p = jax.lax.fori_loop(0, n_sa, collide_first, p)
            rhs = lambda kin, t: zero_d_rhs(kin, t, field, alpha_syn=alpha_syn)
            kin = rk4_step(rhs, p.kin, i * dt, dt)
            p = ParticleState(kin, p.weight, p.alive, p.pid)

            def collide_second(j, pp):
                step_id = (2 * i + 1) * n_sa + j
                return small_angle_step(
                    pp, bg, dt_sa_half, base_key, step_id, sa_runtime
                )

            p = jax.lax.fori_loop(0, n_sa, collide_second, p)
            p = gamma_absorbing_boundary(p, gamma_min)

            def do_large_angle(pp):
                if conservative:
                    return apply_moller_gain_loss(
                        pp, bg, dt_la, base_key, i, la_runtime
                    )
                return apply_moller_source_only(
                    pp, bg, dt_la, base_key, i, la_runtime
                )

            p, q_here = jax.lax.cond(
                ((i + 1) % large_angle_every) == 0,
                do_large_angle,
                lambda pp: (pp, jnp.asarray(0.0, dtype=jnp.float64)),
                p,
            )
            max_q_ = jnp.maximum(max_q_, q_here)

            def record(vals):
                pp, tt, ww, ii = vals
                tt = tt.at[ii].set((i + 1) * dt)
                ww = ww.at[ii].set(jnp.sum(pp.weight))
                return pp, tt, ww, ii + 1

            p, times_, weights_, idx_ = jax.lax.cond(
                ((i + 1) % sample_every) == 0,
                record,
                lambda vals: vals,
                (p, times_, weights_, idx_),
            )
            return p, times_, weights_, idx_, max_q_

        return jax.lax.fori_loop(
            0,
            n_steps,
            body,
            (particles, times, weights, sample_index, max_q),
        )

    return run


def run_growth_replicates(
    e_over_ec,
    *,
    alpha,
    zeff,
    coulog0,
    gamma_min=1.02,
    conservative=False,
    source_relativistic_log=False,
    reduced_log=False,
    n_markers=1024,
    total_time=10.0,
    dt=0.005,
    large_angle_dt=0.1,
    sample_dt=0.1,
    seeds=(0, 1, 2),
    fit_start_fraction=0.35,
    vte_over_c=DEFAULT_VTE,
    partial_screening=False,
    impurity_fraction=0.0,
    impurity_nuclear_charge=18.0,
    impurity_charge_state=1.0,
    impurity_radius_abohr=0.329,
    impurity_mean_excitation_ev=219.4,
    target_electron_factor=1.0,
    relativistic_coulog=False,
    energy_scattering=False,
    initial_gamma_range=(10.0, 20.0),
    initial_xi_range=(-1.0, -0.8),
    initialization="uniform",
    rosenbluth_pitch_width=None,
    rosenbluth_lmax=24,
    rosenbluth_energy_extent=1.7,
    rosenbluth_seed_floor_fraction=0.15,
    small_angle_substeps=1,
    max_nu_dt=0.25,
    return_final=False,
):
    large_angle_every = max(1, int(round(large_angle_dt / dt)))
    sample_every = max(1, int(round(sample_dt / dt)))
    n_steps = int(round(total_time / dt))
    kernel = _build_kernel(
        zeff=zeff,
        coulog0=coulog0,
        dt=dt,
        n_steps=n_steps,
        large_angle_every=large_angle_every,
        sample_every=sample_every,
        conservative=conservative,
        source_relativistic_log=source_relativistic_log,
        reduced_log=reduced_log,
        vte_over_c=vte_over_c,
        partial_screening=partial_screening,
        impurity_fraction=impurity_fraction,
        impurity_nuclear_charge=impurity_nuclear_charge,
        impurity_charge_state=impurity_charge_state,
        impurity_radius_abohr=impurity_radius_abohr,
        impurity_mean_excitation_ev=impurity_mean_excitation_ev,
        target_electron_factor=target_electron_factor,
        relativistic_coulog=relativistic_coulog,
        energy_scattering=energy_scattering,
        small_angle_substeps=small_angle_substeps,
        max_nu_dt=max_nu_dt,
    )

    results = []
    histories = []
    finals = []
    for seed in seeds:
        if initialization == "uniform":
            p0 = _initialize_seed(
                n_markers,
                seed,
                gamma_range=initial_gamma_range,
                xi_range=initial_xi_range,
            )
        elif initialization == "rosenbluth_legendre":
            warm_scale = rosenbluth_energy_scale(
                float(e_over_ec),
                float(zeff),
                float(coulog0),
                alpha_syn=float(alpha),
            )
            warm_gamma_min = max(
                float(gamma_min),
                1.0 + float(rosenbluth_seed_floor_fraction) * warm_scale,
            )
            p0 = rosenbluth_legendre_markers(
                jax.random.key(seed + 17011),
                n_markers,
                e_over_ec=float(e_over_ec),
                zeff=float(zeff),
                coulog=float(coulog0),
                gamma_min=warm_gamma_min,
                energy_extent=float(rosenbluth_energy_extent),
                pitch_width=(
                    None
                    if rosenbluth_pitch_width is None
                    else float(rosenbluth_pitch_width)
                ),
                lmax=int(rosenbluth_lmax),
                alpha_syn=float(alpha),
            )
        else:
            raise ValueError(
                "initialization must be 'uniform' or 'rosenbluth_legendre'"
            )
        _pf, time, weight, idx, max_q = kernel(
            p0,
            jax.random.key(seed + 88001),
            jnp.asarray(e_over_ec),
            jnp.asarray(gamma_min),
            jnp.asarray(alpha),
        )
        idx = int(idx)
        t = np.asarray(time[:idx])
        w = np.asarray(weight[:idx])
        growth, intercept, r2 = fit_exponential_growth(t, w, fit_start_fraction)
        results.append((growth, r2, float(max_q)))
        histories.append((seed, t, w, growth, intercept, r2))
        if return_final:
            finals.append(jax.device_get(_pf))

    arr = np.asarray(results)
    if len(seeds) > 1:
        growth_mean, growth_std, growth_sem = replicate_mean_sem(arr[:, 0])
    else:
        growth_mean, growth_std, growth_sem = float(arr[0, 0]), 0.0, 0.0
    return {
        "growth": growth_mean,
        "std": growth_std,
        "sem": growth_sem,
        "r2_mean": float(np.mean(arr[:, 1])),
        "max_q": float(np.max(arr[:, 2])),
        "replicates": arr[:, 0],
        "histories": histories,
        "finals": finals,
    }




def threshold_interpolation_sem(values):
    """Propagate growth-rate SEMs through a linear zero-crossing estimate.

    ``values`` is an iterable of ``(E/Ec, growth, growth_SEM)`` tuples.  The
    first neighboring pair that brackets zero is used.  Independent growth
    estimates are assumed, which is appropriate for the independent PRNG
    replicate ensembles used by these benchmarks.
    """

    vals = sorted(values)
    for (e0, g0, s0), (e1, g1, s1) in zip(vals[:-1], vals[1:]):
        if not ((g0 <= 0.0 <= g1) or (g1 <= 0.0 <= g0)):
            continue
        de = float(e1 - e0)
        dg = float(g1 - g0)
        if abs(dg) < 1.0e-30:
            return float("nan")
        d_edg0 = -de * float(g1) / (dg * dg)
        d_edg1 = de * float(g0) / (dg * dg)
        return float(np.sqrt((d_edg0 * s0) ** 2 + (d_edg1 * s1) ** 2))
    return float("nan")

def estimate_threshold(e_values, **growth_kwargs):
    """Estimate zero-growth crossing by local linear interpolation."""

    values = []
    for e in e_values:
        res = run_growth_replicates(e, **growth_kwargs)
        values.append((float(e), res["growth"], res["sem"]))
    values = sorted(values)
    for (e0, g0, _), (e1, g1, _) in zip(values[:-1], values[1:]):
        if g0 <= 0.0 <= g1 or g1 <= 0.0 <= g0:
            threshold = e0 - g0 * (e1 - e0) / (g1 - g0)
            return float(threshold), values
    raise ValueError("electric-field scan does not bracket the avalanche threshold")


def estimate_b3_threshold(alpha, zeff, *, bracket_fraction=0.08, **growth_kwargs):
    """Bracket and interpolate the Appendix-B3(b) zero-growth threshold.

    Eq. (B15) is used only to choose a computationally efficient bracket; the
    returned threshold comes exclusively from JONTA Monte Carlo growth rates.
    """

    reference = float(mcdevitt_threshold_fit(alpha, zeff))
    span = max(0.08, bracket_fraction * reference)
    all_values = []
    for _ in range(4):
        fields = (max(1.001, reference - span), reference + span)
        values = []
        for e in fields:
            res = run_growth_replicates(e, alpha=alpha, zeff=zeff, **growth_kwargs)
            values.append((float(e), res["growth"], res["sem"]))
        all_values.extend(values)
        (e0, g0, _), (e1, g1, _) = values
        if (g0 <= 0.0 <= g1) or (g1 <= 0.0 <= g0):
            threshold = e0 - g0 * (e1 - e0) / (g1 - g0)
            return float(threshold), sorted(all_values)
        span *= 1.7
    raise ValueError(
        f"could not bracket B3 threshold for alpha={alpha}, Zeff={zeff}; "
        f"samples={all_values}"
    )



def estimate_b4_threshold(
    impurity_fraction,
    charge_state,
    alpha,
    *,
    bracket_fraction=0.10,
    **growth_kwargs,
):
    """Particle-MC threshold for McDevitt Appendix Fig. B4.

    The digitized paper marker is used only to choose the initial field bracket;
    the returned zero crossing is obtained from JONTA growth rates.
    """

    reference = _b4_reference_value(charge_state, alpha, impurity_fraction)
    span = max(0.35, bracket_fraction * reference)
    vte = np.sqrt(2.0 * 10.0 / ME_C2_EV)
    zeff, target_factor = argon_mixture_parameters(impurity_fraction, charge_state)
    a_i, i_i = argon_hesslow_parameters(charge_state)
    common = dict(
        alpha=alpha,
        zeff=zeff,
        coulog0=10.0,
        gamma_min=1.02,
        vte_over_c=vte,
        partial_screening=True,
        impurity_fraction=impurity_fraction,
        impurity_nuclear_charge=18.0,
        impurity_charge_state=float(charge_state),
        impurity_radius_abohr=a_i,
        impurity_mean_excitation_ev=i_i,
        target_electron_factor=target_factor,
        max_nu_dt=0.5,
    )
    common.update(growth_kwargs)

    all_values = []
    for _ in range(4):
        fields = (max(1.001, reference - span), reference + span)
        values = []
        for e in fields:
            res = run_growth_replicates(e, **common)
            values.append((float(e), res["growth"], res["sem"]))
        all_values.extend(values)
        (e0, g0, _), (e1, g1, _) = values
        if (g0 <= 0.0 <= g1) or (g1 <= 0.0 <= g0):
            threshold = e0 - g0 * (e1 - e0) / (g1 - g0)
            return float(threshold), sorted(all_values)
        span *= 1.7
    raise ValueError(
        f"could not bracket B4 threshold for Ar{charge_state}+, "
        f"alpha={alpha}, f={impurity_fraction}; samples={all_values}"
    )

def test_threshold_fit_matches_published_equation_b15():
    alpha = np.asarray([0.5, 0.2, 0.1, 0.05, 0.025, 0.0125])
    for zeff in (1.0, 5.0):
        expected = 1.0 + 1.0906 * (alpha**0.6 * (zeff + 1.0)) ** 0.6801
        np.testing.assert_allclose(
            np.asarray(mcdevitt_threshold_fit(alpha, zeff)), expected, rtol=2e-15
        )


def test_mixed_operator_coulomb_logs_match_mcdevitt_eqs_32_and_33():
    vte = 0.1
    te = _te_from_vte(vte)
    ln0 = 15.0
    gamma_min = 1.02
    reduced = reduced_large_angle_coulomb_log(te, ln0, gamma_min)
    expected_reduced = ln0 + np.log(2.0 * np.sqrt(gamma_min - 1.0) / vte)
    np.testing.assert_allclose(float(reduced), expected_reduced, rtol=2e-15)

    gamma = jnp.asarray([2.0, 5.0, 10.0])
    source = source_only_large_angle_coulomb_log(gamma, te, ln0)
    expected_source = ln0 + np.log(np.sqrt(2.0 * (np.asarray(gamma) - 1.0)) / vte)
    np.testing.assert_allclose(np.asarray(source), expected_source, rtol=2e-15)


def test_reference_file_contains_b3_growth_and_fig13_cutoff_data():
    b3 = b3_growth_reference()
    assert len(b3) == 8
    assert np.isclose(b3[5]["e_over_ec"], 3.23210)
    assert np.isclose(b3[5]["value"], 0.027317)
    assert len(fig13_reference(2.05)) == 8
    assert len(fig13_reference(2.25)) == 8
    assert len(fig13_reference(2.5)) == 8
    assert len(fig13_reference(3.0)) == 8


def test_b4_reference_and_atomic_inputs_cover_all_published_series():
    rows = b4_reference()
    assert len(rows) == 20
    assert {(r["charge_state"], r["alpha"]) for r in rows} == {
        (1, 0.0),
        (1, 0.05),
        (3, 0.05),
        (5, 0.05),
    }
    a1, i1 = argon_hesslow_parameters(1)
    assert np.isclose(a1, 0.329)
    assert np.isclose(i1, 219.4)


@pytest.mark.slow
def test_b3_source_limit_growth_matches_published_monte_carlo():
    """Selected Fig. B3(a) points using the production fixed-N MC source."""

    refs = {round(r["e_over_ec"], 2): r["value"] for r in b3_growth_reference()}
    for e in (3.23, 3.73, 4.23):
        res = run_growth_replicates(
            e,
            alpha=0.5,
            zeff=2.0,
            coulog0=20.0,
            n_markers=1024,
            total_time=10.0,
            dt=0.005,
            large_angle_dt=0.1,
            seeds=(11, 29, 47, 71),
        )
        # Compare against the published Monte-Carlo marker using uncertainty
        # from independent JONTA replicas.  The small absolute floor accounts
        # for the deliberately short CPU integration; --paper reduces both.
        tolerance = max(3.0 * res["sem"], 0.006)
        assert abs(res["growth"] - refs[round(e, 2)]) < tolerance
        assert res["max_q"] < 0.20


@pytest.mark.slow
@pytest.mark.parametrize("zeff", [1.0, 5.0])
@pytest.mark.parametrize("inv_alpha", [4.0, 20.0])
def test_mcdevitt_b3_threshold_scan_tracks_eq_b15_for_both_zeff(zeff, inv_alpha):
    alpha = 1.0 / inv_alpha
    threshold, _ = estimate_b3_threshold(
        alpha,
        zeff,
        coulog0=20.0,
        n_markers=768,
        total_time=40.0,
        dt=0.005,
        large_angle_dt=0.1,
        seeds=(17, 43),
        fit_start_fraction=0.5,
    )
    expected = float(mcdevitt_threshold_fit(alpha, zeff))
    # Figure B3(b) notes the fit is best for 1/alpha > 10 and modestly
    # overestimates the Monte Carlo threshold below that range.
    assert abs(threshold - expected) / expected < 0.08


@pytest.mark.slow
def test_fig13_conservative_cutoff_plateau_is_recovered():
    """Growth is insensitive to the FP/Boltzmann partition below the X point."""

    values = []
    for delta_gamma in (0.04, 0.08, 0.16, 0.32):
        res = run_growth_replicates(
            2.5,
            alpha=0.1,
            zeff=1.0,
            coulog0=15.0,
            gamma_min=1.0 + delta_gamma,
            conservative=True,
            reduced_log=True,
            n_markers=768,
            total_time=8.0,
            dt=0.005,
            large_angle_dt=0.08,
            seeds=(5, 17),
        )
        values.append(res["growth"])
    values = np.asarray(values)
    # Fig. 13's physical point is the broad plateau, not a particular noisy
    # marker value. Require the spread to be small compared with the rate.
    assert np.ptp(values) < max(0.015, 0.55 * abs(np.mean(values)))


@pytest.mark.slow
def test_fig14_conservative_and_source_only_growth_are_close():
    """McDevitt Fig. 14: high-fidelity and source-only models nearly coincide."""

    common = dict(
        alpha=0.1,
        zeff=1.0,
        coulog0=15.0,
        gamma_min=1.02,
        n_markers=512,
        total_time=6.0,
        dt=0.005,
        large_angle_dt=0.08,
        seeds=(7, 19),
    )
    conservative = run_growth_replicates(
        3.0, conservative=True, reduced_log=True, **common
    )
    source = run_growth_replicates(
        3.0, conservative=False, source_relativistic_log=True, **common
    )
    assert abs(conservative["growth"] - source["growth"]) < 0.03


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run_driver(output_dir: Path, paper: bool = False, case: str = "all"):
    """Run the benchmark family and create publication-quality comparisons."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    if paper:
        n_markers, total_time, seeds = 8192, 40.0, tuple(range(8))
        b3_fields = [r["e_over_ec"] for r in b3_growth_reference()]
    else:
        n_markers, total_time, seeds = 1024, 10.0, (0, 1, 2, 3)
        b3_fields = [r["e_over_ec"] for r in b3_growth_reference()]

    growth_rows = []
    if case in ("all", "b3"):
        ref_lookup = {r["e_over_ec"]: r["value"] for r in b3_growth_reference()}
        for e in b3_fields:
            res = run_growth_replicates(
                e,
                alpha=0.5,
                zeff=2.0,
                coulog0=20.0,
                n_markers=n_markers,
                total_time=total_time,
                dt=0.0025 if paper else 0.005,
                large_angle_dt=0.05 if paper else 0.1,
                seeds=seeds,
            )
            growth_rows.append(
                {
                    "E_over_Ec": e,
                    "JONTA_growth": res["growth"],
                    "JONTA_SEM": res["sem"],
                    "paper_growth": ref_lookup[e],
                    "difference": res["growth"] - ref_lookup[e],
                    "fit_R2_mean": res["r2_mean"],
                    "max_event_probability": res["max_q"],
                }
            )
        _write_csv(output_dir / "large_angle_b3_growth.csv", growth_rows)

        paper_ref = b3_growth_reference()
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        ax.plot([r["e_over_ec"] for r in paper_ref], [r["value"] for r in paper_ref], "o-", label="McDevitt Fig. B3 MC")
        ax.errorbar(
            [r["E_over_Ec"] for r in growth_rows],
            [r["JONTA_growth"] for r in growth_rows],
            yerr=[r["JONTA_SEM"] for r in growth_rows],
            fmt="s",
            capsize=3,
            label="JONTA fixed-N MC",
        )
        ax.axhline(0.0, linewidth=0.8)
        ax.set(xlabel=r"$E/E_c$", ylabel=r"$\gamma_{av}\tau_c$", title="Avalanche growth: McDevitt Fig. B3(a)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "large_angle_b3_growth.png", dpi=360)
        plt.close(fig)

    # Appendix Fig. B3(b): full threshold scan for both published Zeff values.
    if case in ("all", "threshold"):
        threshold_time = 100.0 if paper else 40.0
        threshold_markers = 4096 if paper else 768
        threshold_seeds = tuple(range(8)) if paper else (17, 43)
        threshold_grid = B3_INV_ALPHA_GRID
        threshold_rows = []
        threshold_sample_rows = []
        for zeff in (1.0, 5.0):
            for inv_a in threshold_grid:
                alpha = 1.0 / inv_a
                threshold_mc, values = estimate_b3_threshold(
                    alpha,
                    zeff,
                    coulog0=20.0,
                    n_markers=threshold_markers,
                    total_time=threshold_time,
                    dt=0.0025 if paper else 0.005,
                    large_angle_dt=0.05 if paper else 0.1,
                    sample_dt=0.1,
                    seeds=threshold_seeds,
                    fit_start_fraction=0.6 if paper else 0.5,
                )
                expected = float(mcdevitt_threshold_fit(alpha, zeff))
                threshold_rows.append(
                    {
                        "Zeff": zeff,
                        "inv_alpha": inv_a,
                        "alpha": alpha,
                        "JONTA_Eav_over_Ec": threshold_mc,
                        "JONTA_Eav_SEM": threshold_interpolation_sem(values),
                        "Eq_B15_Eav_over_Ec": expected,
                        "relative_difference": (threshold_mc - expected) / expected,
                    }
                )
                for e, g, sem in values:
                    threshold_sample_rows.append(
                        {
                            "Zeff": zeff,
                            "inv_alpha": inv_a,
                            "E_over_Ec": e,
                            "growth": g,
                            "growth_SEM": sem,
                        }
                    )
        _write_csv(output_dir / "large_angle_b3_threshold.csv", threshold_rows)
        _write_csv(output_dir / "large_angle_b3_threshold_scan.csv", threshold_sample_rows)

        inv_alpha_curve = np.linspace(1.5, 80.0, 400)
        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        for zeff, marker in ((1.0, "o"), (5.0, "s")):
            curve = np.asarray(
                mcdevitt_threshold_fit(1.0 / inv_alpha_curve, zeff)
            )
            ax.plot(
                inv_alpha_curve,
                curve,
                label=fr"Eq. B15, $Z_{{eff}}={int(zeff)}$",
            )
            rr = [r for r in threshold_rows if r["Zeff"] == zeff]
            ax.errorbar(
                [r["inv_alpha"] for r in rr],
                [r["JONTA_Eav_over_Ec"] for r in rr],
                yerr=[r["JONTA_Eav_SEM"] for r in rr],
                fmt=marker,
                fillstyle="none",
                markersize=6,
                linestyle="none",
                capsize=3,
                label=fr"JONTA MC, $Z_{{eff}}={int(zeff)}$",
            )
        ax.set(
            xlim=(0.0, 82.0),
            xlabel=r"$1/\alpha$",
            ylabel=r"$E_{av}/E_c$",
            title="Avalanche threshold: McDevitt Fig. B3(b)",
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "large_angle_b3_threshold.png", dpi=360)
        plt.close(fig)

    # Conservative cutoff scan at E/Ec=2.5. Keep the expensive full four-E
    # figure for --paper; the default is sufficient to expose the plateau.
    cutoff_rows = []
    if case in ("all", "cutoff"):
        cutoff_e = (2.05, 2.25, 2.5, 3.0) if paper else (2.5,)
        cutoff_dg = (0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56) if paper else (0.04, 0.08, 0.16, 0.32, 0.64, 1.28, 2.56)
        for e in cutoff_e:
            for dg in cutoff_dg:
                res = run_growth_replicates(
                    e,
                    alpha=0.1,
                    zeff=1.0,
                    coulog0=15.0,
                    gamma_min=1.0 + dg,
                    conservative=True,
                    reduced_log=True,
                    n_markers=n_markers,
                    total_time=total_time,
                    dt=0.0025 if paper else 0.005,
                    large_angle_dt=0.05 if paper else 0.08,
                    seeds=seeds,
                )
                ref = fig13_reference(e)
                ref_val = min(ref, key=lambda r: abs(r["gamma_min_minus_1"] - dg))["value"]
                cutoff_rows.append(
                    {
                        "E_over_Ec": e,
                        "gamma_min_minus_1": dg,
                        "JONTA_growth": res["growth"],
                        "JONTA_SEM": res["sem"],
                        "paper_growth": ref_val,
                        "max_event_probability": res["max_q"],
                    }
                )
        _write_csv(output_dir / "large_angle_cutoff_scan.csv", cutoff_rows)

        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        for e in cutoff_e:
            rr = [r for r in cutoff_rows if np.isclose(r["E_over_Ec"], e)]
            paper_cutoff_ref = fig13_reference(e)
            ax.plot([r["gamma_min_minus_1"] for r in paper_cutoff_ref], [r["value"] for r in paper_cutoff_ref], "--", label=fr"paper $E/E_c={e:g}$")
            ax.errorbar([r["gamma_min_minus_1"] for r in rr], [r["JONTA_growth"] for r in rr], yerr=[r["JONTA_SEM"] for r in rr], fmt="o", capsize=3, label=fr"JONTA $E/E_c={e:g}$")
        ax.set_xscale("log")
        ax.set(xlabel=r"$\gamma_{min}^{LA}-1$", ylabel=r"$\gamma_{av}\tau_c$", title="Conservative cutoff invariance: McDevitt Fig. 13")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "large_angle_cutoff_scan.png", dpi=360)
        plt.close(fig)

    # Fig. 14 source-only / conservative comparison.
    compare_rows = []
    if case in ("all", "compare"):
        compare_fields = (2.25, 2.5, 3.0, 3.5)
        for e in compare_fields:
            common = dict(
                alpha=0.1,
                zeff=1.0,
                coulog0=15.0,
                gamma_min=1.02,
                n_markers=n_markers,
                total_time=total_time,
                dt=0.0025 if paper else 0.005,
                large_angle_dt=0.05 if paper else 0.08,
                seeds=seeds,
            )
            con = run_growth_replicates(e, conservative=True, reduced_log=True, **common)
            src = run_growth_replicates(e, conservative=False, source_relativistic_log=True, **common)
            compare_rows.append(
                {
                    "E_over_Ec": e,
                    "conservative_growth": con["growth"],
                    "conservative_SEM": con["sem"],
                    "source_only_growth": src["growth"],
                    "source_only_SEM": src["sem"],
                    "difference": con["growth"] - src["growth"],
                }
            )
        _write_csv(output_dir / "large_angle_conservative_vs_source.csv", compare_rows)

        fig, ax = plt.subplots(figsize=(6.2, 4.3))
        ax.errorbar([r["E_over_Ec"] for r in compare_rows], [r["conservative_growth"] for r in compare_rows], yerr=[r["conservative_SEM"] for r in compare_rows], fmt="o-", capsize=3, label="conservative gain-loss + Eq. 32")
        ax.errorbar([r["E_over_Ec"] for r in compare_rows], [r["source_only_growth"] for r in compare_rows], yerr=[r["source_only_SEM"] for r in compare_rows], fmt="s--", capsize=3, label="source only + Eq. 33")
        ax.set(xlabel=r"$E/E_c$", ylabel=r"$\gamma_{av}\tau_c$", title="Large-angle model comparison: McDevitt Fig. 14")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "large_angle_conservative_vs_source.png", dpi=360)
        plt.close(fig)

    # Appendix Fig. B4: partially screened avalanche thresholds.  This uses
    # the production Hesslow/RAMc drag + pitch-scattering coefficients and
    # includes bound electrons in the Moller target density.
    if case in ("all", "b4"):
        b4_markers = 2048 if paper else 128
        b4_time = 100.0 if paper else 3.0
        b4_seeds = tuple(range(8)) if paper else (23,)
        b4_dt = 5.0e-5 if paper else 1.0e-4
        b4_ladt = 5.0e-4 if paper else 1.0e-3
        b4_rows = []
        b4_scan_rows = []
        for charge_state, alpha_b4 in ((1, 0.0), (1, 0.05), (3, 0.05), (5, 0.05)):
            refs = [
                r
                for r in b4_reference()
                if r["charge_state"] == charge_state
                and np.isclose(r["alpha"], alpha_b4)
            ]
            for ref in refs:
                f_imp = ref["n_ar_over_n_d"]
                threshold_mc, values = estimate_b4_threshold(
                    f_imp,
                    charge_state,
                    alpha_b4,
                    n_markers=b4_markers,
                    total_time=b4_time,
                    dt=b4_dt,
                    large_angle_dt=b4_ladt,
                    sample_dt=0.05 if paper else 0.02,
                    seeds=b4_seeds,
                    fit_start_fraction=0.6 if paper else 0.45,
                )
                b4_rows.append(
                    {
                        "charge_state": charge_state,
                        "alpha": alpha_b4,
                        "n_Ar_over_n_D": f_imp,
                        "JONTA_Eav_over_Ec": threshold_mc,
                        "JONTA_Eav_SEM": threshold_interpolation_sem(values),
                        "paper_Eav_over_Ec": ref["eav_over_ec"],
                        "relative_difference": (threshold_mc - ref["eav_over_ec"])
                        / ref["eav_over_ec"],
                    }
                )
                for e, g, sem in values:
                    b4_scan_rows.append(
                        {
                            "charge_state": charge_state,
                            "alpha": alpha_b4,
                            "n_Ar_over_n_D": f_imp,
                            "E_over_Ec": e,
                            "growth": g,
                            "growth_SEM": sem,
                        }
                    )
        _write_csv(output_dir / "large_angle_b4_threshold.csv", b4_rows)
        _write_csv(output_dir / "large_angle_b4_threshold_scan.csv", b4_scan_rows)

        fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
        ax = axes[0]
        for alpha_b4, marker, label in ((0.0, "o", r"$\alpha=0$"), (0.05, "s", r"$\alpha=0.05$")):
            refs = [r for r in b4_reference() if r["charge_state"] == 1 and np.isclose(r["alpha"], alpha_b4)]
            rr = [r for r in b4_rows if r["charge_state"] == 1 and np.isclose(r["alpha"], alpha_b4)]
            ax.plot([r["n_ar_over_n_d"] for r in refs], [r["eav_over_ec"] for r in refs], "--", label=f"paper {label}")
            ax.errorbar([r["n_Ar_over_n_D"] for r in rr], [r["JONTA_Eav_over_Ec"] for r in rr], yerr=[r["JONTA_Eav_SEM"] for r in rr], fmt=marker, fillstyle="none", linestyle="none", capsize=3, label=f"JONTA {label}")
        ax.set_xscale("log")
        ax.set(xlabel=r"$n_{Ar}/n_D$", ylabel=r"$E_{av}/E_c$", title=r"(a) $Ar^+$")
        ax.legend(fontsize=7)

        ax = axes[1]
        for charge_state, marker in ((3, "o"), (5, "s")):
            refs = [r for r in b4_reference() if r["charge_state"] == charge_state]
            rr = [r for r in b4_rows if r["charge_state"] == charge_state]
            ax.plot([r["n_ar_over_n_d"] for r in refs], [r["eav_over_ec"] for r in refs], "--", label=fr"paper $Ar^{{{charge_state}+}}$")
            ax.errorbar([r["n_Ar_over_n_D"] for r in rr], [r["JONTA_Eav_over_Ec"] for r in rr], yerr=[r["JONTA_Eav_SEM"] for r in rr], fmt=marker, fillstyle="none", linestyle="none", capsize=3, label=fr"JONTA $Ar^{{{charge_state}+}}$")
        ax.set_xscale("log")
        ax.set(xlabel=r"$n_{Ar}/n_D$", ylabel=r"$E_{av}/E_c$", title=r"(b) $Ar^{3+}$ and $Ar^{5+}$")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(output_dir / "large_angle_b4_threshold.png", dpi=360)
        plt.close(fig)

    return growth_rows, cutoff_rows, compare_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("large_angle_results"))
    parser.add_argument("--paper", action="store_true", help="high-statistics GPU-oriented configuration")
    parser.add_argument(
        "--case",
        choices=("all", "b3", "threshold", "cutoff", "compare", "b4"),
        default="all",
        help="run one benchmark family at a time (useful for long GPU/CPU jobs)",
    )
    args = parser.parse_args()
    run_driver(args.output_dir, args.paper, args.case)


if __name__ == "__main__":
    main()
