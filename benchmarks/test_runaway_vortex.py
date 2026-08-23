"""Guo-2017 runaway-vortex / bump-on-tail validation targets.

This benchmark is intentionally *particle only*.  The external references are

1. numerical values reported by Guo, McDevitt & Tang, PPCF 59, 044003 (2017),
   stored in ``benchmarks/slab/bump_on_tail/reference/guo_2017_fig9.csv``; and
2. the analytical relations derived in that paper, especially Eqs. (16) and
   (22)--(25).

The production physics exercised here is the 0D Monte-Carlo system with
parallel electric acceleration, test-particle collisional drag, energy
scattering, pitch-angle scattering, and synchrotron radiation.  Large-angle
collisions and external sources are excluded.

The distribution-level reproduction is performed with the production marker
Monte-Carlo algorithm and compared directly with the published data and
analytical formulas.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from boundaries.basic import momentum_reservoir_boundary
from collisions.coulomb import (
    chandrasekhar,
    d_chandrasekhar_dx,
    thermal_coulomb_log,
)
from collisions.small_angle import pitch_scattering_frequency, small_angle_step
from core.config import SmallAngleConfig
from core.constants import ME_C2_EV
from core.initialization import particles_from_arrays
from core.rng import normal_by_particle
from core.state import BackgroundProfiles, KinematicState, ParticleState
from diagnostics.runaway_vortex import (
    guo_acceleration_channel_width,
    guo_bump_momentum,
    guo_bump_width,
    guo_o_point_momentum,
    guo_x_point_momentum,
)
from fields.uniform import UniformField
from integrators.explicit import rk4_step
from orbits.zero_d import zero_d_rhs

jax.config.update("jax_enable_x64", True)

DEFAULT_VTE_OVER_C = 0.1
DEFAULT_Z = 1.0
DEFAULT_ALPHA = 0.2
REFERENCE_DATA = Path(__file__).parent / "slab" / "bump_on_tail" / "reference" / "guo_2017_fig9.csv"


def load_published_reference_data(path: Path = REFERENCE_DATA):
    """Load values transcribed directly from Guo 2017 figures/text."""

    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "quantity": row["quantity"],
                    "e_over_ec": float(row["e_over_ec"]),
                    "value": float(row["value"]),
                    "source_note": row["source_note"],
                }
            )
    return rows


def published_bump_points():
    return {
        row["e_over_ec"]: row["value"]
        for row in load_published_reference_data()
        if row["quantity"] == "bump_momentum"
    }


def published_vortex_threshold():
    values = [
        row["value"]
        for row in load_published_reference_data()
        if row["quantity"] == "vortex_threshold"
    ]
    if len(values) != 1:
        raise ValueError("expected exactly one published vortex-threshold value")
    return values[0]


def _guo_collision_coefficients(p, vte_over_c=DEFAULT_VTE_OVER_C, z=DEFAULT_Z):
    """Guo Eqs. (3)--(5): C_F, C_A, C_B in tau_c normalization."""

    p = jnp.asarray(p, dtype=jnp.float64)
    gamma = jnp.sqrt(1.0 + p * p)
    v = p / gamma
    x = v / vte_over_c
    psi = chandrasekhar(x)
    phi = jax.scipy.special.erf(x)
    c_f = 2.0 * psi / (vte_over_c * vte_over_c)
    c_a = (gamma / p) * psi
    c_b = 0.5 * (gamma / p) * (
        z + phi - psi + 0.5 * vte_over_c**4 * x * x
    )
    return c_f, c_a, c_b


def _canonical_background(vte_over_c=DEFAULT_VTE_OVER_C, zeff=DEFAULT_Z):
    """Uniform background with an exactly prescribed thermal speed."""

    te_ev = 0.5 * ME_C2_EV * vte_over_c**2
    ne_cm3 = 1.0e14
    coulog = float(thermal_coulomb_log(ne_cm3, te_ev))
    radius = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    background = BackgroundProfiles(
        radius,
        jnp.full_like(radius, ne_cm3),
        jnp.full_like(radius, te_ev),
        jnp.full_like(radius, te_ev),
        jnp.full_like(radius, zeff),
        jnp.ones_like(radius),
    )
    return background, ne_cm3, te_ev, coulog


def test_published_guo_reference_values_are_preserved_verbatim():
    """The benchmark must compare against the paper, not another solver."""

    bumps = published_bump_points()
    assert bumps == {2.25: 6.0551, 2.5: 8.59096667}
    assert published_vortex_threshold() == 1.85


def test_guo_analytic_reference_relations_are_encoded_directly():
    """Check the diagnostic helpers against Guo Eqs. (16), (22)--(25)."""

    e = jnp.asarray([2.0, 2.5, 3.0, 4.0], dtype=jnp.float64)
    alpha = DEFAULT_ALPHA
    z = DEFAULT_Z

    expected_o = jnp.sqrt(2.0) * (e + alpha) * (e - 1.0) / ((1.0 + z) * alpha)
    expected_x = jnp.sqrt((1.0 + (1.0 + z) / (2.0 * jnp.sqrt(2.0))) / e)
    np.testing.assert_allclose(
        np.asarray(guo_o_point_momentum(e, alpha, z)), np.asarray(expected_o)
    )
    np.testing.assert_allclose(
        np.asarray(guo_x_point_momentum(e, z)), np.asarray(expected_x)
    )
    np.testing.assert_allclose(
        np.asarray(guo_bump_momentum(e, alpha, z)), np.asarray(expected_o / 1.55)
    )
    np.testing.assert_allclose(
        np.asarray(guo_bump_width(e, alpha, z)), np.asarray((expected_o - expected_x) / 1.8)
    )

    p = jnp.asarray([8.0, 10.0, 15.0], dtype=jnp.float64)
    gamma = jnp.sqrt(1.0 + p * p)
    expected_width = (2.5 - 1.0 - 1.0 / (p * p)) / (
        2.0 * alpha * p * gamma + 2.5
    )
    np.testing.assert_allclose(
        np.asarray(guo_acceleration_channel_width(p, 2.5, alpha)),
        np.asarray(expected_width),
    )


def test_production_0d_coefficients_match_guo_kinetic_equation():
    """Match every production 0D term to Guo's analytical coefficients."""

    p = jnp.asarray([0.5, 1.0, 5.0, 10.0], dtype=jnp.float64)
    gamma = jnp.sqrt(1.0 + p * p)
    xi = jnp.asarray([-0.95, -0.8, -0.5, -0.2], dtype=jnp.float64)
    zero = jnp.zeros_like(p)

    # Deterministic electric + synchrotron characteristics.
    kin = KinematicState(gamma, xi, zero, zero, zero)
    rhs = zero_d_rhs(kin, 0.0, UniformField(2.5), alpha_syn=DEFAULT_ALPHA)
    dp_dt = gamma / p * rhs.gamma
    expected_dp_dt = -xi * 2.5 - DEFAULT_ALPHA * p * gamma * (1.0 - xi * xi)
    expected_dxi_dt = (
        -(1.0 - xi * xi) * 2.5 / p
        + DEFAULT_ALPHA * xi * (1.0 - xi * xi) / gamma
    )
    np.testing.assert_allclose(
        np.asarray(dp_dt), np.asarray(expected_dp_dt), rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        np.asarray(rhs.xi), np.asarray(expected_dxi_dt), rtol=2e-13, atol=2e-13
    )

    background, ne_cm3, te_ev, coulog = _canonical_background()
    particles = particles_from_arrays(
        gamma,
        xi,
        zero,
        zero,
        zero,
        jnp.ones_like(p),
        pid=jnp.arange(p.size, dtype=jnp.int64),
    )
    dt = jnp.asarray(1.0e-8, dtype=jnp.float64)
    key = jax.random.key(901)

    # Collisional friction: dp/dt = -C_F.
    drag_cfg = SmallAngleConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog,
        pitch_scattering=False,
        friction=True,
        energy_scattering=False,
    )
    dragged = small_angle_step(particles, background, dt, key, 3, drag_cfg)
    c_f, c_a, _ = _guo_collision_coefficients(p)
    expected_drag_gamma = -(p / gamma) * c_f
    np.testing.assert_allclose(
        np.asarray((dragged.kin.gamma - gamma) / dt),
        np.asarray(expected_drag_gamma),
        rtol=2e-8,
        atol=2e-8,
    )

    # Energy diffusion: production gamma-space SDE is the Ito transform of
    # Guo's p-space C_A operator.
    energy_cfg = SmallAngleConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog,
        pitch_scattering=False,
        friction=False,
        energy_scattering=True,
    )
    scattered = small_angle_step(particles, background, dt, key, 7, energy_cfg)
    v = p / gamma
    x = v / DEFAULT_VTE_OVER_C
    psi = chandrasekhar(x)
    dpsi_dgamma = d_chandrasekhar_dx(x) / (
        DEFAULT_VTE_OVER_C * p * gamma * gamma
    )
    gamma_drift = 2.0 * psi / p + v * dpsi_dgamma
    gamma_sigma = jnp.sqrt(2.0 * v * psi)

    def c_a_scalar(pp):
        gg = jnp.sqrt(1.0 + pp * pp)
        vv = pp / gg
        xx = vv / DEFAULT_VTE_OVER_C
        return (gg / pp) * chandrasekhar(xx)

    dca_dp = jax.vmap(jax.grad(c_a_scalar))(p)
    transformed_drift = (p / gamma) * (2.0 * c_a / p + dca_dp) + c_a / gamma**3
    transformed_sigma = (p / gamma) * jnp.sqrt(2.0 * c_a)
    np.testing.assert_allclose(
        np.asarray(gamma_drift), np.asarray(transformed_drift), rtol=2e-11, atol=2e-11
    )
    np.testing.assert_allclose(
        np.asarray(gamma_sigma), np.asarray(transformed_sigma), rtol=2e-11, atol=2e-11
    )
    z_energy = normal_by_particle(key, particles.pid, 7, stream=12)
    expected_gamma = gamma + gamma_drift * dt + z_energy * gamma_sigma * jnp.sqrt(dt)
    np.testing.assert_allclose(
        np.asarray(scattered.kin.gamma),
        np.asarray(expected_gamma),
        rtol=2e-13,
        atol=2e-13,
    )

    # Pitch diffusion: nu_D = 2 C_B / p^2, the SDE equivalent of Guo Eq. (5).
    config = SmallAngleConfig(ne0_cm3=ne_cm3, coulog0=coulog)
    nu_d = pitch_scattering_frequency(
        gamma,
        jnp.ones_like(gamma),
        jnp.ones_like(gamma),
        jnp.full_like(gamma, te_ev),
        jnp.full_like(gamma, coulog),
        config,
    )
    _, _, c_b = _guo_collision_coefficients(p)
    np.testing.assert_allclose(
        np.asarray(nu_d),
        np.asarray(2.0 * c_b / (p * p)),
        rtol=2e-12,
        atol=2e-12,
    )


def test_jitted_monte_carlo_increment_statistics_match_guo_coefficients():
    """Statistically verify the production stochastic step, not a grid solve.

    A large identical-marker ensemble is advanced by one JIT-compiled
    small-angle step.  Sample means and variances are compared with the SDE
    moments implied by Guo Eqs. (2)--(5).  The chosen point is far from pitch
    and low-energy boundaries so reflection does not contaminate the moments.
    """

    n = 65_536
    p0 = 2.0
    xi0 = -0.70
    dt = 2.0e-4
    gamma0 = np.sqrt(1.0 + p0 * p0)

    gamma = jnp.full((n,), gamma0, dtype=jnp.float64)
    xi = jnp.full((n,), xi0, dtype=jnp.float64)
    zero = jnp.zeros((n,), dtype=jnp.float64)
    particles = particles_from_arrays(
        gamma,
        xi,
        zero,
        zero,
        zero,
        jnp.ones((n,), dtype=jnp.float64),
        pid=jnp.arange(n, dtype=jnp.int64),
    )
    background, ne_cm3, _te_ev, coulog = _canonical_background()
    config = SmallAngleConfig(ne0_cm3=ne_cm3, coulog0=coulog, max_nu_dt=0.5)
    key = jax.random.key(1729)

    step = jax.jit(
        lambda state: small_angle_step(state, background, dt, key, 0, config)
    )
    out = step(particles)
    dgamma = np.asarray(out.kin.gamma - gamma)
    dxi = np.asarray(out.kin.xi - xi)

    p = jnp.asarray(p0, dtype=jnp.float64)
    g = jnp.asarray(gamma0, dtype=jnp.float64)
    v = p / g
    x = v / DEFAULT_VTE_OVER_C
    psi = chandrasekhar(x)
    dpsi_dgamma = d_chandrasekhar_dx(x) / (
        DEFAULT_VTE_OVER_C * p * g * g
    )
    c_f, _c_a, c_b = _guo_collision_coefficients(p)
    gamma_mean = float((-(p / g) * c_f + 2.0 * psi / p + v * dpsi_dgamma) * dt)
    gamma_var = float(2.0 * v * psi * dt)

    nu_d = float(2.0 * c_b / (p * p))
    xi_mean = -xi0 * nu_d * dt
    xi_var = (1.0 - xi0 * xi0) * nu_d * dt

    # Statistical tolerances are expressed in standard errors of the sampled
    # moments.  The extra small floor prevents brittle failures from tiny
    # platform-dependent PRNG/roundoff differences.
    gamma_mean_se = np.sqrt(gamma_var / n)
    xi_mean_se = np.sqrt(xi_var / n)
    assert abs(np.mean(dgamma) - gamma_mean) < 6.0 * gamma_mean_se + 2.0e-7
    assert abs(np.mean(dxi) - xi_mean) < 6.0 * xi_mean_se + 2.0e-7
    assert abs(np.var(dgamma, ddof=1) / gamma_var - 1.0) < 0.03
    assert abs(np.var(dxi, ddof=1) / xi_var - 1.0) < 0.03


def test_guo_acceleration_channel_formula_matches_convective_energy_balance():
    """Guo Eq. (16) locates the large-p zero of convective energy flow."""

    e = 2.5
    alpha = DEFAULT_ALPHA
    p = jnp.asarray([8.0, 10.0, 15.0], dtype=jnp.float64)
    gamma = jnp.sqrt(1.0 + p * p)
    delta_xi = guo_acceleration_channel_width(p, e, alpha)
    xi = -1.0 + delta_xi
    c_f, _, _ = _guo_collision_coefficients(p)

    # Exact convective p-flow coefficient from Guo Eq. (14).
    flow = -xi * e - c_f - alpha * p * gamma * (1.0 - xi * xi)

    # Eq. (16) is a large-p approximation, so the residual must decrease as p
    # grows rather than being identically zero at finite p.
    residual = np.abs(np.asarray(flow))
    assert residual[-1] < residual[0]
    assert residual[-1] < 3.0e-2


# ---------------------------------------------------------------------------
# Distribution-level particle Monte-Carlo benchmark
# ---------------------------------------------------------------------------

DEFAULT_P_MIN = 3.0 * DEFAULT_VTE_OVER_C
DEFAULT_P_MAX = 20.0
DEFAULT_P_INIT_MAX = 14.0
DEFAULT_VORTEX_DT = 4.0e-3
DEFAULT_VORTEX_TOTAL_TIME = 60.0
DEFAULT_VORTEX_BURN_TIME = 30.0
DEFAULT_VORTEX_BINS = 320
DEFAULT_SAMPLE_EVERY = 5


def _make_vortex_initial_ensemble(
    n_markers: int,
    seed: int,
    *,
    p_min: float = DEFAULT_P_MIN,
    p_init_max: float = DEFAULT_P_INIT_MAX,
):
    """Broad fixed-capacity marker ensemble for steady-state burn-in.

    Guo et al. solve for a steady state, so the initial distribution is not a
    validation target.  A broad proposal in ``p`` and ``xi`` deliberately
    populates both the thermal-return and runaway-vortex regions and therefore
    shortens burn-in.  All markers carry equal weight.
    """

    rng = np.random.default_rng(seed)
    p = rng.uniform(p_min, p_init_max, n_markers)
    xi = rng.uniform(-1.0, 1.0, n_markers)
    gamma = np.sqrt(1.0 + p * p)
    zero = np.zeros(n_markers)
    return particles_from_arrays(
        gamma,
        xi,
        zero,
        zero,
        zero,
        np.full(n_markers, 1.0 / n_markers),
        pid=np.arange(n_markers, dtype=np.int64),
    )


def _make_vortex_kernel(
    *,
    n_bins: int = DEFAULT_VORTEX_BINS,
    p_min: float = DEFAULT_P_MIN,
    p_max: float = DEFAULT_P_MAX,
    phase_bins: tuple[int, int] | None = None,
    vte_over_c: float = DEFAULT_VTE_OVER_C,
    zeff: float = DEFAULT_Z,
    alpha: float = DEFAULT_ALPHA,
    energy_scattering: bool = True,
):
    """Build one JIT-compiled fixed-capacity runaway-vortex Monte-Carlo loop.

    The hot loop contains no Python timestepping or particle loops.  A Strang
    split is used between the production small-angle operator and RK4
    deterministic electric/synchrotron dynamics.  The pitch-averaged
    distribution is accumulated on device.  Optional 2-D phase-space
    occupancy is also accumulated for diagnostic reconstruction of the
    probability current; it is not used as an evolution solver.
    """

    background, ne_cm3, _te_ev, coulog = _canonical_background(vte_over_c, zeff)
    config = SmallAngleConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog,
        pitch_scattering=True,
        friction=True,
        energy_scattering=energy_scattering,
        max_nu_dt=0.5,
    )
    collect_phase = phase_bins is not None
    if phase_bins is None:
        n_p_phase, n_xi_phase = 1, 1
    else:
        n_p_phase, n_xi_phase = phase_bins

    @jax.jit
    def kernel(
        particles: ParticleState,
        e_over_ec,
        dt,
        n_steps,
        burn_steps,
        sample_every,
        phase_sample_every,
        base_key,
    ):
        field = UniformField(e_over_ec)
        hist0 = jnp.zeros((n_bins,), dtype=jnp.float64)
        phase0 = jnp.zeros((n_p_phase, n_xi_phase), dtype=jnp.float64)
        count0 = jnp.asarray(0, dtype=jnp.int32)
        phase_count0 = jnp.asarray(0, dtype=jnp.int32)

        def body(i, carry):
            state, histogram, phase_hist, count, phase_count = carry

            state = small_angle_step(
                state, background, 0.5 * dt, base_key, 2 * i, config
            )
            def rhs(kin, time):
                return zero_d_rhs(kin, time, field, alpha_syn=alpha)
            kin = rk4_step(rhs, state.kin, i * dt, dt)
            state = ParticleState(kin, state.weight, state.alive, state.pid)
            state = small_angle_step(
                state, background, 0.5 * dt, base_key, 2 * i + 1, config
            )
            state = momentum_reservoir_boundary(state, p_min, p_max)

            after_burn = i >= burn_steps
            take_sample = after_burn & (((i - burn_steps) % sample_every) == 0)

            def add_1d(args):
                state, histogram, count = args
                p = jnp.sqrt(jnp.maximum(state.kin.gamma * state.kin.gamma - 1.0, 0.0))
                p_index = jnp.clip(
                    ((p - p_min) * n_bins / (p_max - p_min)).astype(jnp.int32),
                    0,
                    n_bins - 1,
                )
                increment = jnp.bincount(
                    p_index,
                    weights=state.weight,
                    length=n_bins,
                )
                return state, histogram + increment, count + 1

            state, histogram, count = jax.lax.cond(
                take_sample,
                add_1d,
                lambda args: args,
                (state, histogram, count),
            )

            if collect_phase:
                take_phase = after_burn & (
                    ((i - burn_steps) % phase_sample_every) == 0
                )

                def add_2d(args):
                    state, phase_hist, phase_count = args
                    p = jnp.sqrt(
                        jnp.maximum(state.kin.gamma * state.kin.gamma - 1.0, 0.0)
                    )
                    ip = jnp.clip(
                        ((p - p_min) * n_p_phase / (p_max - p_min)).astype(jnp.int32),
                        0,
                        n_p_phase - 1,
                    )
                    ixi = jnp.clip(
                        ((state.kin.xi + 1.0) * 0.5 * n_xi_phase).astype(jnp.int32),
                        0,
                        n_xi_phase - 1,
                    )
                    flat = ip * n_xi_phase + ixi
                    increment = jnp.bincount(
                        flat,
                        weights=state.weight,
                        length=n_p_phase * n_xi_phase,
                    ).reshape((n_p_phase, n_xi_phase))
                    return state, phase_hist + increment, phase_count + 1

                state, phase_hist, phase_count = jax.lax.cond(
                    take_phase,
                    add_2d,
                    lambda args: args,
                    (state, phase_hist, phase_count),
                )

            return state, histogram, phase_hist, count, phase_count

        return jax.lax.fori_loop(
            0,
            n_steps,
            body,
            (particles, hist0, phase0, count0, phase_count0),
        )

    return kernel


def _distribution_from_histogram(hist, n_samples: int, *, p_min=DEFAULT_P_MIN, p_max=DEFAULT_P_MAX):
    """Convert marker counts per ``dp`` into pitch-integrated ``<f>_xi``.

    Marker probability in spherical momentum coordinates carries the
    ``p^2 dp dxi`` Jacobian.  Dividing the sampled momentum density by ``p^2``
    therefore recovers the shape plotted by Guo et al. up to an irrelevant
    normalization constant.
    """

    hist = np.asarray(hist, dtype=float)
    n_bins = hist.shape[-1]
    dp = (p_max - p_min) / n_bins
    p = p_min + (np.arange(n_bins) + 0.5) * dp
    density_dp = hist / max(int(n_samples), 1) / dp
    f_pitch = density_dp / np.maximum(p * p, 1.0e-30)
    return p, f_pitch


def _smooth_distribution(values, window: int = 21):
    values = np.asarray(values, dtype=float)
    window = int(max(1, min(window, values.size)))
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


def _estimate_runaway_bump(
    p,
    f_pitch,
    *,
    search_min: float = 4.0,
    search_max: float = 14.0,
    smooth_window: int = 21,
):
    """Return ``(has_bump, p_b)`` from a noisy particle distribution.

    The search deliberately starts above the thermal/transition shoulder.  A
    monotone tail therefore places its maximum on the first search bin and is
    classified as no bump.  For a genuine interior maximum, a local quadratic
    fit removes bin-centering bias.
    """

    p = np.asarray(p, dtype=float)
    smooth = _smooth_distribution(f_pitch, smooth_window)
    indices = np.where((p >= search_min) & (p <= search_max))[0]
    if indices.size < 7:
        raise ValueError("bump search interval contains too few bins")
    local = indices[np.argmax(smooth[indices])]
    edge_guard = max(2, smooth_window // 8)
    if local <= indices[0] + edge_guard or local >= indices[-1] - edge_guard:
        return False, np.nan

    # Require a visible rise relative to the low-p edge of the runaway-tail
    # search region.  This rejects small noisy extrema in a monotone tail.
    if smooth[local] <= 1.03 * smooth[indices[0]]:
        return False, np.nan

    half = 4
    fit_indices = np.arange(max(indices[0], local - half), min(indices[-1] + 1, local + half + 1))
    coeff = np.polyfit(p[fit_indices], smooth[fit_indices], 2)
    if coeff[0] >= 0.0:
        return True, float(p[local])
    p_peak = -coeff[1] / (2.0 * coeff[0])
    if not (p[fit_indices[0]] <= p_peak <= p[fit_indices[-1]]):
        p_peak = p[local]
    return True, float(p_peak)


def run_particle_vortex_case(
    kernel,
    e_over_ec: float,
    *,
    n_markers: int = 4096,
    seed: int = 41,
    dt: float = DEFAULT_VORTEX_DT,
    total_time: float = DEFAULT_VORTEX_TOTAL_TIME,
    burn_time: float = DEFAULT_VORTEX_BURN_TIME,
    sample_every: int = DEFAULT_SAMPLE_EVERY,
    phase_sample_every: int = 25,
    p_min: float = DEFAULT_P_MIN,
    p_max: float = DEFAULT_P_MAX,
    p_init_max: float = DEFAULT_P_INIT_MAX,
    vte_over_c: float = DEFAULT_VTE_OVER_C,
    zeff: float = DEFAULT_Z,
    alpha: float = DEFAULT_ALPHA,
):
    """Run one fixed-capacity particle-only steady runaway-vortex calculation.

    The distribution is time averaged after burn-in.  Marker groups provide a
    simple independent-subensemble estimate of statistical uncertainty in the
    inferred bump location.
    """

    particles = _make_vortex_initial_ensemble(
        n_markers, seed, p_min=p_min, p_init_max=p_init_max
    )
    n_steps = int(round(total_time / dt))
    burn_steps = int(round(burn_time / dt))
    if burn_steps >= n_steps:
        raise ValueError("burn_time must be smaller than total_time")

    final, histogram, phase_hist, n_samples, n_phase_samples = kernel(
        particles,
        jnp.asarray(e_over_ec, dtype=jnp.float64),
        jnp.asarray(dt, dtype=jnp.float64),
        jnp.asarray(n_steps, dtype=jnp.int32),
        jnp.asarray(burn_steps, dtype=jnp.int32),
        jnp.asarray(sample_every, dtype=jnp.int32),
        jnp.asarray(phase_sample_every, dtype=jnp.int32),
        jax.random.key(seed),
    )
    # Synchronize once per case; this is also the only device-to-host transfer
    # in the hot benchmark path.
    histogram = np.asarray(jax.device_get(histogram))
    phase_hist = np.asarray(jax.device_get(phase_hist))
    n_samples = int(jax.device_get(n_samples))
    n_phase_samples = int(jax.device_get(n_phase_samples))
    final = jax.device_get(final)

    p, f_pitch = _distribution_from_histogram(
        histogram, n_samples, p_min=p_min, p_max=p_max
    )
    has_bump, p_bump = _estimate_runaway_bump(p, f_pitch)
    bump_sem = np.nan

    return {
        "e_over_ec": float(e_over_ec),
        "n_markers": int(n_markers),
        "dt_tau_c": float(dt),
        "total_time_tau_c": float(total_time),
        "burn_time_tau_c": float(burn_time),
        "n_samples": n_samples,
        "has_bump": bool(has_bump),
        "p_bump": float(p_bump) if np.isfinite(p_bump) else np.nan,
        "p_bump_sem": bump_sem,
        "p": p,
        "f_pitch": f_pitch,
        "histogram": histogram,
        "phase_hist": phase_hist,
        "n_phase_samples": n_phase_samples,
        "final": final,
        "p_min": float(p_min),
        "p_max": float(p_max),
        "vte_over_c": float(vte_over_c),
        "zeff": float(zeff),
        "alpha": float(alpha),
    }


def test_particle_runaway_vortex_recovers_guo_bump_on_tail():
    """Distribution-level particle-only comparison with Guo Fig. 9.

    This is the primary acceptance test for the runaway vortex.  It uses the
    same production particle operators as JONTA.  The CPU-size configuration
    is intentionally modest; the
    command-line ``--paper`` mode is intended for the target GPUs.
    """

    kernel = _make_vortex_kernel(n_bins=240)
    reference = published_bump_points()
    cases = []
    for offset, e_over_ec in enumerate((2.0, 2.25, 2.5)):
        cases.append(
            run_particle_vortex_case(
                kernel,
                e_over_ec,
                n_markers=1024,
                seed=3100 + offset,
                dt=4.0e-3,
                total_time=35.0,
                burn_time=17.5,
                sample_every=5,
            )
        )

    # Guo Fig. 9: the pitch-integrated tail is monotone at E/Ec=2 even though
    # the two-dimensional runaway vortex already exists.
    assert not cases[0]["has_bump"]

    for result in cases[1:]:
        assert result["has_bump"]
        published = reference[result["e_over_ec"]]
        assert abs(result["p_bump"] - published) / published < 0.12


def _phase_space_distribution(result, *, p_min=None, p_max=None):
    if p_min is None:
        p_min = result.get("p_min", DEFAULT_P_MIN)
    if p_max is None:
        p_max = result.get("p_max", DEFAULT_P_MAX)
    hist = np.asarray(result["phase_hist"], dtype=float)
    np_bins, nxi_bins = hist.shape
    p = p_min + (np.arange(np_bins) + 0.5) * (p_max - p_min) / np_bins
    xi = -1.0 + (np.arange(nxi_bins) + 0.5) * 2.0 / nxi_bins
    f = hist / max(result["n_phase_samples"], 1)
    f = f / np.maximum(p[:, None] ** 2, 1.0e-30)
    return p, xi, f


def _smooth_2d(values, passes: int = 2):
    out = np.asarray(values, dtype=float)
    kernel = np.array([0.25, 0.5, 0.25])
    for _ in range(passes):
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 0, out)
        out = np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, out)
    return out


def reconstruct_phase_space_flux(result):
    """Reconstruct Guo's probability current from the MC distribution.

    This is diagnostic post-processing only: finite differences are applied to
    the *particle-sampled distribution* to visualize the current.  No PDE is
    solved and the reconstructed field never feeds back into particle
    evolution.
    """

    p, xi, f = _phase_space_distribution(result)
    f = _smooth_2d(f, passes=3)
    df_dp = np.gradient(f, p, axis=0, edge_order=2)
    df_dxi = np.gradient(f, xi, axis=1, edge_order=2)

    vte = result.get("vte_over_c", DEFAULT_VTE_OVER_C)
    zeff = result.get("zeff", DEFAULT_Z)
    alpha = result.get("alpha", DEFAULT_ALPHA)
    c_f, c_a, c_b = _guo_collision_coefficients(
        jnp.asarray(p), vte_over_c=vte, z=zeff
    )
    c_f = np.asarray(c_f)[:, None]
    c_a = np.asarray(c_a)[:, None]
    c_b = np.asarray(c_b)[:, None]
    pp = p[:, None]
    xx = xi[None, :]
    gamma = np.sqrt(1.0 + pp * pp)
    e = result["e_over_ec"]
    one_minus_xi2 = np.maximum(1.0 - xx * xx, 0.0)

    gamma_p = (
        -xx * e
        - c_f
        - alpha * pp * gamma * one_minus_xi2
    ) * f - c_a * df_dp
    gamma_xi = -one_minus_xi2 * (
        e * f / pp
        - alpha * xx * f / gamma
        + c_b * df_dxi / (pp * pp)
    )
    return p, xi, f, gamma_p, gamma_xi


def _write_particle_vortex_results(results, output_dir: Path):
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    published = published_bump_points()

    # Numerical summary.
    summary_path = output_dir / "runaway_vortex_particle_results.csv"
    with summary_path.open("w", newline="") as handle:
        fieldnames = [
            "e_over_ec",
            "n_markers",
            "dt_tau_c",
            "total_time_tau_c",
            "burn_time_tau_c",
            "n_samples",
            "has_bump",
            "p_bump",
            "p_bump_sem",
            "published_p_bump",
            "relative_error",
            "guo_eq24_p_bump",
            "runtime_seconds",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            e = result["e_over_ec"]
            ref = published.get(e, np.nan)
            rel = (result["p_bump"] - ref) / ref if np.isfinite(ref) and result["has_bump"] else np.nan
            writer.writerow(
                {
                    **{k: result[k] for k in fieldnames[:9]},
                    "published_p_bump": ref,
                    "relative_error": rel,
                    "guo_eq24_p_bump": float(guo_bump_momentum(e, DEFAULT_ALPHA, DEFAULT_Z)),
                    "runtime_seconds": result.get("runtime_seconds", np.nan),
                }
            )

    # Pitch-integrated runaway-tail distribution.  Each curve is normalized
    # at p=4 so the non-monotonic bump is visible without the unresolved
    # thermal-reservoir shoulder dominating the vertical scale.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for result in results:
        p = result["p"]
        f = _smooth_distribution(result["f_pitch"], 21)
        mask = (p >= 3.5) & (p <= 16.0)
        baseline = np.interp(4.0, p, f)
        tail = f / max(baseline, 1.0e-300)
        line = ax.plot(
            p[mask],
            np.maximum(tail[mask], 1.0e-5),
            label=f"JONTA E/Ec={result['e_over_ec']:g}",
        )[0]
        if result["has_bump"]:
            y_peak = np.interp(result["p_bump"], p, tail)
            ax.scatter(
                [result["p_bump"]],
                [y_peak],
                marker="o",
                s=35,
                color=line.get_color(),
            )
        p_ref = published.get(result["e_over_ec"])
        if p_ref is not None:
            y_ref = np.interp(p_ref, p, tail)
            ax.scatter(
                [p_ref],
                [y_ref],
                marker="x",
                s=55,
                color=line.get_color(),
            )
    ax.set_xlabel(r"$p/(m_e c)$")
    ax.set_ylabel(r"$\langle f\rangle_\xi / \langle f(4)\rangle_\xi$")
    ax.set_xlim(3.5, 16.0)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "runaway_vortex_energy_distribution.png", dpi=300)
    plt.close(fig)

    # Bump locations: MC vs published data and Guo Eq. (24).
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    e_curve = np.linspace(2.0, 4.0, 300)
    analytic = np.asarray(guo_bump_momentum(e_curve, DEFAULT_ALPHA, DEFAULT_Z))
    ax.plot(e_curve, analytic, linestyle="--", label="Guo Eq. (24)")
    for result in results:
        if not result["has_bump"]:
            continue
        yerr = result["p_bump_sem"] if np.isfinite(result["p_bump_sem"]) else None
        ax.errorbar(
            result["e_over_ec"],
            result["p_bump"],
            yerr=yerr,
            marker="o",
            linestyle="none",
            capsize=3,
            label=f"JONTA MC E/Ec={result['e_over_ec']:g}",
        )
    ax.scatter(list(published), list(published.values()), marker="x", s=60, label="Guo Fig. 9")
    ax.set_xlabel(r"$E/E_c$")
    ax.set_ylabel(r"bump momentum $p_b/(m_e c)$")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "runaway_vortex_bump_comparison.png", dpi=300)
    plt.close(fig)

    # Monte-Carlo reconstructed phase-space current.
    phase_results = [r for r in results if r["phase_hist"].shape != (1, 1)]
    if phase_results:
        fig, axes = plt.subplots(1, len(phase_results), figsize=(6.1 * len(phase_results), 4.8), squeeze=False)
        for ax, result in zip(axes[0], phase_results):
            p, xi, f, gp, gxi = reconstruct_phase_space_flux(result)
            mask_p = p <= 14.0
            pp = p[mask_p]
            ff = f[mask_p]
            gp = gp[mask_p]
            gxi = gxi[mask_p]
            positive = ff[ff > 0.0]
            floor = max(np.max(ff) * 1.0e-6, np.min(positive) if positive.size else 1.0e-30)
            log_density = np.log10(np.maximum(ff.T, floor))
            levels = np.linspace(np.nanmin(log_density), np.nanmax(log_density), 40)
            image = ax.contourf(pp, xi, log_density, levels=levels, cmap="plasma")
            speed = np.sqrt(gp * gp + gxi * gxi)
            u = gp / np.maximum(speed, 1.0e-300)
            v = gxi / np.maximum(speed, 1.0e-300)
            reliable = ff > np.max(ff) * 2.0e-4
            u = np.where(reliable, u, 0.0)
            v = np.where(reliable, v, 0.0)
            # Long white streamlines show complete probability-current
            # trajectories, matching the phase-space-flow presentation used
            # in Guo et al. The vectors are normalized because only the
            # direction field is integrated for this overlay.
            ax.streamplot(
                pp,
                xi,
                u.T,
                v.T,
                color="white",
                density=1.6,
                linewidth=0.45,
                arrowsize=0.75,
                minlength=0.2,
                maxlength=100.0,
                integration_direction="both",
            )
            ax.set_title(f"E/Ec={result['e_over_ec']:g}")
            ax.set_xlabel(r"$p/(m_e c)$")
            ax.set_ylabel(r"$\xi$")
            fig.colorbar(image, ax=ax, label=r"$\log_{10}\langle f\rangle$ (arb.)")
        fig.tight_layout()
        fig.savefig(output_dir / "runaway_vortex_phase_space_flux.png", dpi=300)
        plt.close(fig)

    return summary_path


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runaway_vortex_results"))
    parser.add_argument("--markers", type=int, default=4096)
    parser.add_argument("--dt", type=float, default=DEFAULT_VORTEX_DT)
    parser.add_argument("--total-time", type=float, default=DEFAULT_VORTEX_TOTAL_TIME)
    parser.add_argument("--burn-time", type=float, default=DEFAULT_VORTEX_BURN_TIME)
    parser.add_argument("--bins", type=int, default=DEFAULT_VORTEX_BINS)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--paper", action="store_true", help="GPU-oriented high-statistics settings")
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.paper:
        args.markers = max(args.markers, 16_384)
        args.dt = min(args.dt, 2.0e-3)
        args.total_time = max(args.total_time, 100.0)
        args.burn_time = max(args.burn_time, 50.0)
        args.bins = max(args.bins, 480)

    kernel = _make_vortex_kernel(
        n_bins=args.bins,
        phase_bins=(120, 72),
    )
    results = []
    for offset, e_over_ec in enumerate((2.0, 2.25, 2.5)):
        results.append(
            run_particle_vortex_case(
                kernel,
                e_over_ec,
                n_markers=args.markers,
                seed=args.seed + offset,
                dt=args.dt,
                total_time=args.total_time,
                burn_time=args.burn_time,
                sample_every=DEFAULT_SAMPLE_EVERY,
                phase_sample_every=25,
            )
        )
    summary = _write_particle_vortex_results(results, args.output_dir)
    for result in results:
        ref = published_bump_points().get(result["e_over_ec"], np.nan)
        if result["has_bump"]:
            rel = (result["p_bump"] - ref) / ref if np.isfinite(ref) else np.nan
            print(
                f"E/Ec={result['e_over_ec']:.2f}: p_b={result['p_bump']:.5f} "
                f"+/- {result['p_bump_sem']:.5f}, published={ref:.8g}, rel.err={rel:+.2%}"
            )
        else:
            print(f"E/Ec={result['e_over_ec']:.2f}: no pitch-integrated runaway bump")
    print(f"wrote {summary}")


if __name__ == "__main__":
    main()
