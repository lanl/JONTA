"""Axisymmetric neoclassical spatial-transport benchmarks.

This reproduces Figs. 3 and 6 and Appendix A of McDevitt, Guo & Tang,
Plasma Phys. Control. Fusion 61, 024004 (2019).  Figure 3 validates the fully
ionized low-Z pitch-scattering branch; Figure 6 repeats the calculation for a
deuterium plasma with singly ionized argon at n_Ar+ = n_D/10 and validates the
partially screened Hesslow/RAMc pitch-scattering coefficients.  The published
marker values and visible error-bar limits are digitized into
``reference_data`` and overlaid directly on JONTA output.  The analytic
scalings from Eq. (1) and Eqs. (7)-(8) are evaluated independently.
Particles are monoenergetic and isotropic in pitch.  The inductive electric
field, synchrotron radiation, collisional drag, energy diffusion, and
large-angle collisions are disabled; only guiding-center motion and
small-angle pitch scattering remain.

The transport coefficients are estimated from markers binned by *initial*
radius,

    V(r) = d <Delta r> / dt,
    D(r) = 1/2 d [<Delta r^2> - <Delta r>^2] / dt.

Both paper benchmarks use c*tau_c/a = 5e6 and integrate for 20 tau_c.  The
marker radii are taken from the digitized abscissae because the captions do
not list them explicitly.  The full cases are exposed as ``--paper`` and are
intended for GPU runs.
The pytest/CPU mode uses a shorter, less asymptotic case so continuous
integration remains practical while exercising the identical JIT kernel.

The hot loop is a single JIT-compiled ``jax.lax.scan``.  Particle pushes,
pitch-scattering kicks, radial binning, and diagnostic moments stay on device;
only the final moment history is transferred to the host for linear fits.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from collisions.coulomb import thermal_coulomb_log
from collisions.small_angle import pitch_scattering_frequency, small_angle_step
from core.config import OrbitNormalization, SmallAngleConfig
from core.constants import C_LIGHT_M_S, E_CHARGE_C, M_E_KG, ME_C2_EV, R_E_CM
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles, QuadraticCircularFieldProfiles
from diagnostics.transport import fit_transport_coefficients
from integrators import rk4_step
from orbits.ramc_circular import ramc_circular_rhs

jax.config.update("jax_enable_x64", True)

# Figure-3 parameters.
PAPER_EPSILON = 1.0 / 3.0
PAPER_C_TAU_OVER_A = 5.0e6
PAPER_Z = 2.0
PAPER_VTE_OVER_C = 0.0063
PAPER_Q0 = 2.1
PAPER_Q2 = 2.0
PAPER_A_M = 2.0
PAPER_B0_T = 5.3
PAPER_FINAL_TIME = 20.0
PAPER_ENERGIES_EV = (1.0e4, 1.0e5, 5.0e5, 1.5e6)
# Figure-3 marker abscissae digitized from the published plot.  Using the
# digitized x-locations avoids silently assuming round radii that were not
# specified in the caption.
PAPER_RADII = (0.222203, 0.333215, 0.444287, 0.555299, 0.666489, 0.777561)
REFERENCE_DATA_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "one_d" / "radial_transport" / "reference" / "mcdevitt_2019_fig3.csv"

# Figure-6 partially screened benchmark.  The PDF caption gives a deuterium
# background with singly ionized argon at n_Ar+ = n_D/10.  The RAMc source
# distributed with the project uses the Hesslow coefficients with these
# argon fit parameters.
FIG6_REFERENCE_DATA_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "one_d" / "radial_transport" / "reference" / "mcdevitt_2019_fig6.csv"
FIG6_MAIN_Z = 1.0
FIG6_IMPURITY_FRACTION = 0.1  # n_Ar+ / n_D
FIG6_Z0 = 18.0
FIG6_ZI = 1.0
FIG6_AI = 0.329
FIG6_SCREENING_K = 5.0

# CPU demonstration/CI parameters.  The high-energy cases are already deep in
# the banana regime at this transit/collision ratio; 10 keV is deliberately
# retained as a finite-collisionality stress case.  The full paper values are
# available through --paper without changing the kernel.
CPU_C_TAU_OVER_A = 1.0e5
CPU_FINAL_TIME = 1.0
CPU_DT = 1.0e-5
CPU_COLLISION_DT = 5.0e-5
CPU_RADII = (0.25, 0.45, 0.65, 0.80)


def _te_from_vte(vte_over_c=PAPER_VTE_OVER_C):
    return 0.5 * ME_C2_EV * vte_over_c * vte_over_c


def _a_omega_ce_over_c(a_m=PAPER_A_M, b0_t=PAPER_B0_T):
    omega_ce = E_CHARGE_C * b0_t / M_E_KG
    return a_m * omega_ce / C_LIGHT_M_S


def _constant_background(te_ev: float, ne_cm3: float = 1.0e14, zeff: float = PAPER_Z):
    grid = jnp.linspace(0.0, 1.0, 16)
    return BackgroundProfiles(
        r=grid,
        ne_cm3=jnp.full_like(grid, ne_cm3),
        te_ev=jnp.full_like(grid, te_ev),
        ti_ev=jnp.full_like(grid, te_ev),
        zeff=jnp.full_like(grid, zeff),
        eta_bar=jnp.ones_like(grid),
    )


def _ne_from_c_tau_over_a(c_tau_over_a: float, te_ev: float, a_m: float = PAPER_A_M):
    """Solve the RAMc normalization relation for the total free density.

    RAMc uses

        c*tau_c/a = 1 / (4*pi*lnLambda0*n_e*r_e^2*a),

    with ``n_e`` in cm^-3, ``r_e`` and ``a`` in cm, and a thermal Coulomb
    logarithm that itself depends weakly on ``n_e``.  Figure 3 is insensitive
    to this absolute density because only lnLambda/lnLambda0=1 appears, but
    Figure 6 needs the actual lnLambda0 in the partially screened coefficients.
    """

    a_cm = 100.0 * a_m
    ne = 1.0e14
    for _ in range(32):
        coulog = float(thermal_coulomb_log(ne, te_ev))
        ne = 1.0 / (4.0 * np.pi * coulog * R_E_CM * R_E_CM * a_cm * c_tau_over_a)
    return float(ne)


def _collision_config(
    te_ev: float,
    ne_cm3: float = 1.0e14,
    pitch_scattering: bool = True,
    collision_model: str = "fully_ionized",
):
    coulog = float(thermal_coulomb_log(ne_cm3, te_ev))
    common = dict(
        ne0_cm3=ne_cm3,
        coulog0=coulog,
        pitch_scattering=pitch_scattering,
        friction=False,
        energy_scattering=False,
        max_nu_dt=0.10,
    )
    if collision_model == "fully_ionized":
        return SmallAngleConfig(**common)
    if collision_model == "partial_screening":
        return SmallAngleConfig(
            **common,
            partial_screening=True,
            impurity_fraction=FIG6_IMPURITY_FRACTION,
            impurity_nuclear_charge=FIG6_Z0,
            impurity_charge_state=FIG6_ZI,
            impurity_radius_abohr=FIG6_AI,
            screening_k=FIG6_SCREENING_K,
        )
    raise ValueError(f"unknown collision_model: {collision_model}")


def make_transport_case(
    *,
    energies_ev=PAPER_ENERGIES_EV,
    radii=CPU_RADII,
    markers_per_group=32,
    n_replicates=1,
    c_tau_over_a=CPU_C_TAU_OVER_A,
    seed=1234,
    pitch_scattering=True,
    collision_model="fully_ionized",
):
    """Build one fixed-shape ensemble containing every energy/radius group."""

    rng = np.random.default_rng(seed)
    energies_ev = np.asarray(energies_ev, dtype=float)
    radii = np.asarray(radii, dtype=float)
    n_energy = energies_ev.size
    n_radius = radii.size
    if n_replicates < 1:
        raise ValueError("n_replicates must be positive")
    if markers_per_group < n_replicates or markers_per_group % n_replicates:
        raise ValueError("markers_per_group must be divisible by n_replicates")
    markers_per_replicate = markers_per_group // n_replicates
    n_physical_group = n_energy * n_radius
    n_group = n_physical_group * n_replicates
    n_total = n_physical_group * markers_per_group

    energy_index = np.repeat(np.arange(n_energy), n_radius * markers_per_group)
    radius_index = np.tile(
        np.repeat(np.arange(n_radius), markers_per_group),
        n_energy,
    )
    marker_in_group = np.tile(np.arange(markers_per_group), n_physical_group)
    replicate_index = marker_in_group // markers_per_replicate
    physical_group_index = energy_index * n_radius + radius_index
    group_index = physical_group_index * n_replicates + replicate_index

    kinetic_ev = energies_ev[energy_index]
    r0 = radii[radius_index]
    theta0 = rng.uniform(0.0, 2.0 * np.pi, n_total)
    phi0 = rng.uniform(-np.pi, np.pi, n_total)
    # Isotropic distribution in velocity direction => xi uniform on [-1, 1].
    xi0 = rng.uniform(-1.0, 1.0, n_total)
    gamma0 = 1.0 + kinetic_ev / ME_C2_EV

    particles = particles_from_arrays(
        gamma=gamma0,
        xi=xi0,
        x=r0 * np.cos(theta0),
        y=r0 * np.sin(theta0),
        phi=phi0,
        weight=np.ones(n_total),
        pid=np.arange(n_total, dtype=np.int64),
    )

    fields = QuadraticCircularFieldProfiles(
        e1=jnp.asarray(0.0, dtype=jnp.float64),
        q0=jnp.asarray(PAPER_Q0, dtype=jnp.float64),
        q2=jnp.asarray(PAPER_Q2, dtype=jnp.float64),
    )
    norm = OrbitNormalization(
        epsilon=PAPER_EPSILON,
        c_tau_over_a=float(c_tau_over_a),
        a_omega_ce_over_c=_a_omega_ce_over_c(),
        alpha_syn=0.0,
    )
    te_ev = _te_from_vte()
    if collision_model == "partial_screening":
        ne_cm3 = _ne_from_c_tau_over_a(c_tau_over_a, te_ev)
        zeff = (
            1.0 + FIG6_ZI * FIG6_ZI * FIG6_IMPURITY_FRACTION
        ) / (1.0 + FIG6_ZI * FIG6_IMPURITY_FRACTION)
        main_z = FIG6_MAIN_Z
        tau_main_over_tau_total = 1.0 + FIG6_ZI * FIG6_IMPURITY_FRACTION
    elif collision_model == "fully_ionized":
        ne_cm3 = 1.0e14
        zeff = PAPER_Z
        main_z = PAPER_Z
        tau_main_over_tau_total = 1.0
    else:
        raise ValueError(f"unknown collision_model: {collision_model}")

    background = _constant_background(te_ev, ne_cm3=ne_cm3, zeff=zeff)
    collision_config = _collision_config(
        te_ev,
        ne_cm3=ne_cm3,
        pitch_scattering=pitch_scattering,
        collision_model=collision_model,
    )

    return {
        "particles": particles,
        "fields": fields,
        "norm": norm,
        "background": background,
        "collision_config": collision_config,
        "r0": jnp.asarray(r0),
        "group_index": jnp.asarray(group_index, dtype=jnp.int32),
        "group_counts": jnp.bincount(
            jnp.asarray(group_index, dtype=jnp.int32), length=n_group
        ),
        "energies_ev": energies_ev,
        "radii": radii,
        "n_group": n_group,
        "n_replicates": int(n_replicates),
        "markers_per_replicate": int(markers_per_replicate),
        "collision_model": collision_model,
        "ne_cm3": float(ne_cm3),
        "zeff": float(zeff),
        "main_z": float(main_z),
        "tau_main_over_tau_total": float(tau_main_over_tau_total),
    }


def _make_transport_kernel(base_key, *, n_inner: int, collision_stride: int, n_samples: int):
    """Create one GPU-ready compiled kernel returning binned radial moments."""

    if n_inner <= 0 or collision_stride <= 0:
        raise ValueError("n_inner and collision_stride must be positive")

    @jax.jit
    def kernel(
        particles,
        fields,
        norm,
        background,
        collision_config,
        r0,
        group_index,
        group_counts,
        dt,
    ):
        n_group = group_counts.shape[0]

        def orbit_rhs(y, t):
            return ramc_circular_rhs(y, t, fields, norm)

        def inner_step(local_i, carry):
            state, global_i = carry
            time = dt * global_i
            kin_new = rk4_step(orbit_rhs, state.kin, time, dt)
            state = state._replace(kin=kin_new)
            next_i = global_i + 1

            # Collision operator is subcycled independently from the orbit
            # pusher.  A finite pitch kick represents the integrated
            # small-angle operator over collision_stride orbit steps.
            do_collision = (next_i % collision_stride) == 0

            def collide(s):
                return small_angle_step(
                    s,
                    background,
                    dt * collision_stride,
                    base_key,
                    next_i // collision_stride,
                    collision_config,
                )

            state = jax.lax.cond(do_collision, collide, lambda s: s, state)
            return state, next_i

        def sample_step(carry, _):
            state, global_i = carry
            state, global_i = jax.lax.fori_loop(
                0,
                n_inner,
                inner_step,
                (state, global_i),
            )
            r = jnp.sqrt(state.kin.x * state.kin.x + state.kin.y * state.kin.y)
            dr = r - r0
            sum_dr = jnp.bincount(group_index, weights=dr, length=n_group)
            sum_dr2 = jnp.bincount(group_index, weights=dr * dr, length=n_group)
            mean_dr = sum_dr / group_counts
            mean_dr2 = sum_dr2 / group_counts
            return (state, global_i), (mean_dr, mean_dr2)

        (_, _), (mean_dr, mean_dr2) = jax.lax.scan(
            sample_step,
            (particles, jnp.asarray(0, dtype=jnp.int32)),
            xs=None,
            length=n_samples,
        )
        return mean_dr, mean_dr2

    return kernel


def reference_diffusivity_d0(
    radii,
    c_tau_over_a=PAPER_C_TAU_OVER_A,
    main_z=PAPER_Z,
):
    """Return McDevitt et al. Eq. (3): tau_c D0 / a^2."""

    del c_tau_over_a  # D0 in tau_c units is independent of c*tau_c/a.
    radii = np.asarray(radii, dtype=float)
    eps_local = PAPER_EPSILON * radii
    q = PAPER_Q0 + PAPER_Q2 * radii * radii
    inv_awce = 1.0 / _a_omega_ce_over_c()
    return (
        0.689
        * np.sqrt(2.0 * eps_local)
        * inv_awce**2
        * (q / (PAPER_EPSILON * radii)) ** 2
        * (main_z + 1.0)
        / 2.0
    )


def load_mcdevitt_fig3_reference(path: Path = REFERENCE_DATA_PATH):
    """Load digitized Figure-3 markers and visible error-bar limits.

    The CSV is a plot digitization rather than the authors' raw numerical data.
    See ``benchmarks/one_d/radial_transport/reference/metadata.yaml`` for provenance and axis calibration.
    """

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return data


def load_mcdevitt_fig6_reference(path: Path = FIG6_REFERENCE_DATA_PATH):
    """Load digitized Figure-6 partially screened markers and error bars."""

    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.asarray([data], dtype=data.dtype)
    return data


def mcdevitt_nonrelativistic_scaling(energy_ev):
    """Return the Figure-3 Eq.-(1) nonrelativistic scaling ``D/D0 = c/v``.

    For a fully ionized pure-Z plasma, Eq. (1) uses
    ``A f_t ~= 0.689 sqrt(2 epsilon)``.  Dividing by the reference diffusivity
    in Eq. (3) cancels the geometry and charge factors, leaving ``c/v``.
    The published Figure 3 plots this scaling only for the 10-keV population.
    """

    energy_ev = np.asarray(energy_ev, dtype=float)
    if np.any(energy_ev <= 0.0):
        raise ValueError("energy_ev must be positive")
    v_over_c = np.sqrt(2.0 * energy_ev / ME_C2_EV)
    return 1.0 / v_over_c


def mcdevitt_partial_screening_scaling(
    energy_ev,
    *,
    c_tau_over_a=PAPER_C_TAU_OVER_A,
):
    """Return the Figure-6 banana-regime estimate from Eqs. (7)-(8).

    The Figure-6 distribution is isotropic.  Using the same
    small-inverse-aspect-ratio trapped-fraction factor that defines ``D0`` in
    Eq. (3), the geometry cancels and the relativistic estimate becomes

        D/D0 ~= (tau_c^a/tau_c) * C_B / (Z+1),

    where ``C_B = p^2 tau_c nu_D^ps`` and ``Z=1`` for the deuterium main ion.
    The extra collision-time ratio is required because Eq. (3) uses the free
    electrons associated with the main ion, whereas Figure 6 normalizes time
    with the total free-electron density.
    """

    scalar_input = np.ndim(energy_ev) == 0
    energy_ev = np.atleast_1d(np.asarray(energy_ev, dtype=float))
    te_ev = _te_from_vte()
    ne_cm3 = _ne_from_c_tau_over_a(c_tau_over_a, te_ev)
    coulog0 = float(thermal_coulomb_log(ne_cm3, te_ev))
    zeff = (
        1.0 + FIG6_ZI * FIG6_ZI * FIG6_IMPURITY_FRACTION
    ) / (1.0 + FIG6_ZI * FIG6_IMPURITY_FRACTION)
    cfg = _collision_config(
        te_ev,
        ne_cm3=ne_cm3,
        pitch_scattering=True,
        collision_model="partial_screening",
    )
    gamma = jnp.asarray(1.0 + energy_ev / ME_C2_EV, dtype=jnp.float64)
    p = jnp.sqrt(gamma * gamma - 1.0)
    nu_d = pitch_scattering_frequency(
        gamma,
        jnp.full_like(gamma, zeff),
        jnp.ones_like(gamma),
        jnp.full_like(gamma, te_ev),
        jnp.full_like(gamma, coulog0),
        cfg,
    )
    cb = np.asarray(jax.device_get(p * p * nu_d), dtype=float)
    tau_ratio = 1.0 + FIG6_ZI * FIG6_IMPURITY_FRACTION
    out = tau_ratio * cb / (FIG6_MAIN_Z + 1.0)
    if scalar_input:
        return float(out[0])
    return out


def _compare_with_reference(result, reference):
    """Interpolate a digitized McDevitt transport figure to a JONTA grid."""

    rows = []
    for ie, energy in enumerate(result["energies_ev"]):
        mask = np.isclose(reference["energy_eV"], energy, rtol=0.0, atol=1.0e-9)
        if not np.any(mask):
            continue
        ref = reference[mask]
        order = np.argsort(ref["r_over_a"])
        rr = np.asarray(ref["r_over_a"][order], dtype=float)
        yy = np.asarray(ref["D_over_D0"][order], dtype=float)
        lo = np.asarray(ref["D_over_D0_lower"][order], dtype=float)
        hi = np.asarray(ref["D_over_D0_upper"][order], dtype=float)
        for ir, radius in enumerate(result["radii"]):
            if radius < rr[0] or radius > rr[-1]:
                continue
            paper = float(np.interp(radius, rr, yy))
            lower = float(np.interp(radius, rr, lo))
            upper = float(np.interp(radius, rr, hi))
            jonta = float(result["D_over_D0"][ie, ir])
            jonta_sem = float(result["D_over_D0_sem"][ie, ir])
            paper_sigma = 0.5 * (upper - lower)
            combined_sigma = np.sqrt(paper_sigma * paper_sigma + jonta_sem * jonta_sem)
            rows.append(
                {
                    "energy_eV": float(energy),
                    "r_over_a": float(radius),
                    "jonta_D_over_D0": jonta,
                    "jonta_sem": jonta_sem,
                    "paper_D_over_D0": paper,
                    "paper_lower": lower,
                    "paper_upper": upper,
                    "relative_error": (jonta - paper) / paper,
                    "combined_z": (jonta - paper) / combined_sigma if combined_sigma > 0 else np.nan,
                    "inside_paper_errorbar": lower <= jonta <= upper,
                }
            )
    return rows


def compare_with_mcdevitt_fig3(result, reference=None):
    """Interpolate digitized Figure-3 data to a JONTA result grid."""

    if reference is None:
        reference = load_mcdevitt_fig3_reference()
    return _compare_with_reference(result, reference)


def compare_with_mcdevitt_fig6(result, reference=None):
    """Interpolate digitized Figure-6 data to a JONTA result grid."""

    if reference is None:
        reference = load_mcdevitt_fig6_reference()
    return _compare_with_reference(result, reference)


def _reference_for_result(result):
    if result["collision_model"] == "partial_screening":
        return 6, load_mcdevitt_fig6_reference(), compare_with_mcdevitt_fig6
    return 3, load_mcdevitt_fig3_reference(), compare_with_mcdevitt_fig3


def run_transport_scan(
    *,
    energies_ev=PAPER_ENERGIES_EV,
    radii=CPU_RADII,
    markers_per_group=32,
    n_replicates=1,
    c_tau_over_a=CPU_C_TAU_OVER_A,
    final_time=CPU_FINAL_TIME,
    dt=CPU_DT,
    collision_dt=CPU_COLLISION_DT,
    n_samples=80,
    seed=1234,
    pitch_scattering=True,
    collision_model="fully_ionized",
):
    """Run the orbit+pitch-scattering transport scan and fit D(r)."""

    n_steps = max(1, int(round(final_time / dt)))
    dt = final_time / n_steps
    collision_stride = max(1, int(round(collision_dt / dt)))
    collision_dt = collision_stride * dt
    n_inner = max(1, n_steps // n_samples)
    n_samples = n_steps // n_inner
    n_steps_used = n_samples * n_inner
    final_time_used = n_steps_used * dt

    case = make_transport_case(
        energies_ev=energies_ev,
        radii=radii,
        markers_per_group=markers_per_group,
        n_replicates=n_replicates,
        c_tau_over_a=c_tau_over_a,
        seed=seed,
        pitch_scattering=pitch_scattering,
        collision_model=collision_model,
    )
    kernel = _make_transport_kernel(
        jax.random.key(seed),
        n_inner=n_inner,
        collision_stride=collision_stride,
        n_samples=n_samples,
    )
    mean_dr, mean_dr2 = kernel(
        case["particles"],
        case["fields"],
        case["norm"],
        case["background"],
        case["collision_config"],
        case["r0"],
        case["group_index"],
        case["group_counts"],
        jnp.asarray(dt, dtype=jnp.float64),
    )
    mean_dr, mean_dr2 = jax.device_get((mean_dr, mean_dr2))
    time = np.arange(1, n_samples + 1, dtype=float) * (n_inner * dt)
    fits = fit_transport_coefficients(time, mean_dr, mean_dr2, fit_start_fraction=0.25)

    n_energy = len(energies_ev)
    n_radius = len(radii)
    n_rep = int(case["n_replicates"])
    d_rep = fits["D"].reshape(n_energy, n_radius, n_rep)
    v_rep = fits["V"].reshape(n_energy, n_radius, n_rep)
    r2_rep = fits["r2_variance"].reshape(n_energy, n_radius, n_rep)

    mean_dr_rep = np.asarray(mean_dr).reshape(n_samples, n_energy, n_radius, n_rep)
    mean_dr2_rep = np.asarray(mean_dr2).reshape(n_samples, n_energy, n_radius, n_rep)
    variance_rep = np.asarray(fits["variance"]).reshape(n_samples, n_energy, n_radius, n_rep)
    mean_dr_pooled = np.mean(mean_dr_rep, axis=3)
    mean_dr2_pooled = np.mean(mean_dr2_rep, axis=3)

    # The central estimate must be formed from the full pooled ensemble.
    # Averaging per-batch diffusion coefficients is biased because each batch
    # subtracts its own finite-sample <Delta r>^2.  Replicates are used only
    # to estimate statistical uncertainty.
    pooled_fits = fit_transport_coefficients(
        time,
        mean_dr_pooled.reshape(n_samples, n_energy * n_radius),
        mean_dr2_pooled.reshape(n_samples, n_energy * n_radius),
        fit_start_fraction=0.25,
    )
    d = pooled_fits["D"].reshape(n_energy, n_radius)
    v = pooled_fits["V"].reshape(n_energy, n_radius)
    r2 = pooled_fits["r2_variance"].reshape(n_energy, n_radius)
    variance_pooled = pooled_fits["variance"].reshape(n_samples, n_energy, n_radius)
    variance_slope = pooled_fits["variance_slope"].reshape(n_energy, n_radius)
    variance_intercept = pooled_fits["variance_intercept"].reshape(n_energy, n_radius)
    fit_start_index = int(pooled_fits["fit_start_index"])
    if n_rep > 1:
        d_sem = np.std(d_rep, axis=2, ddof=1) / np.sqrt(n_rep)
        v_sem = np.std(v_rep, axis=2, ddof=1) / np.sqrt(n_rep)
        variance_sem = np.std(variance_rep, axis=3, ddof=1) / np.sqrt(n_rep)
    else:
        d_sem = np.full_like(d, np.nan)
        v_sem = np.full_like(v, np.nan)
        variance_sem = np.full_like(variance_pooled, np.nan)
    d0 = reference_diffusivity_d0(radii, main_z=case["main_z"])
    tau_ratio = case["tau_main_over_tau_total"]

    return {
        "time": time,
        "mean_dr": mean_dr_pooled,
        "mean_dr2": mean_dr2_pooled,
        "variance": variance_pooled,
        "variance_sem": variance_sem,
        "variance_slope": variance_slope,
        "variance_intercept": variance_intercept,
        "fit_start_index": fit_start_index,
        "mean_dr_replicates": mean_dr_rep,
        "mean_dr2_replicates": mean_dr2_rep,
        "variance_replicates": variance_rep,
        "D": d,
        "D_sem": d_sem,
        "D_replicates": d_rep,
        "V": v,
        "V_sem": v_sem,
        "V_replicates": v_rep,
        "r2_variance": r2,
        "r2_variance_replicates": r2_rep,
        "D0": d0,
        "D_over_D0": tau_ratio * d / d0[None, :],
        "D_over_D0_sem": tau_ratio * d_sem / d0[None, :],
        "D_over_D0_replicates": tau_ratio * d_rep / d0[None, :, None],
        "energies_ev": np.asarray(energies_ev, dtype=float),
        "radii": np.asarray(radii, dtype=float),
        "markers_per_group": markers_per_group,
        "n_replicates": n_rep,
        "markers_per_replicate": case["markers_per_replicate"],
        "c_tau_over_a": c_tau_over_a,
        "dt": dt,
        "collision_dt": collision_dt,
        "n_steps": n_steps_used,
        "final_time": final_time_used,
        "collision_model": case["collision_model"],
        "ne_cm3": case["ne_cm3"],
        "zeff": case["zeff"],
        "main_z": case["main_z"],
        "tau_main_over_tau_total": tau_ratio,
    }


def test_spatial_transport_pitch_scattering_produces_radial_diffusion():
    """GPU-ready smoke/convergence test of the Fig.-3 transport pathway."""

    kwargs = dict(
        energies_ev=(1.0e5, 5.0e5, 1.5e6),
        radii=(0.35, 0.60),
        markers_per_group=12,
        c_tau_over_a=5.0e4,
        final_time=0.08,
        dt=2.0e-5,
        collision_dt=1.0e-4,
        n_samples=24,
        seed=901,
    )
    collisional = run_transport_scan(**kwargs, pitch_scattering=True)
    collisionless = run_transport_scan(**kwargs, pitch_scattering=False)

    assert np.all(np.isfinite(collisional["D"]))
    assert np.all(np.isfinite(collisional["D_over_D0"]))

    # Compare with an otherwise identical collisionless ensemble.  This makes
    # the smoke test insensitive to the bounded finite-orbit-width oscillation
    # in instantaneous r and verifies that the stochastic pitch operator adds
    # net radial broadening.
    final_var_on = np.mean(collisional["variance"][-1])
    final_var_off = np.mean(collisionless["variance"][-1])
    assert final_var_on > final_var_off
    assert np.median(collisional["D"] - collisionless["D"]) > 0.0


def test_mcdevitt_fig3_reference_and_analytic_scaling():
    """The checked-in digitization and Eq.-(1) scaling are internally sane."""

    reference = load_mcdevitt_fig3_reference()
    assert reference.size == 24
    assert set(np.unique(reference["energy_eV"])) == set(PAPER_ENERGIES_EV)
    assert np.all(reference["D_over_D0_lower"] <= reference["D_over_D0"])
    assert np.all(reference["D_over_D0"] <= reference["D_over_D0_upper"])

    # The 10-keV dashed line digitizes to about 5.02.  Computing Eq. (1)
    # independently gives 5.05; agreement at the percent level is expected
    # for figure digitization.
    scaling_10kev = float(mcdevitt_nonrelativistic_scaling(1.0e4))
    assert scaling_10kev == pytest.approx(5.02, rel=0.02)


def test_mcdevitt_fig6_reference_and_partial_screening_scaling():
    """Figure 6 and the Eq.-(7)-(8) partial-screening estimate are sane."""

    reference = load_mcdevitt_fig6_reference()
    assert reference.size == 24
    assert set(np.unique(reference["energy_eV"])) == set(PAPER_ENERGIES_EV)
    assert np.all(reference["D_over_D0_lower"] <= reference["D_over_D0"])
    assert np.all(reference["D_over_D0"] <= reference["D_over_D0_upper"])

    scaling = np.asarray(mcdevitt_partial_screening_scaling(PAPER_ENERGIES_EV))
    assert np.all(np.isfinite(scaling))
    # Partial screening prevents the MeV transport from collapsing to the
    # fully ionized D/D0~1 asymptote.  The paper's Figure 6 sits around 10-25.
    assert np.min(scaling) > 8.0
    assert np.max(scaling) < 30.0

    # The asymptotic estimate is most appropriate away from the small-radius
    # finite-collisionality region.  Require order-20% agreement with the
    # outermost digitized points as an independent equation/data cross-check.
    outer = reference[np.isclose(reference["r_over_a"], PAPER_RADII[-1])]
    outer = outer[np.argsort(outer["energy_eV"])]
    assert np.allclose(scaling, outer["D_over_D0"], rtol=0.20, atol=0.0)


def test_partial_screening_enhances_axisymmetric_diffusion():
    """The Figure-6 collision model must diffuse faster than Figure 3."""

    kwargs = dict(
        energies_ev=(5.0e5,),
        radii=(0.40, 0.65),
        markers_per_group=24,
        n_replicates=2,
        c_tau_over_a=2.0e5,
        final_time=0.12,
        dt=1.0e-5,
        collision_dt=5.0e-5,
        n_samples=32,
        seed=1776,
    )
    fully_ionized = run_transport_scan(**kwargs, collision_model="fully_ionized")
    screened = run_transport_scan(**kwargs, collision_model="partial_screening")

    assert np.all(np.isfinite(screened["D"]))
    assert np.median(screened["D"] / fully_ionized["D"]) > 3.0


def _write_csv(path: Path, result):
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "energy_eV",
                "r_over_a",
                "D_tau_c_over_a2",
                "D0_tau_c_over_a2",
                "D_over_D0",
                "D_over_D0_SEM",
                "V_tau_c_over_a",
                "V_SEM_tau_c_over_a",
                "variance_fit_R2",
                "markers",
                "n_replicates",
                "markers_per_replicate",
                "c_tau_over_a",
                "dt_tau_c",
                "collision_dt_tau_c",
                "final_time_tau_c",
                "collision_model",
                "ne_cm3",
                "zeff",
                "main_Z",
                "tau_c_main_over_tau_c_total",
            ]
        )
        for ie, energy in enumerate(result["energies_ev"]):
            for ir, radius in enumerate(result["radii"]):
                writer.writerow(
                    [
                        energy,
                        radius,
                        result["D"][ie, ir],
                        result["D0"][ir],
                        result["D_over_D0"][ie, ir],
                        result["D_over_D0_sem"][ie, ir],
                        result["V"][ie, ir],
                        result["V_sem"][ie, ir],
                        result["r2_variance"][ie, ir],
                        result["markers_per_group"],
                        result["n_replicates"],
                        result["markers_per_replicate"],
                        result["c_tau_over_a"],
                        result["dt"],
                        result["collision_dt"],
                        result["final_time"],
                        result["collision_model"],
                        result["ne_cm3"],
                        result["zeff"],
                        result["main_z"],
                        result["tau_main_over_tau_total"],
                    ]
                )


def _write_comparison_csv(path: Path, result):
    _fig_number, _reference, compare = _reference_for_result(result)
    rows = compare(result)
    with path.open("w", newline="") as f:
        fieldnames = [
            "energy_eV",
            "r_over_a",
            "jonta_D_over_D0",
            "jonta_sem",
            "paper_D_over_D0",
            "paper_lower",
            "paper_upper",
            "relative_error",
            "combined_z",
            "inside_paper_errorbar",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_variance_history_csv(path: Path, result):
    """Write the Appendix-A variance histories and the fitted diffusion lines."""

    fit_start = int(result["fit_start_index"])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "time_tau_c",
                "energy_eV",
                "r_over_a",
                "variance",
                "variance_SEM",
                "linear_fit",
                "in_fit_window",
                "variance_fit_R2",
                "D_tau_c_over_a2",
                "D_over_D0",
            ]
        )
        for it, time in enumerate(result["time"]):
            for ie, energy in enumerate(result["energies_ev"]):
                for ir, radius in enumerate(result["radii"]):
                    fit_value = (
                        result["variance_intercept"][ie, ir]
                        + result["variance_slope"][ie, ir] * time
                    )
                    writer.writerow(
                        [
                            time,
                            energy,
                            radius,
                            result["variance"][it, ie, ir],
                            result["variance_sem"][it, ie, ir],
                            fit_value,
                            int(it >= fit_start),
                            result["r2_variance"][ie, ir],
                            result["D"][ie, ir],
                            result["D_over_D0"][ie, ir],
                        ]
                    )


def _plot_results(outdir: Path, result):
    import matplotlib.pyplot as plt

    fig_number, reference, compare = _reference_for_result(result)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(7.6, 5.2))
    plot_energies = tuple(float(x) for x in result["energies_ev"])
    for ie, energy in enumerate(plot_energies):
        color = colors[ie % len(colors)]
        label = f"{energy / 1e3:g} keV" if energy < 1e6 else f"{energy / 1e6:g} MeV"
        mask = np.isclose(reference["energy_eV"], energy, rtol=0.0, atol=1.0e-9)
        ref = reference[mask]
        yerr = np.vstack(
            [
                ref["D_over_D0"] - ref["D_over_D0_lower"],
                ref["D_over_D0_upper"] - ref["D_over_D0"],
            ]
        )
        ax.errorbar(
            ref["r_over_a"],
            ref["D_over_D0"],
            yerr=yerr,
            fmt="o",
            fillstyle="none",
            capsize=2.5,
            color=color,
            label=f"McDevitt 2019: {label}",
        )

        matches = np.where(np.isclose(result["energies_ev"], energy))[0]
        if matches.size:
            j = int(matches[0])
            yerr_jonta = result["D_over_D0_sem"][j]
            if np.all(np.isnan(yerr_jonta)):
                yerr_jonta = None
            ax.errorbar(
                result["radii"],
                result["D_over_D0"][j],
                yerr=yerr_jonta,
                fmt="x-",
                capsize=3.0,
                color=color,
                linewidth=1.0,
                label=f"JONTA: {label}",
            )

    if result["collision_model"] == "partial_screening":
        analytic = np.asarray(
            mcdevitt_partial_screening_scaling(
                result["energies_ev"], c_tau_over_a=result["c_tau_over_a"]
            )
        )
        for ie, (energy, value) in enumerate(zip(result["energies_ev"], analytic)):
            color = colors[ie % len(colors)]
            label = f"{energy / 1e3:g} keV" if energy < 1e6 else f"{energy / 1e6:g} MeV"
            ax.axhline(
                value,
                linestyle="--",
                linewidth=1.0,
                color=color,
                alpha=0.7,
                label=f"Eqs. (7)-(8): {label}",
            )
    else:
        analytic = float(mcdevitt_nonrelativistic_scaling(1.0e4))
        ax.axhline(
            analytic,
            linestyle="--",
            linewidth=1.2,
            label="McDevitt Eq. (1), 10 keV: D/D0 = c/v",
        )
    if len(plot_energies) > 1:
        ax.set_yscale("log")
    ax.set_xlim(0.15, 0.85)
    ax.set_xlabel("r/a")
    ax.set_ylabel("D / D0")
    model_label = (
        "partially screened Ar+" if result["collision_model"] == "partial_screening"
        else "fully ionized"
    )
    title = (
        f"McDevitt et al. (2019) Figure {fig_number} spatial-transport benchmark"
        f"\n({model_label})"
    )
    if (
        not np.isclose(result["c_tau_over_a"], PAPER_C_TAU_OVER_A)
        or not np.isclose(result["final_time"], PAPER_FINAL_TIME)
    ):
        title += (
            f"\n(preliminary JONTA run: t={result['final_time']:.3g} tau_c; "
            f"published case uses {PAPER_FINAL_TIME:g} tau_c)"
        )
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outdir / f"spatial_transport_figure{fig_number}_comparison.png", dpi=360)
    plt.close(fig)

    # Relative-difference panel makes statistically significant disagreement
    # visible rather than hiding it on the log-scale overlay.
    comparison = compare(result, reference)
    if comparison:
        fig, ax = plt.subplots(figsize=(7.6, 4.2))
        for ie, energy in enumerate(plot_energies):
            color = colors[ie % len(colors)]
            rows = [row for row in comparison if np.isclose(row["energy_eV"], energy)]
            if not rows:
                continue
            rr = np.asarray([row["r_over_a"] for row in rows])
            rel = np.asarray([row["relative_error"] for row in rows])
            sem = np.asarray([row["jonta_sem"] / row["paper_D_over_D0"] for row in rows])
            label = f"{energy / 1e3:g} keV" if energy < 1e6 else f"{energy / 1e6:g} MeV"
            ax.errorbar(rr, rel, yerr=sem, fmt="o-", capsize=3, color=color, label=label)
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.set_xlabel("r/a")
        ax.set_ylabel("(JONTA - McDevitt) / McDevitt")
        ax.set_title("Pointwise spatial-transport residual")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f"spatial_transport_figure{fig_number}_residual.png", dpi=360)
        plt.close(fig)

    # Appendix-A diffusion diagnostic.  The quantity that must become linear
    # is the radial variance, sigma_r^2=<Delta r^2>-<Delta r>^2.  Show the
    # raw Monte Carlo history, its sub-ensemble standard error, and the exact
    # least-squares line used to infer D=0.5*d sigma_r^2/dt.  One panel per
    # energy keeps the fit quality visible rather than hiding it in a crowded
    # multi-curve plot.
    ir = int(np.argmin(np.abs(result["radii"] - 0.5)))
    n_energy = len(result["energies_ev"])
    ncols = 2
    nrows = int(np.ceil(n_energy / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(9.0, 3.8 * nrows), squeeze=False)
    fit_start = int(result["fit_start_index"])
    tfit = result["time"][fit_start:]
    for ie, energy in enumerate(result["energies_ev"]):
        ax = axes.flat[ie]
        label = f"{energy / 1e3:g} keV" if energy < 1e6 else f"{energy / 1e6:g} MeV"
        variance = result["variance"][:, ie, ir]
        sem = result["variance_sem"][:, ie, ir]
        ax.plot(result["time"], variance, linewidth=1.2, label="JONTA variance")
        if not np.all(np.isnan(sem)):
            ax.fill_between(
                result["time"], variance - sem, variance + sem, alpha=0.18, label="MC SEM"
            )
        fit_line = (
            result["variance_intercept"][ie, ir]
            + result["variance_slope"][ie, ir] * tfit
        )
        ax.plot(tfit, fit_line, "--", linewidth=1.3, label="linear fit")
        ax.axvline(result["time"][fit_start], linestyle=":", linewidth=0.9)
        ax.set_title(
            f"{label}, r/a={result['radii'][ir]:.3f}\n"
            f"R2={result['r2_variance'][ie, ir]:.5f}, "
            f"D/D0={result['D_over_D0'][ie, ir]:.3g}"
        )
        ax.set_xlabel("t / tau_c")
        ax.set_ylabel("<Delta r^2> - <Delta r>^2")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    for ax in axes.flat[n_energy:]:
        ax.set_visible(False)
    fig.suptitle("Diffusive-regime linearity used to infer D", y=1.01)
    fig.tight_layout()
    fig.savefig(outdir / "spatial_transport_variance_linearity.png", dpi=360, bbox_inches="tight")
    plt.close(fig)

    # Every reported Fig.-3 diffusivity point gets its own variance-history
    # panel.  This prevents a good mid-radius trace from masking a non-diffusive
    # fit elsewhere in the scan.
    for ie, energy in enumerate(result["energies_ev"]):
        n_radius = len(result["radii"])
        ncols = 3
        nrows = int(np.ceil(n_radius / ncols))
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(11.4, 3.35 * nrows), squeeze=False, sharex=True
        )
        for ir, radius in enumerate(result["radii"]):
            ax = axes.flat[ir]
            variance = result["variance"][:, ie, ir]
            sem = result["variance_sem"][:, ie, ir]
            ax.plot(result["time"], variance, linewidth=1.1, label="variance")
            if not np.all(np.isnan(sem)):
                ax.fill_between(result["time"], variance - sem, variance + sem, alpha=0.18)
            fit_line = (
                result["variance_intercept"][ie, ir]
                + result["variance_slope"][ie, ir] * tfit
            )
            ax.plot(tfit, fit_line, "--", linewidth=1.2, label="linear fit")
            ax.axvline(result["time"][fit_start], linestyle=":", linewidth=0.8)
            ax.set_title(
                f"r/a={radius:.3f}; R2={result['r2_variance'][ie, ir]:.4f}\n"
                f"D/D0={result['D_over_D0'][ie, ir]:.3g}"
            )
            ax.grid(True, alpha=0.25)
            if ir % ncols == 0:
                ax.set_ylabel("radial variance")
            if ir >= n_radius - ncols:
                ax.set_xlabel("t / tau_c")
        for ax in axes.flat[n_radius:]:
            ax.set_visible(False)
        axes.flat[0].legend(fontsize=8)
        energy_label = (
            f"{energy / 1e3:g} keV" if energy < 1e6 else f"{energy / 1e6:g} MeV"
        )
        fig.suptitle(
            f"Variance linearity for every radius: {energy_label}", y=1.01
        )
        fig.tight_layout()
        tag = (
            f"{energy / 1e3:g}keV" if energy < 1e6 else f"{energy / 1e6:g}MeV"
        ).replace(".", "p")
        fig.savefig(
            outdir / f"spatial_transport_variance_linearity_{tag}.png",
            dpi=360,
            bbox_inches="tight",
        )
        plt.close(fig)

    # Also expose the linear-fit quality over the full radius/energy scan.
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for ie, energy in enumerate(result["energies_ev"]):
        label = f"{energy / 1e3:g} keV" if energy < 1e6 else f"{energy / 1e6:g} MeV"
        ax.plot(result["radii"], result["r2_variance"][ie], "o-", label=label)
    ax.axhline(0.98, linestyle="--", linewidth=1.0, label="R2 = 0.98 guide")
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("r/a")
    ax.set_ylabel("variance linear-fit R2")
    ax.set_title("Linearity of <Delta r^2>-<Delta r>^2")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "spatial_transport_variance_fit_quality.png", dpi=360)
    plt.close(fig)


def _run_cli_case(outdir: Path, kwargs, collision_model: str):
    result = run_transport_scan(**kwargs, collision_model=collision_model)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_csv(outdir / "spatial_transport.csv", result)
    _write_comparison_csv(outdir / "spatial_transport_comparison.csv", result)
    _write_variance_history_csv(outdir / "spatial_transport_variance_history.csv", result)
    _plot_results(outdir, result)

    fig_number, _reference, compare = _reference_for_result(result)
    print("\n" + "=" * 72)
    print(
        f"Figure {fig_number}: {collision_model}; "
        f"c*tau_c/a={result['c_tau_over_a']:.3g}, "
        f"N/group={result['markers_per_group']} "
        f"({result['n_replicates']}x{result['markers_per_replicate']} batches), "
        f"dt={result['dt']:.3g}, t_final={result['final_time']:.3g} tau_c"
    )
    if collision_model == "partial_screening":
        print(
            f"n_e={result['ne_cm3']:.6g} cm^-3, "
            f"n_Ar+/n_D={FIG6_IMPURITY_FRACTION:g}, "
            f"tau_c^a/tau_c={result['tau_main_over_tau_total']:.6g}"
        )
    print("D/D0:")
    print("energy[eV] " + " ".join(f"r={r:.2f}" for r in result["radii"]))
    for ie, energy in enumerate(result["energies_ev"]):
        values = " ".join(f"{x:9.4g}" for x in result["D_over_D0"][ie])
        print(f"{energy:10.4g} {values}")

    print(
        "Variance-fit R2: "
        f"median={np.median(result['r2_variance']):.5f}, "
        f"minimum={np.min(result['r2_variance']):.5f}"
    )

    comparison = compare(result)
    if comparison:
        abs_rel = np.asarray([abs(row["relative_error"]) for row in comparison])
        print(
            f"Digitized McDevitt Fig. {fig_number} comparison: "
            f"median |relative error| = {np.median(abs_rel):.3g}, "
            f"max = {np.max(abs_rel):.3g}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="run the full published Figure-3/Figure-6 parameter set",
    )
    parser.add_argument(
        "--case",
        choices=("both", "fully-ionized", "partial-screening"),
        default="both",
        help="collision model(s) to benchmark",
    )
    parser.add_argument("--markers-per-group", type=int, default=None)
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="independent marker sub-ensembles used for JONTA error bars",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("transport_results"))
    args = parser.parse_args()

    if args.paper:
        kwargs = dict(
            energies_ev=PAPER_ENERGIES_EV,
            radii=PAPER_RADII,
            markers_per_group=args.markers_per_group or 1024,
            n_replicates=args.replicates or 8,
            c_tau_over_a=PAPER_C_TAU_OVER_A,
            final_time=PAPER_FINAL_TIME,
            dt=2.0e-7,
            collision_dt=2.0e-5,
            n_samples=160,
        )
    else:
        kwargs = dict(
            energies_ev=PAPER_ENERGIES_EV,
            radii=CPU_RADII,
            markers_per_group=args.markers_per_group or 32,
            n_replicates=args.replicates or 4,
            c_tau_over_a=CPU_C_TAU_OVER_A,
            final_time=CPU_FINAL_TIME,
            dt=CPU_DT,
            collision_dt=CPU_COLLISION_DT,
            n_samples=80,
        )

    cases = []
    if args.case in ("both", "fully-ionized"):
        cases.append(("fully_ionized", "figure3_fully_ionized"))
    if args.case in ("both", "partial-screening"):
        cases.append(("partial_screening", "figure6_partial_screening"))

    print("JAX devices:", jax.devices())
    for collision_model, dirname in cases:
        outdir = args.output_dir / dirname if len(cases) > 1 else args.output_dir
        _run_cli_case(outdir, kwargs, collision_model)


if __name__ == "__main__":
    main()
