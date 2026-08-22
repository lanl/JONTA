"""Figure-by-figure reproduction suite for McDevitt, Guo & Tang (2019).

The acceptance target is deliberately literal: reproduce Figures
2, 3, 4, 5, 6, 7, 8, 10, 11, 13, 14, B2, B3 and B4 of
PPCF 61, 054008 with JONTA's particle Monte-Carlo implementation.

No continuum kinetic equation is evolved here.  Figures 5/6 reconstruct the
phase-space probability current only as *diagnostic post-processing* of a
particle-sampled distribution.  Toroidal figures use the production circular
RAMc guiding-center RHS.  Appendix figures use the production Moller source or
conservative gain-loss operator.

The exact paper runs are intentionally exposed through ``--paper`` because the
toroidal cases require ~1e8 RK4 orbit substeps per 100 tau_c trajectory at
c*tau_c/a=5e5.  Figure 2 has no reduced/default mode: its radial-localization
claim cannot be tested by a short trajectory. Ordinary pytest only checks the
figure specification and helper relations; paper-scale reproduction is a GPU
validation job.
"""

from __future__ import annotations

import argparse
import csv
import json
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from boundaries.basic import gamma_absorbing_boundary
from collisions.moller import apply_moller_source_only
from collisions.small_angle import relativistic_coulomb_logs, small_angle_step
from core.config import MollerConfig, OrbitNormalization, SmallAngleConfig
from core.constants import C_LIGHT_M_S, E_CHARGE_C, M_E_KG, ME_C2_EV
from core.initialization import particles_from_arrays
from core.state import (
    KinematicState,
    ParticleState,
    QuadraticCircularFieldProfiles,
)
from diagnostics.avalanche import fit_exponential_growth
from diagnostics.runaway_vortex import guo_x_point_momentum
from integrators.explicit import rk4_step
from orbits.ramc_circular import ramc_circular_rhs
from tests.convergence import test_large_angle_avalanche as la
from tests.convergence import test_runaway_vortex as rv

jax.config.update("jax_enable_x64", True)

FIGURE_TARGETS = ("2", "3", "4", "5", "6", "7", "8", "10", "11", "13", "14", "B2", "B3", "B4")

EPSILON = 1.0 / 3.0
A_M = 2.0
Q0 = 2.1
Q2 = 2.0
C_TAU_OVER_A = 5.0e5
GAMMA_MIN = 1.02

# Digitized/visually transcribed centers from the supplied PDF, used only to
# choose efficient MC threshold brackets and to draw paper-reference markers.
# They are not used as simulated answers.
FIG3A_REFERENCE = {
    1.0: (1.63, 1.67, 1.70, 1.73, 1.78),
    5.0: (2.39, 2.52, 2.64, 2.77, 2.92),
    10.0: (3.04, 3.32, 3.54, 3.78, 4.03),
}
FIG3B_REFERENCE = {
    0.05: (1.47, 1.48, 1.49, 1.51, 1.54),
    0.10: FIG3A_REFERENCE[1.0],
    0.30: (2.10, 2.12, 2.16, 2.22, 2.30),
}
FIG7_REFERENCE = (16.5, 17.4, 18.0, 18.7, 19.45)
RADII = (0.0, 0.2, 0.4, 0.6, 0.8)


def _a_omega_ce_over_c(b0_t: float, a_m: float = A_M):
    return a_m * (E_CHARGE_C * b0_t / M_E_KG) / C_LIGHT_M_S


def _uniform_background(vte_over_c: float, zeff: float, coulog0: float):
    return la._background(vte_over_c, zeff, coulog0)


def _toroidal_seed(n_markers: int, seed: int, radius: float, *, width=1.0e-3):
    rng = np.random.default_rng(seed)
    if radius <= 0.0:
        r = rng.uniform(0.0, width, n_markers)
    else:
        r = rng.uniform(max(0.0, radius - width), radius, n_markers)
    theta = rng.uniform(-np.pi, np.pi, n_markers)
    phi = rng.uniform(-np.pi, np.pi, n_markers)
    gamma = rng.uniform(10.0, 20.0, n_markers)
    xi = rng.uniform(-1.0, -0.8, n_markers)
    return particles_from_arrays(
        gamma,
        xi,
        r * np.cos(theta),
        r * np.sin(theta),
        phi,
        np.full(n_markers, 1.0 / n_markers),
        pid=np.arange(n_markers, dtype=np.int64),
    )


@lru_cache(maxsize=64)
def _build_toroidal_kernel(
    *,
    zeff: float,
    coulog0: float,
    vte_over_c: float,
    b0_t: float,
    q0: float,
    q2: float,
    c_tau_over_a: float,
    macro_dt: float,
    total_time: float,
    sample_dt: float,
    orbit_substeps: int,
    small_angle_substeps: int,
    large_angle_every: int,
    partial_screening: bool = False,
    impurity_fraction: float = 0.0,
    impurity_charge_state: float = 1.0,
    impurity_radius_abohr: float = 0.329,
    impurity_mean_excitation_ev: float = 219.4,
    target_electron_factor: float = 1.0,
    relativistic_coulog: bool = False,
):
    """Compile one fixed-shape toroidal avalanche kernel.

    Orbit and small-angle collision substeps are static JAX loops.  The loop
    voltage and synchrotron parameter remain dynamic inputs, so scans do not
    trigger recompilation.
    """

    bg, ne_cm3, _te_ev = _uniform_background(vte_over_c, zeff, coulog0)
    sa_cfg = SmallAngleConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog0,
        pitch_scattering=True,
        friction=True,
        energy_scattering=False,
        max_nu_dt=0.5,
        partial_screening=partial_screening,
        impurity_fraction=impurity_fraction,
        impurity_nuclear_charge=18.0,
        impurity_charge_state=impurity_charge_state,
        impurity_radius_abohr=impurity_radius_abohr,
        impurity_mean_excitation_ev=impurity_mean_excitation_ev,
        relativistic_coulog=relativistic_coulog,
    )
    la_cfg = MollerConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog0,
        gamma_min=GAMMA_MIN,
        max_collision_fraction=0.20,
        bisection_steps=24,
        target_electron_factor=target_electron_factor,
    )
    n_steps = int(round(total_time / macro_dt))
    sample_every = max(1, int(round(sample_dt / macro_dt)))
    n_samples = n_steps // sample_every + 1
    dt_orbit = macro_dt / int(orbit_substeps)
    dt_sa_half = 0.5 * macro_dt / int(small_angle_substeps)
    dt_la = macro_dt * int(large_angle_every)
    aomega = _a_omega_ce_over_c(b0_t)

    @jax.jit
    def run(particles, base_key, e_over_ec, alpha_syn):
        fields = QuadraticCircularFieldProfiles(
            e1=jnp.asarray(e_over_ec),
            q0=jnp.asarray(q0),
            q2=jnp.asarray(q2),
        )
        norm = OrbitNormalization(
            epsilon=EPSILON,
            c_tau_over_a=c_tau_over_a,
            a_omega_ce_over_c=aomega,
            alpha_syn=alpha_syn,
        )
        times = jnp.zeros((n_samples,), dtype=jnp.float64).at[0].set(0.0)
        weights = jnp.zeros((n_samples,), dtype=jnp.float64).at[0].set(jnp.sum(particles.weight))
        mean_r = jnp.zeros((n_samples,), dtype=jnp.float64).at[0].set(
            jnp.sum(particles.weight * jnp.sqrt(particles.kin.x**2 + particles.kin.y**2))
            / jnp.maximum(jnp.sum(particles.weight), 1e-30)
        )
        sample_i = jnp.asarray(1, dtype=jnp.int32)
        max_q = jnp.asarray(0.0, dtype=jnp.float64)

        def macro_body(i, carry):
            state, tt, ww, rr, si, max_q_ = carry

            def sa_first(j, s):
                return small_angle_step(
                    s,
                    bg,
                    dt_sa_half,
                    base_key,
                    (2 * i) * small_angle_substeps + j,
                    sa_cfg,
                )

            state = jax.lax.fori_loop(0, small_angle_substeps, sa_first, state)

            def orbit_body(j, kin):
                t = i * macro_dt + j * dt_orbit
                rhs = lambda y, time: ramc_circular_rhs(y, time, fields, norm)
                return rk4_step(rhs, kin, t, dt_orbit)

            kin = jax.lax.fori_loop(0, orbit_substeps, orbit_body, state.kin)
            state = ParticleState(kin, state.weight, state.alive, state.pid)

            def sa_second(j, s):
                return small_angle_step(
                    s,
                    bg,
                    dt_sa_half,
                    base_key,
                    (2 * i + 1) * small_angle_substeps + j,
                    sa_cfg,
                )

            state = jax.lax.fori_loop(0, small_angle_substeps, sa_second, state)
            state = gamma_absorbing_boundary(state, GAMMA_MIN)

            def large_angle(s):
                return apply_moller_source_only(s, bg, dt_la, base_key, i, la_cfg)

            state, q_here = jax.lax.cond(
                ((i + 1) % large_angle_every) == 0,
                large_angle,
                lambda s: (s, jnp.asarray(0.0, dtype=jnp.float64)),
                state,
            )
            max_q_ = jnp.maximum(max_q_, q_here)

            def record(vals):
                s, tt_, ww_, rr_, si_ = vals
                wsum = jnp.sum(s.weight)
                rad = jnp.sqrt(s.kin.x**2 + s.kin.y**2)
                tt_ = tt_.at[si_].set((i + 1) * macro_dt)
                ww_ = ww_.at[si_].set(wsum)
                rr_ = rr_.at[si_].set(jnp.sum(s.weight * rad) / jnp.maximum(wsum, 1e-30))
                return s, tt_, ww_, rr_, si_ + 1

            state, tt, ww, rr, si = jax.lax.cond(
                ((i + 1) % sample_every) == 0,
                record,
                lambda vals: vals,
                (state, tt, ww, rr, si),
            )
            return state, tt, ww, rr, si, max_q_

        return jax.lax.fori_loop(
            0,
            n_steps,
            macro_body,
            (particles, times, weights, mean_r, sample_i, max_q),
        )

    return run


def run_toroidal_growth(
    e_over_ec: float,
    radius: float,
    *,
    alpha: float,
    zeff: float,
    coulog0: float = 15.0,
    vte_over_c: float = 0.1,
    b0_t: float = 5.3,
    q0: float = Q0,
    q2: float = Q2,
    c_tau_over_a: float = C_TAU_OVER_A,
    n_markers: int = 256,
    total_time: float = 4.0,
    macro_dt: float = 5.0e-3,
    orbit_dt: float = 1.0e-5,
    small_angle_substeps: int = 8,
    large_angle_dt: float = 5.0e-2,
    sample_dt: float = 0.1,
    seeds=(1,),
    fit_start_fraction: float = 0.5,
    partial_screening: bool = False,
    impurity_fraction: float = 0.0,
    impurity_charge_state: int = 1,
    impurity_radius_abohr: float = 0.329,
    impurity_mean_excitation_ev: float = 219.4,
    target_electron_factor: float = 1.0,
    relativistic_coulog: bool = False,
    return_final: bool = False,
):
    orbit_substeps = max(1, int(round(macro_dt / orbit_dt)))
    large_angle_every = max(1, int(round(large_angle_dt / macro_dt)))
    kernel = _build_toroidal_kernel(
        zeff=zeff,
        coulog0=coulog0,
        vte_over_c=vte_over_c,
        b0_t=b0_t,
        q0=q0,
        q2=q2,
        c_tau_over_a=c_tau_over_a,
        macro_dt=macro_dt,
        total_time=total_time,
        sample_dt=sample_dt,
        orbit_substeps=orbit_substeps,
        small_angle_substeps=small_angle_substeps,
        large_angle_every=large_angle_every,
        partial_screening=partial_screening,
        impurity_fraction=impurity_fraction,
        impurity_charge_state=float(impurity_charge_state),
        impurity_radius_abohr=impurity_radius_abohr,
        impurity_mean_excitation_ev=impurity_mean_excitation_ev,
        target_electron_factor=target_electron_factor,
        relativistic_coulog=relativistic_coulog,
    )
    rates = []
    histories = []
    finals = []
    for seed in seeds:
        p0 = _toroidal_seed(n_markers, seed, radius)
        pf, time, weight, mean_r, idx, max_q = kernel(
            p0,
            jax.random.key(91000 + seed),
            jnp.asarray(e_over_ec),
            jnp.asarray(alpha),
        )
        idx = int(idx)
        t = np.asarray(time[:idx])
        w = np.asarray(weight[:idx])
        g, intercept, r2 = fit_exponential_growth(t, w, fit_start_fraction)
        rates.append((g, r2, float(max_q)))
        histories.append((t, w, np.asarray(mean_r[:idx]), g, intercept, r2))
        if return_final:
            finals.append(jax.device_get(pf))
    arr = np.asarray(rates)
    return {
        "growth": float(arr[:, 0].mean()),
        "sem": float(arr[:, 0].std(ddof=1) / np.sqrt(len(seeds))) if len(seeds) > 1 else 0.0,
        "r2": float(arr[:, 1].mean()),
        "max_q": float(arr[:, 2].max()),
        "histories": histories,
        "finals": finals,
    }


def estimate_toroidal_threshold(reference: float, radius: float, **kwargs):
    span = max(0.10, 0.08 * reference)
    samples = []
    for _ in range(4):
        fields = (max(1.001, reference - span), reference + span)
        values = []
        for e in fields:
            res = run_toroidal_growth(e, radius, **kwargs)
            values.append((e, res["growth"], res["sem"]))
        samples.extend(values)
        (e0, g0, _), (e1, g1, _) = values
        if (g0 <= 0 <= g1) or (g1 <= 0 <= g0):
            return float(e0 - g0 * (e1 - e0) / (g1 - g0)), samples
        span *= 1.7
    raise ValueError(f"threshold not bracketed around {reference}: {samples}")


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fig2_ring_diagnostics(initial, final):
    """Measure the physical localization claim made by McDevitt Figure 2.

    Figure 2 is a toroidal-orbit validation, not a generic short-run demo:
    the seeded runaway ring must remain a ring.  Keep this check independent
    of plotting so a failed physics case cannot be mistaken for a successful
    smoke test.
    """

    r_initial = np.hypot(np.asarray(initial.kin.x), np.asarray(initial.kin.y))
    r_final = np.hypot(np.asarray(final.kin.x), np.asarray(final.kin.y))
    initial_mean = float(np.mean(r_initial))
    final_mean = float(np.mean(r_final))
    final_p95 = float(np.percentile(r_final, 95.0))
    relative_mean_shift = abs(final_mean - initial_mean) / max(initial_mean, 1.0e-15)

    # The paper's qualitative claim is radial localization, not a particular
    # Monte-Carlo width.  These conservative bounds reject a collapsed or
    # wall-scale cloud while leaving room for physical broadening.
    accepted = bool(relative_mean_shift <= 0.25 and final_p95 <= 1.25 * initial_mean)
    return {
        "accepted": accepted,
        "initial_mean_radius": initial_mean,
        "final_mean_radius": final_mean,
        "final_radius_p95": final_p95,
        "relative_mean_shift": relative_mean_shift,
        "criteria": {
            "max_relative_mean_shift": 0.25,
            "max_final_radius_p95_over_initial_mean": 1.25,
        },
    }


def reproduce_fig2(outdir: Path, *, paper: bool):
    """Figure 2: narrow runaway ring remains radially localized."""
    if not paper:
        raise ValueError(
            "Figure 2 has no reduced/default mode: its radial-localization "
            "claim cannot be tested by a short smoke run. Re-run explicitly "
            "with --paper after the toroidal orbit validation is resolved."
        )

    import matplotlib.pyplot as plt

    n_markers = 4096
    total_time = 100.0
    res = run_toroidal_growth(
        2.75,
        0.4,
        alpha=0.1,
        zeff=5.0,
        q0=1.9,
        q2=2.0,
        n_markers=n_markers,
        total_time=total_time,
        macro_dt=5e-3,
        # The nominal 1e-5 step is not orbit-converged for this case: a
        # collisionless probe collapses the ring through RK4 phase error.
        orbit_dt=1e-6,
        small_angle_substeps=16,
        large_angle_dt=0.05,
        seeds=(22,),
        return_final=True,
    )
    p0 = _toroidal_seed(n_markers, 22, 0.4)
    pf = res["finals"][0]
    diagnostics = _fig2_ring_diagnostics(p0, pf)
    (outdir / "mcdevitt_fig2_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    if not diagnostics["accepted"]:
        raise RuntimeError(
            "McDevitt Figure 2 failed radial-localization acceptance: "
            f"initial mean r={diagnostics['initial_mean_radius']:.6g}, "
            f"final mean r={diagnostics['final_mean_radius']:.6g}. "
            "No figure was written."
        )

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.1), sharex=True, sharey=True)
    for ax, state, title in zip(axes, (p0, pf), ("(a) Initial", "(b) Final")):
        ax.hist2d(np.asarray(state.kin.x), np.asarray(state.kin.y), bins=120, range=[[-1, 1], [-1, 1]])
        ax.set(xlabel="x/a", ylabel="y/a", title=title, xlim=(-1, 1), ylim=(-1, 1))
    fig.tight_layout()
    fig.savefig(outdir / "mcdevitt_fig2.png", dpi=320)
    plt.close(fig)
    return res


def reproduce_fig3(outdir: Path, *, paper: bool):
    """Figure 3: avalanche threshold versus radius, Zeff and alpha."""
    import matplotlib.pyplot as plt

    n_markers = 4096 if paper else 96
    total_time = 100.0 if paper else 1.0
    seeds = tuple(range(6)) if paper else (31, 32)
    rows = []
    for zeff, refs in FIG3A_REFERENCE.items():
        for radius, ref in zip(RADII, refs):
            th, th_samples = estimate_toroidal_threshold(
                ref,
                radius,
                alpha=0.1,
                zeff=zeff,
                n_markers=n_markers,
                total_time=total_time,
                macro_dt=5e-3,
                orbit_dt=1e-5,
                small_angle_substeps=16 if zeff >= 5 else 8,
                large_angle_dt=0.05,
                seeds=seeds,
            )
            rows.append({"panel": "a", "radius": radius, "zeff": zeff, "alpha": 0.1, "JONTA_Eav": th, "JONTA_Eav_SEM": la.threshold_interpolation_sem(th_samples), "paper_Eav": ref})
    for alpha, refs in FIG3B_REFERENCE.items():
        for radius, ref in zip(RADII, refs):
            th, th_samples = estimate_toroidal_threshold(
                ref,
                radius,
                alpha=alpha,
                zeff=1.0,
                n_markers=n_markers,
                total_time=total_time,
                macro_dt=5e-3,
                orbit_dt=1e-5,
                small_angle_substeps=8,
                large_angle_dt=0.05,
                seeds=seeds,
            )
            rows.append({"panel": "b", "radius": radius, "zeff": 1.0, "alpha": alpha, "JONTA_Eav": th, "JONTA_Eav_SEM": la.threshold_interpolation_sem(th_samples), "paper_Eav": ref})
    _write_csv(outdir / "mcdevitt_fig3.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.0))
    for zeff in (1.0, 5.0, 10.0):
        rr = [r for r in rows if r["panel"] == "a" and r["zeff"] == zeff]
        axes[0].plot(RADII, FIG3A_REFERENCE[zeff], "--")
        axes[0].errorbar([r["radius"] for r in rr], [r["JONTA_Eav"] for r in rr], yerr=[r["JONTA_Eav_SEM"] for r in rr], fmt="o-", fillstyle="none", capsize=3, label=fr"$Z_{{eff}}={int(zeff)}$")
    for alpha in (0.05, 0.1, 0.3):
        rr = [r for r in rows if r["panel"] == "b" and np.isclose(r["alpha"], alpha)]
        axes[1].plot(RADII, FIG3B_REFERENCE[alpha], "--")
        axes[1].errorbar([r["radius"] for r in rr], [r["JONTA_Eav"] for r in rr], yerr=[r["JONTA_Eav_SEM"] for r in rr], fmt="o-", fillstyle="none", capsize=3, label=fr"$\alpha={alpha:g}$")
    for ax in axes:
        ax.set(xlabel="r/a", ylabel=r"$E_{av}/E_c$")
        ax.legend(fontsize=8)
    axes[0].set_title("(a) Zeff scan")
    axes[1].set_title("(b) synchrotron scan")
    fig.tight_layout(); fig.savefig(outdir / "mcdevitt_fig3.png", dpi=320); plt.close(fig)
    return rows


def reproduce_fig4(outdir: Path, *, paper: bool):
    """Figure 4: fully ionized toroidal avalanche growth-rate scan."""
    import matplotlib.pyplot as plt

    fields = (2.0, 4.0, 6.0, 8.0, 10.0)
    n_markers = 4096 if paper else 128
    total_time = 40.0 if paper else 1.0
    seeds = tuple(range(4)) if paper else (44, 45)
    rows = []
    for radius in RADII:
        for e in fields:
            res = run_toroidal_growth(
                e,
                radius,
                alpha=0.1,
                zeff=1.0,
                n_markers=n_markers,
                total_time=total_time,
                macro_dt=5e-3,
                orbit_dt=1e-5,
                small_angle_substeps=8,
                large_angle_dt=0.05,
                seeds=seeds,
            )
            rows.append({"radius": radius, "E_over_Ec": e, "growth": res["growth"], "SEM": res["sem"]})
    _write_csv(outdir / "mcdevitt_fig4.csv", rows)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    for radius in RADII:
        rr = [r for r in rows if np.isclose(r["radius"], radius)]
        ax.errorbar([r["E_over_Ec"] for r in rr], [r["growth"] for r in rr], yerr=[r["SEM"] for r in rr], marker="o", label=fr"$r/a={radius:g}$")
    ax.set(xlabel=r"$E/E_c$", ylabel=r"$\gamma_{av}\tau_c$", title="McDevitt Fig. 4")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(outdir / "mcdevitt_fig4.png", dpi=320); plt.close(fig)
    return rows


def _reproduce_phase_flow(outdir: Path, *, paper: bool, fig: str):
    import matplotlib.pyplot as plt

    if fig == "5":
        cases = [(1.0, 1.676), (1.0, 2.0)]
    elif fig == "6":
        cases = [(5.0, 2.39), (10.0, 3.057)]
    else:
        raise ValueError(fig)
    n_markers = 16384 if paper else 1536
    total_time = 80.0 if paper else 24.0
    burn_time = 40.0 if paper else 12.0
    figobj, axes = plt.subplots(1, 2, figsize=(10.0, 4.1), sharey=True)
    outputs = []
    for i, ((zeff, e), ax) in enumerate(zip(cases, axes)):
        kernel = rv._make_vortex_kernel(
            n_bins=240,
            p_min=0.3,
            p_max=20.0,
            phase_bins=(90, 80),
            vte_over_c=0.1,
            zeff=zeff,
            alpha=0.1,
            energy_scattering=True,
        )
        result = rv.run_particle_vortex_case(
            kernel,
            e,
            n_markers=n_markers,
            seed=6100 + i + 10 * int(zeff),
            dt=0.004,
            total_time=total_time,
            burn_time=burn_time,
            sample_every=8,
            phase_sample_every=16,
            p_min=0.3,
            p_max=20.0,
            p_init_max=17.0,
            vte_over_c=0.1,
            zeff=zeff,
            alpha=0.1,
        )
        p, xi, f, gp, gx = rv.reconstruct_phase_space_flux(result)
        speed = np.sqrt(gp * gp + gx * gx)
        gp_n = gp / np.maximum(speed, np.nanpercentile(speed, 60) * 0.08 + 1e-300)
        gx_n = gx / np.maximum(speed, np.nanpercentile(speed, 60) * 0.08 + 1e-300)
        ax.streamplot(p, xi, gp_n.T, gx_n.T, density=1.6, linewidth=0.7, arrowsize=0.7)
        trap = -np.sqrt((2.0 * 0.8 * EPSILON) / (1.0 + 0.8 * EPSILON))
        ax.axhline(trap, linestyle="--", linewidth=1.0)
        ax.set(xlim=(0, 20), ylim=(-1, 0), xlabel=r"$p/(m_ec)$", title=fr"$Z_{{eff}}={int(zeff)}, E/E_c={e:g}$")
        outputs.append(result)
    axes[0].set_ylabel(r"$\xi$")
    figobj.tight_layout(); figobj.savefig(outdir / f"mcdevitt_fig{fig}.png", dpi=320); plt.close(figobj)
    return outputs


def reproduce_fig7(outdir: Path, *, paper: bool):
    """Figure 7: toroidal threshold with singly ionized argon."""
    import matplotlib.pyplot as plt

    f_imp = 1.0
    zi = 1
    zeff, target = la.argon_mixture_parameters(f_imp, zi)
    ai, ii = la.argon_hesslow_parameters(zi)
    vte = 0.0063
    rows = []
    for radius, ref in zip(RADII, FIG7_REFERENCE):
        th, th_samples = estimate_toroidal_threshold(
            ref,
            radius,
            alpha=0.05,
            zeff=zeff,
            coulog0=10.0,
            vte_over_c=vte,
            b0_t=53.0,
            n_markers=4096 if paper else 96,
            total_time=100.0 if paper else 1.0,
            macro_dt=2e-3,
            orbit_dt=1e-5,
            small_angle_substeps=40,
            large_angle_dt=0.01,
            seeds=tuple(range(6)) if paper else (71, 72),
            partial_screening=True,
            impurity_fraction=f_imp,
            impurity_charge_state=zi,
            impurity_radius_abohr=ai,
            impurity_mean_excitation_ev=ii,
            target_electron_factor=target,
        )
        rows.append({"radius": radius, "JONTA_Eav": th, "JONTA_Eav_SEM": la.threshold_interpolation_sem(th_samples), "paper_Eav": ref})
    _write_csv(outdir / "mcdevitt_fig7.csv", rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.2)); ax.plot(RADII, FIG7_REFERENCE, "--", label="paper"); ax.errorbar(RADII, [r["JONTA_Eav"] for r in rows], yerr=[r["JONTA_Eav_SEM"] for r in rows], fmt="o-", fillstyle="none", capsize=3, label="JONTA"); ax.set(xlabel="r/a", ylabel=r"$E_{av}/E_c$", title="McDevitt Fig. 7"); ax.legend(); fig.tight_layout(); fig.savefig(outdir / "mcdevitt_fig7.png", dpi=320); plt.close(fig)
    return rows


def reproduce_fig8(outdir: Path, *, paper: bool):
    """Figure 8: radial avalanche-efficiency measure psi_10."""
    import matplotlib.pyplot as plt

    rows = []
    for zeff in (1.0, 5.0):
        for radius in RADII:
            growth = []
            for e in (10.0, 20.0, 30.0):
                res = run_toroidal_growth(
                    e,
                    radius,
                    alpha=0.1,
                    zeff=zeff,
                    n_markers=4096 if paper else 96,
                    total_time=30.0 if paper else 0.8,
                    macro_dt=5e-3,
                    orbit_dt=1e-5,
                    small_angle_substeps=16,
                    large_angle_dt=0.05,
                    seeds=tuple(range(4)) if paper else (81, 82),
                )
                growth.append((e, res["growth"], res["sem"]))
            x = np.asarray([e - 1.0 for e, _, _ in growth])
            y = np.asarray([g for _, g, _ in growth])
            ysem = np.asarray([sem for _, _, sem in growth])
            denom = float(np.dot(x, x))
            gamma0 = float(np.dot(x, y) / denom)
            gamma0_sem = float(np.sqrt(np.sum((x * ysem) ** 2)) / denom)
            psi10 = la.psi10_current_ma(gamma0)
            psi10_sem = abs(psi10 * gamma0_sem / gamma0) if gamma0 != 0 else np.nan
            rows.append({"zeff": zeff, "radius": radius, "gamma0_tau_c": gamma0, "gamma0_SEM": gamma0_sem, "psi10_MA": psi10, "psi10_SEM_MA": psi10_sem})
    _write_csv(outdir / "mcdevitt_fig8.csv", rows)
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    for zeff in (1.0, 5.0):
        rr = [r for r in rows if r["zeff"] == zeff]
        ax.errorbar([r["radius"] for r in rr], [r["psi10_MA"] for r in rr], yerr=[r["psi10_SEM_MA"] for r in rr], fmt="o-", capsize=3, label=fr"JONTA $Z_{{eff}}={int(zeff)}$")
        analytic = [la.psi10_current_ma(la.rosenbluth_putvinski_gamma0_tau_c(r, EPSILON, zeff, 15.0)) for r in RADII]
        ax.plot(RADII, analytic, "--", label=fr"RP $Z_{{eff}}={int(zeff)}$")
    ax.set(xlabel="r/a", ylabel=r"$2\pi\psi_{10}/(\mu_0R_0)$ [MA]", title="McDevitt Fig. 8"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(outdir / "mcdevitt_fig8.png", dpi=320); plt.close(fig)
    return rows


def reproduce_fig10(outdir: Path, *, paper: bool):
    """Figure 10: energy-dependent Coulomb-log avalanche efficiency."""
    import matplotlib.pyplot as plt

    fields = (30.0, 45.0, 60.0, 80.0, 100.0, 120.0)
    rows = []
    vte = np.sqrt(20.0 / ME_C2_EV)
    for zeff in (1.0, 5.0):
        constant_gamma0 = []
        for e in fields:
            for model in ("constant", "energy_dependent"):
                res = la.run_growth_replicates(
                    e,
                    alpha=0.1,
                    zeff=zeff,
                    coulog0=10.0,
                    n_markers=4096 if paper else 512,
                    total_time=15.0 if paper else 3.0,
                    dt=0.0025,
                    large_angle_dt=0.05,
                    seeds=tuple(range(5)) if paper else (91, 113),
                    fit_start_fraction=0.5,
                    vte_over_c=vte,
                    relativistic_coulog=(model == "energy_dependent"),
                )
                gamma0 = res["growth"] / (e - 1.0)
                psi = la.psi10_current_ma(gamma0)
                gamma0_sem = res["sem"] / (e - 1.0)
                psi_sem = abs(psi * gamma0_sem / gamma0) if gamma0 != 0 else np.nan
                rows.append({"zeff": zeff, "model": model, "E_over_Ec": e, "gamma0_tau_c": gamma0, "gamma0_SEM": gamma0_sem, "psi10_MA": psi, "psi10_SEM_MA": psi_sem})
                if model == "constant":
                    constant_gamma0.append(gamma0)
        gconst = float(np.mean(constant_gamma0))
        for e in fields:
            px = float(guo_x_point_momentum(e, zeff))
            gx = np.sqrt(1.0 + px * px)
            bg, _ne, te = la._background(vte, zeff, 10.0)
            del bg
            lee, _lei = relativistic_coulomb_logs(jnp.asarray(gx), jnp.asarray(te), jnp.asarray(10.0))
            gamma_model = gconst * 10.0 / float(lee)
            rows.append({"zeff": zeff, "model": "Eq16", "E_over_Ec": e, "gamma0_tau_c": gamma_model, "gamma0_SEM": 0.0, "psi10_MA": la.psi10_current_ma(gamma_model), "psi10_SEM_MA": 0.0})
    _write_csv(outdir / "mcdevitt_fig10.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0), sharex=True)
    for ax, zeff in zip(axes, (1.0, 5.0)):
        for model, ls in (("constant", "-"), ("energy_dependent", "o-"), ("Eq16", "--")):
            rr = [r for r in rows if r["zeff"] == zeff and r["model"] == model]
            if model == "Eq16":
                ax.plot([r["E_over_Ec"] - 1.0 for r in rr], [r["psi10_MA"] for r in rr], ls, label=model)
            else:
                ax.errorbar([r["E_over_Ec"] - 1.0 for r in rr], [r["psi10_MA"] for r in rr], yerr=[r["psi10_SEM_MA"] for r in rr], fmt=ls, capsize=3, label=model)
        ax.set_xscale("log"); ax.set(xlabel=r"$E/E_c-1$", ylabel=r"$2\pi\psi_{10}/(\mu_0R_0)$ [MA]", title=fr"$Z_{{eff}}={int(zeff)}$"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(outdir / "mcdevitt_fig10.png", dpi=320); plt.close(fig)
    return rows


def reproduce_fig11(outdir: Path, *, paper: bool):
    """Figure 11: partially screened toroidal growth and inferred psi_10."""
    import matplotlib.pyplot as plt

    f_imp = 1.0; zi = 1
    zeff, target = la.argon_mixture_parameters(f_imp, zi)
    ai, ii = la.argon_hesslow_parameters(zi)
    fields = (20.0, 25.0, 30.0, 40.0, 50.0, 60.0, 70.0)
    rows = []
    for radius in RADII:
        for e in fields:
            res = run_toroidal_growth(
                e,
                radius,
                alpha=0.05,
                zeff=zeff,
                coulog0=10.0,
                vte_over_c=0.0028,
                b0_t=53.0,
                n_markers=4096 if paper else 96,
                total_time=20.0 if paper else 0.8,
                macro_dt=1e-3,
                orbit_dt=1e-5,
                small_angle_substeps=80,
                large_angle_dt=0.005,
                seeds=tuple(range(4)) if paper else (121, 122),
                partial_screening=True,
                impurity_fraction=f_imp,
                impurity_charge_state=zi,
                impurity_radius_abohr=ai,
                impurity_mean_excitation_ev=ii,
                target_electron_factor=target,
            )
            gamma0 = res["growth"] / (e - 1.0)
            psi10 = la.psi10_current_ma(gamma0) if gamma0 > 0 else np.nan
            gamma0_sem = res["sem"] / (e - 1.0)
            psi10_sem = abs(psi10 * gamma0_sem / gamma0) if gamma0 > 0 else np.nan
            rows.append({"radius": radius, "E_over_Ec": e, "growth": res["growth"], "SEM": res["sem"], "psi10_MA": psi10, "psi10_SEM_MA": psi10_sem})
    _write_csv(outdir / "mcdevitt_fig11.csv", rows)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    for radius in RADII:
        rr = [r for r in rows if np.isclose(r["radius"], radius)]
        axes[0].errorbar([r["E_over_Ec"] for r in rr], [r["growth"] for r in rr], yerr=[r["SEM"] for r in rr], fmt="o-", capsize=3, label=fr"$r/a={radius:g}$")
        axes[1].errorbar([r["E_over_Ec"] - 1.0 for r in rr], [r["psi10_MA"] for r in rr], yerr=[r["psi10_SEM_MA"] for r in rr], fmt="o-", capsize=3)
    axes[0].set(xlabel=r"$E/E_c$", ylabel=r"$\gamma_{av}\tau_c$", title="(a) Avalanche growth rate"); axes[0].legend(fontsize=7)
    axes[1].set(xlabel=r"$E/E_c-1$", ylabel=r"$2\pi\psi_{10}/(\mu_0R_0)$ [MA]", title=r"(b) Inferred $\psi_{10}$")
    fig.tight_layout(); fig.savefig(outdir / "mcdevitt_fig11.png", dpi=320); plt.close(fig)
    return rows


def reproduce_b2(outdir: Path, *, paper: bool):
    """Appendix Figure B2: exponential amplification and energy spectrum."""
    import matplotlib.pyplot as plt

    res = la.run_growth_replicates(
        70.0,
        alpha=0.1,
        zeff=1.0,
        coulog0=10.0,
        n_markers=8192 if paper else 4096,
        total_time=5.0,
        dt=0.0025,
        large_angle_dt=0.05,
        sample_dt=0.025,
        seeds=(54,),
        fit_start_fraction=0.15,
        return_final=True,
    )
    _seed, t, w, growth, intercept, r2 = res["histories"][0]
    final = res["finals"][0]
    g = np.asarray(final.kin.gamma)
    wt = np.asarray(final.weight)
    bins = np.linspace(1.0, 191.0, 96)
    hist_raw, edges = np.histogram(g, bins=bins, weights=wt, density=False)
    hist_w2, _ = np.histogram(g, bins=bins, weights=wt * wt, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:]) - 1.0
    total_w = max(np.sum(hist_raw), 1e-300)
    hist = hist_raw / total_w
    hist_sem = np.sqrt(hist_w2) / total_w
    kernel_smooth = np.asarray([0.25, 0.5, 0.25])
    hist = np.convolve(hist, kernel_smooth, mode="same")
    hist_sem = np.sqrt(np.convolve(hist_sem * hist_sem, kernel_smooth * kernel_smooth, mode="same"))
    # Rosenbluth-Putvinski steady avalanche spectrum: T_av/(m_ec^2) = 1/(tau_c*gamma_0).
    gamma0_tau = la.rosenbluth_putvinski_gamma0_tau_c(0.0, EPSILON, 1.0, 10.0)
    theta_av = 1.0 / gamma0_tau
    theory = np.exp(-centers / theta_av)
    mask = (centers > 10) & (centers < 80) & (hist > 0)
    if np.any(mask):
        theory *= np.exp(np.mean(np.log(hist[mask]) - np.log(theory[mask])))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.0))
    axes[0].semilogy(t, w / w[0], label="JONTA")
    axes[0].semilogy(t, np.exp(intercept + growth * t) / w[0], "--", label="fit")
    axes[0].set(xlabel=r"$t/\tau_c$", ylabel=r"$N_{RA}(t)/N_{RA}(0)$", title="(a) Amplification")
    axes[0].legend()
    positive = hist > 0.0
    axes[1].errorbar(centers[positive], hist[positive], yerr=hist_sem[positive], fmt="o", markersize=2.5, linewidth=0.8, capsize=2, label="JONTA")
    axes[1].semilogy(centers, theory, "--", label="theory")
    axes[1].set(xlabel=r"$\gamma-1$", ylabel=r"$N_{RA}(\gamma)$", title="(b) Energy spectrum", xlim=(0, 190))
    axes[1].legend(); fig.tight_layout(); fig.savefig(outdir / "mcdevitt_figB2.png", dpi=320); plt.close(fig)
    rows = [{"growth": growth, "R2": r2, "theta_av": theta_av, "final_weight": float(np.sum(wt))}]
    _write_csv(outdir / "mcdevitt_figB2.csv", rows)
    return rows


def run_figure(figure: str, outdir: Path, *, paper: bool):
    if figure == "2" and not paper:
        raise ValueError(
            "Figure 2 has no reduced/default mode: its radial-localization "
            "claim cannot be tested by a short smoke run. Re-run explicitly "
            "with --paper after the toroidal orbit validation is resolved."
        )
    outdir.mkdir(parents=True, exist_ok=True)
    if figure == "2": return reproduce_fig2(outdir, paper=paper)
    if figure == "3": return reproduce_fig3(outdir, paper=paper)
    if figure == "4": return reproduce_fig4(outdir, paper=paper)
    if figure in ("5", "6"): return _reproduce_phase_flow(outdir, paper=paper, fig=figure)
    if figure == "7": return reproduce_fig7(outdir, paper=paper)
    if figure == "8": return reproduce_fig8(outdir, paper=paper)
    if figure == "10": return reproduce_fig10(outdir, paper=paper)
    if figure == "11": return reproduce_fig11(outdir, paper=paper)
    if figure == "13": return la.run_driver(outdir, paper=paper, case="cutoff")
    if figure == "14": return la.run_driver(outdir, paper=paper, case="compare")
    if figure == "B2": return reproduce_b2(outdir, paper=paper)
    if figure == "B3":
        la.run_driver(outdir, paper=paper, case="b3")
        return la.run_driver(outdir, paper=paper, case="threshold")
    if figure == "B4": return la.run_driver(outdir, paper=paper, case="b4")
    raise ValueError(f"unknown figure {figure}")


def test_requested_mcdevitt_figure_scope_is_exact():
    assert FIGURE_TARGETS == la.MCDEVITT_FIGURE_TARGETS
    assert set(FIGURE_TARGETS) == {"2", "3", "4", "5", "6", "7", "8", "10", "11", "13", "14", "B2", "B3", "B4"}


def test_trapped_passing_boundary_matches_figures_5_and_6_caption_geometry():
    xi_trap = np.sqrt((2.0 * 0.8 * EPSILON) / (1.0 + 0.8 * EPSILON))
    assert np.isclose(xi_trap, 0.6488856845, rtol=1e-9)


def test_fig2_reduced_mode_is_not_a_valid_default(tmp_path):
    with pytest.raises(ValueError, match="no reduced/default mode"):
        reproduce_fig2(tmp_path, paper=False)
    assert not list(tmp_path.iterdir())


def test_fig2_ring_acceptance_rejects_radial_collapse():
    initial = _toroidal_seed(32, 22, 0.4)
    collapsed = ParticleState(
        KinematicState(
            initial.kin.gamma,
            initial.kin.xi,
            jnp.zeros_like(initial.kin.x),
            jnp.zeros_like(initial.kin.y),
            initial.kin.phi,
        ),
        initial.weight,
        initial.alive,
        initial.pid,
    )
    diagnostics = _fig2_ring_diagnostics(initial, collapsed)
    assert not diagnostics["accepted"]
    assert diagnostics["final_mean_radius"] == 0.0


@pytest.mark.slow
def test_b2_high_field_avalanche_is_exponential():
    res = la.run_growth_replicates(
        70.0,
        alpha=0.1,
        zeff=1.0,
        coulog0=10.0,
        n_markers=512,
        total_time=3.0,
        dt=0.005,
        large_angle_dt=0.05,
        sample_dt=0.05,
        seeds=(54,),
        fit_start_fraction=0.15,
    )
    assert res["growth"] > 0.5
    assert res["r2_mean"] > 0.98
    assert res["max_q"] < 0.20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("mcdevitt_2019_figures"))
    parser.add_argument("--figure", choices=FIGURE_TARGETS, required=True, help="run exactly one paper figure per process")
    parser.add_argument("--paper", action="store_true", help="paper-scale statistics and integration times; GPU recommended")
    args = parser.parse_args()
    print(f"[McDevitt 2019] reproducing Figure {args.figure}", flush=True)
    run_figure(args.figure, args.output_dir, paper=args.paper)


if __name__ == "__main__":
    main()
