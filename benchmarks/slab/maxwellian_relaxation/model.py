"""Maxwellian-relaxation benchmark model for the small-angle collision operator.

A test-particle collision operator linearized against a fixed Maxwellian
background must leave that Maxwellian invariant and drive arbitrary marker
distributions toward it. This driver scans both background temperature and
deliberately different initial distributions.

Physics enabled
---------------
* small-angle pitch scattering;
* collisional friction;
* energy diffusion / Ito drift.

Physics disabled
----------------
* deterministic orbit push and electric-field acceleration;
* synchrotron radiation;
* large-angle collisions;
* sources and plasma feedback.

The hot loop is a single JIT-compiled ``jax.lax.fori_loop`` over collision
steps.  There are no Python particle or timestep loops.  The same kernel is
used on CPU and GPU; only the outer validation parameter scan is Python.

The collision timestep and relaxation interval are scaled to the thermal
collision timescale, which varies as (v_Te/c)^3 in the low-energy limit.  This
keeps the numerical resolution comparable across temperatures without using
adaptive stepping.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from collisions.coulomb import thermal_coulomb_log, thermal_speed_over_c
from collisions.small_angle import small_angle_step
from core.config import SmallAngleConfig
from core.constants import ME_C2_EV
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles, ParticleState

jax.config.update("jax_enable_x64", True)

DEFAULT_TEMPERATURES_EV = (1.0e2, 1.0e3, 1.0e4)
INITIAL_DISTRIBUTIONS = ("uniform", "beam", "bimodal")
DEFAULT_NE_CM3 = 1.0e14
DEFAULT_DT_THERMAL = 5.0e-3
DEFAULT_RELAXATION_THERMAL = 30.0


def _homogeneous_background(te_ev: float, ne_cm3: float = DEFAULT_NE_CM3):
    r = jnp.linspace(0.0, 1.0, 8)
    return BackgroundProfiles(
        r=r,
        ne_cm3=jnp.full_like(r, ne_cm3),
        te_ev=jnp.full_like(r, te_ev),
        ti_ev=jnp.full_like(r, te_ev),
        zeff=jnp.ones_like(r),
        eta_bar=jnp.ones_like(r),
    )


def _thermal_time_scale(te_ev: float) -> float:
    """Return the low-energy thermal collision scale in units of tau_c."""

    vte = float(thermal_speed_over_c(te_ev))
    return vte**3


def _collision_config(te_ev: float, ne_cm3: float = DEFAULT_NE_CM3):
    coulog = float(thermal_coulomb_log(ne_cm3, te_ev))
    return SmallAngleConfig(
        ne0_cm3=ne_cm3,
        coulog0=coulog,
        pitch_scattering=True,
        friction=True,
        energy_scattering=True,
        max_nu_dt=0.5,
    )


def _sample_initial_distribution(kind: str, te_ev: float, n: int, seed: int):
    """Return kinetic energy [eV] and pitch for a non-equilibrium ensemble."""

    rng = np.random.default_rng(seed)

    if kind == "uniform":
        # Very broad, already isotropic distribution.  This isolates energy
        # relaxation from pitch isotropization.
        kinetic_ev = rng.uniform(0.0, 8.0 * te_ev, n)
        xi = rng.uniform(-1.0, 1.0, n)
    elif kind == "beam":
        # Narrow hot beam: strongly wrong in both energy and pitch.
        kinetic_ev = np.clip(
            rng.normal(5.0 * te_ev, 0.20 * te_ev, n),
            1.0e-3 * te_ev,
            None,
        )
        xi = np.clip(rng.normal(0.90, 0.03, n), -1.0, 1.0)
    elif kind == "bimodal":
        # Cold/hot counter-directed populations.
        cold = rng.random(n) < 0.5
        kinetic_ev = np.where(
            cold,
            np.clip(
                rng.normal(0.25 * te_ev, 0.05 * te_ev, n),
                1.0e-3 * te_ev,
                None,
            ),
            np.clip(
                rng.normal(6.0 * te_ev, 0.30 * te_ev, n),
                1.0e-3 * te_ev,
                None,
            ),
        )
        xi = np.where(
            cold,
            np.clip(rng.normal(-0.70, 0.05, n), -1.0, 1.0),
            np.clip(rng.normal(+0.70, 0.05, n), -1.0, 1.0),
        )
    else:
        raise ValueError(f"unknown initial distribution: {kind}")

    return kinetic_ev, xi


def make_initial_ensemble(te_ev: float, n_per_distribution: int, seed: int = 41):
    """Concatenate all initial distributions into one GPU-friendly ensemble."""

    kinetic_blocks = []
    xi_blocks = []
    for offset, kind in enumerate(INITIAL_DISTRIBUTIONS):
        kinetic_ev, xi = _sample_initial_distribution(
            kind,
            te_ev,
            n_per_distribution,
            seed + 1009 * offset,
        )
        kinetic_blocks.append(kinetic_ev)
        xi_blocks.append(xi)

    kinetic_ev = np.concatenate(kinetic_blocks)
    xi = np.concatenate(xi_blocks)
    n_total = kinetic_ev.size
    gamma = 1.0 + kinetic_ev / ME_C2_EV

    particles = particles_from_arrays(
        gamma=gamma,
        xi=xi,
        x=np.zeros(n_total),
        y=np.zeros(n_total),
        phi=np.zeros(n_total),
        weight=np.ones(n_total),
        pid=np.arange(n_total, dtype=np.int64),
    )
    return particles, kinetic_ev, xi


def _make_relaxation_kernel(base_key):
    """Build one JIT collision loop reusable across temperature scans."""

    @jax.jit
    def kernel(
        particles: ParticleState,
        background: BackgroundProfiles,
        dt,
        n_steps,
        config: SmallAngleConfig,
    ):
        def body(i, state):
            return small_angle_step(
                state,
                background,
                dt,
                base_key,
                i,
                config,
            )

        return jax.lax.fori_loop(0, n_steps, body, particles)

    return kernel


def maxwellian_energy_pdf(kinetic_ev, te_ev: float):
    """Unnormalized relativistic energy density for f proportional exp(-K/T).

    Marker probability in an isotropic momentum-space Maxwellian contains the
    p^2 dp phase-space Jacobian.  Transforming to kinetic energy K gives a
    shape proportional to p*gamma*exp(-K/T), with constants omitted.
    """

    kinetic_ev = np.asarray(kinetic_ev, dtype=float)
    gamma = 1.0 + kinetic_ev / ME_C2_EV
    p = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
    return p * gamma * np.exp(-kinetic_ev / te_ev)


def _reference_cdf(te_ev: float, max_energy_ev: float):
    upper = max(20.0 * te_ev, 1.05 * max_energy_ev)
    grid = np.linspace(0.0, upper, 50_001)
    pdf = maxwellian_energy_pdf(grid, te_ev)
    cdf = np.concatenate(
        [[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(grid))]
    )
    cdf /= cdf[-1]
    return grid, cdf, pdf


def _energy_ks_distance(kinetic_ev, te_ev: float):
    kinetic_ev = np.asarray(kinetic_ev, dtype=float)
    grid, cdf, _ = _reference_cdf(te_ev, float(np.max(kinetic_ev)))
    samples = np.sort(kinetic_ev)
    empirical = np.arange(1, samples.size + 1, dtype=float) / samples.size
    target = np.interp(samples, grid, cdf)
    return float(np.max(np.abs(empirical - target)))


def _reference_mean_energy(te_ev: float):
    grid = np.linspace(0.0, 20.0 * te_ev, 50_001)
    pdf = maxwellian_energy_pdf(grid, te_ev)
    norm = np.trapezoid(pdf, grid)
    return float(np.trapezoid(grid * pdf, grid) / norm)


def _distribution_metrics(kinetic_ev, xi, te_ev: float):
    kinetic_ev = np.asarray(kinetic_ev, dtype=float)
    xi = np.asarray(xi, dtype=float)
    target_mean = _reference_mean_energy(te_ev)
    return {
        "ks_energy": _energy_ks_distance(kinetic_ev, te_ev),
        "mean_energy_over_target": float(np.mean(kinetic_ev) / target_mean),
        "mean_xi": float(np.mean(xi)),
        "mean_xi2": float(np.mean(xi * xi)),
    }


def run_temperature_case(
    kernel,
    te_ev: float,
    *,
    n_per_distribution: int = 4096,
    seed: int = 41,
    dt_thermal: float = DEFAULT_DT_THERMAL,
    relaxation_thermal: float = DEFAULT_RELAXATION_THERMAL,
):
    """Relax all initial distributions for one background temperature."""

    particles, kinetic_initial, xi_initial = make_initial_ensemble(
        te_ev,
        n_per_distribution,
        seed=seed,
    )
    background = _homogeneous_background(te_ev)
    config = _collision_config(te_ev)

    tau_th = _thermal_time_scale(te_ev)
    dt = dt_thermal * tau_th
    final_time = relaxation_thermal * tau_th
    n_steps = max(1, int(round(final_time / dt)))
    dt = final_time / n_steps

    started = time.perf_counter()
    final = kernel(
        particles,
        background,
        jnp.asarray(dt, dtype=jnp.float64),
        jnp.asarray(n_steps, dtype=jnp.int32),
        config,
    )
    jax.block_until_ready(final)
    elapsed = time.perf_counter() - started
    # Synchronize once per temperature so timings/results are honest on GPU.
    final_gamma = np.asarray(jax.device_get(final.kin.gamma))
    final_xi = np.asarray(jax.device_get(final.kin.xi))
    kinetic_final = (final_gamma - 1.0) * ME_C2_EV

    rows = []
    for index, kind in enumerate(INITIAL_DISTRIBUTIONS):
        sl = slice(index * n_per_distribution, (index + 1) * n_per_distribution)
        initial_metrics = _distribution_metrics(
            kinetic_initial[sl],
            xi_initial[sl],
            te_ev,
        )
        final_metrics = _distribution_metrics(
            kinetic_final[sl],
            final_xi[sl],
            te_ev,
        )
        rows.append(
            {
                "temperature_ev": float(te_ev),
                "initial_distribution": kind,
                "n_markers": int(n_per_distribution),
                "thermal_time_tau_c": tau_th,
                "dt_tau_c": dt,
                "n_steps": n_steps,
                **{f"initial_{k}": v for k, v in initial_metrics.items()},
                **{f"final_{k}": v for k, v in final_metrics.items()},
                "initial_energy_ev": kinetic_initial[sl],
                "final_energy_ev": kinetic_final[sl],
                "runtime_seconds": elapsed,
            }
        )
    return rows


def run_scan(
    temperatures=DEFAULT_TEMPERATURES_EV,
    *,
    n_per_distribution: int = 4096,
    seed: int = 41,
    dt_thermal: float = DEFAULT_DT_THERMAL,
    relaxation_thermal: float = DEFAULT_RELAXATION_THERMAL,
):
    """Run the full temperature/initial-distribution relaxation scan."""

    kernel = _make_relaxation_kernel(jax.random.key(seed))
    temperatures = tuple(float(value) for value in temperatures)
    if temperatures:
        run_temperature_case(
            kernel,
            temperatures[0],
            n_per_distribution=n_per_distribution,
            seed=seed,
            dt_thermal=dt_thermal,
            relaxation_thermal=relaxation_thermal,
        )
    rows = []
    for te_ev in temperatures:
        rows.extend(
            run_temperature_case(
                kernel,
                float(te_ev),
                n_per_distribution=n_per_distribution,
                seed=seed,
                dt_thermal=dt_thermal,
                relaxation_thermal=relaxation_thermal,
            )
        )
    return rows


def check_small_angle_operator_relaxes_arbitrary_distributions_to_maxwellian():
    """Scan temperature and initial shape; all cases must reach one Maxwellian."""

    rows = run_scan(
        DEFAULT_TEMPERATURES_EV,
        n_per_distribution=1024,
        seed=73,
    )

    for row in rows:
        # Every chosen initial condition is visibly non-Maxwellian in energy.
        assert row["initial_ks_energy"] > 0.30

        # The relaxed energy CDF agrees with the Maxwellian to ordinary
        # O(N^-1/2) Monte Carlo accuracy at N=1024.
        assert row["final_ks_energy"] < 0.055
        assert row["final_ks_energy"] < 0.20 * row["initial_ks_energy"]

        # The energy moment recovers the background temperature and pitch
        # scattering recovers an isotropic distribution: <xi>=0, <xi^2>=1/3.
        assert abs(row["final_mean_energy_over_target"] - 1.0) < 0.10
        assert abs(row["final_mean_xi"]) < 0.07
        assert abs(row["final_mean_xi2"] - 1.0 / 3.0) < 0.055


def _write_csv(path: Path, rows):
    fields = [
        "temperature_ev",
        "initial_distribution",
        "n_markers",
        "thermal_time_tau_c",
        "dt_tau_c",
        "n_steps",
        "initial_ks_energy",
        "final_ks_energy",
        "initial_mean_energy_over_target",
        "final_mean_energy_over_target",
        "initial_mean_xi",
        "final_mean_xi",
        "initial_mean_xi2",
        "final_mean_xi2",
        "runtime_seconds",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row[name] for name in fields})


def _write_distribution_plot(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    temperatures = sorted({row["temperature_ev"] for row in rows})
    fig, axes = plt.subplots(
        len(temperatures),
        len(INITIAL_DISTRIBUTIONS),
        figsize=(12.0, 9.0),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)

    for i, te_ev in enumerate(temperatures):
        for j, kind in enumerate(INITIAL_DISTRIBUTIONS):
            ax = axes[i, j]
            row = next(
                r
                for r in rows
                if r["temperature_ev"] == te_ev
                and r["initial_distribution"] == kind
            )
            k0 = np.asarray(row["initial_energy_ev"]) / te_ev
            kf = np.asarray(row["final_energy_ev"]) / te_ev
            bins = np.linspace(0.0, 10.0, 61)
            ax.hist(k0, bins=bins, density=True, histtype="step", alpha=0.65, label="initial")
            ax.hist(kf, bins=bins, density=True, histtype="step", linewidth=1.7, label="relaxed")

            x = np.linspace(0.0, 10.0, 800)
            energy = x * te_ev
            pdf = maxwellian_energy_pdf(energy, te_ev)
            # Convert density in energy to density in K/T.
            pdf /= np.trapezoid(pdf, x)
            ax.plot(x, pdf, "--", linewidth=1.4, label="Maxwellian")
            ax.grid(True, alpha=0.25)
            if i == 0:
                ax.set_title(kind)
            if j == 0:
                ax.set_ylabel(f"T={te_ev:g} eV\nprobability density")
            if i == len(temperatures) - 1:
                ax.set_xlabel(r"kinetic energy $K/T_e$")
            ax.set_xlim(0.0, 10.0)
            ax.set_ylim(bottom=0.0)

    axes[0, -1].legend(fontsize=8)
    fig.suptitle("Small-angle collisions: relaxation to the background Maxwellian")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_summary_plot(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    for kind in INITIAL_DISTRIBUTIONS:
        selected = [row for row in rows if row["initial_distribution"] == kind]
        te = np.asarray([row["temperature_ev"] for row in selected])
        ks_initial = np.asarray([row["initial_ks_energy"] for row in selected])
        ks_final = np.asarray([row["final_ks_energy"] for row in selected])
        axes[0].loglog(te, ks_initial, "o--", alpha=0.45)
        axes[0].loglog(te, ks_final, "o-", label=kind)
        mean_ratio = np.asarray(
            [row["final_mean_energy_over_target"] for row in selected]
        )
        axes[1].semilogx(te, mean_ratio, "o-", label=kind)

    axes[0].set_xlabel(r"background $T_e$ [eV]")
    axes[0].set_ylabel("energy-distribution KS distance")
    axes[0].set_title("Initial (dashed) and relaxed (solid)")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].axhline(1.0, linestyle="--", linewidth=1.2)
    axes[1].set_xlabel(r"background $T_e$ [eV]")
    axes[1].set_ylabel(r"$\langle K\rangle/\langle K\rangle_M$")
    axes[1].set_title("Relaxed energy moment")
    axes[1].grid(True, which="both", alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_runtime_plot(path: Path, rows):
    """Plot synchronized runtime for each temperature case."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    temperatures = sorted({row["temperature_ev"] for row in rows})
    runtimes = []
    for te_ev in temperatures:
        case_rows = [row for row in rows if row["temperature_ev"] == te_ev]
        runtimes.append(float(np.median([row["runtime_seconds"] for row in case_rows])))

    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    bars = ax.bar([f"{te:g}" for te in temperatures], runtimes)
    ax.set_xlabel(r"background $T_e$ [eV]")
    ax.set_ylabel("synchronized runtime [s]")
    ax.set_title("Maxwellian-relaxation runtime by temperature")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, runtime in zip(bars, runtimes):
        ax.annotate(
            f"{runtime:.3f} s",
            (bar.get_x() + bar.get_width() / 2.0, runtime),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True
