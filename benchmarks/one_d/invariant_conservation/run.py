"""Convergence of axisymmetric guiding-center invariants.

This is JONTA's primary deterministic-orbit conservation/convergence test. It
combines the two invariants used by RAMc for verification of collisionless,
radiation-free guiding-center trajectories:

* toroidal canonical momentum P_phi, from axisymmetry;
* magnetic moment mu, the guiding-center adiabatic invariant.

RK4, fixed Bogacki--Shampine 5, and adaptive Bogacki--Shampine 5(4) are
compared on the same collisionless trajectories.

Physics enabled
---------------
* axisymmetric circular tokamak guiding-center dynamics
* finite inductive electric field

Physics disabled
----------------
* small-angle collisions
* large-angle collisions
* synchrotron radiation
* sources and plasma feedback

The default convergence case uses constant q so that the test isolates the
orbit integrator rather than radial-profile interpolation error. The CLI can
also run the RAMc-like q(r)=q0+q2*r^2 profile.

The command-line driver has an explicit validation grid:
``E1/Ec = 0, 10^0, ..., 10^8`` (zero is a non-logarithmic anchor) and
``dt/tau_c = 5.12e-7, 1.28e-7, ..., 1.25e-10, 1e-10``. The timestep sequence
uses four-fold refinements through ``1.25e-10`` and a final floating-point
floor probe at ``1e-10``. These are deliberately broad stress tests, not a
short development run. Both invariants are measured in the *same* orbit
integration, avoiding duplicated particle pushes. One JIT compilation is
reused for every timestep and electric-field value for a given integrator, so
the same code path is suitable for CPU or GPU execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np

from core.config import OrbitNormalization
from core.constants import C_LIGHT_M_S, E_CHARGE_C, M_E_KG
from core.state import CircularFieldProfiles, KinematicState
from diagnostics.invariants import (
    magnetic_moment_circular,
    toroidal_canonical_momentum_circular,
)
from integrators import (
    adaptive_bogacki_shampine5_interval,
    bogacki_shampine5_step,
    midpoint_step,
    rk4_step,
)
from orbits.ramc_circular import ramc_circular_rhs
from parallel.distributed import (
    global_barrier,
    global_values,
    initialize_from_environment,
)

jax.config.update("jax_enable_x64", True)


RAMC_EPSILON = 1.0 / 3.0
RAMC_C_TAU_OVER_A = 2.3e6
RAMC_E1 = 10.0
RAMC_Q0 = 2.1
RAMC_Q2 = 2.0
RAMC_B0_T = 3.0
RAMC_A_M = 1.0

# Deliberately extends far beyond disruption-relevant fields to stress the
# deterministic orbit pusher. E1 is normalized to the Connor-Hastie field Ec.
DEFAULT_EFIELD_SCAN = (
    0.0,
    1.0,
    10.0,
    1.0e2,
    1.0e3,
    1.0e4,
    1.0e5,
    1.0e6,
    1.0e7,
    1.0e8,
)

# Covers the truncation-error regime through the FP64 floor. The endpoint is
# 1e-10 tau_c.
DEFAULT_DT_SCAN = (
    5.12e-7,
    1.28e-7,
    3.2e-8,
    8.0e-9,
    2.0e-9,
    5.0e-10,
    1.25e-10,
    1.0e-10,
)

INVARIANTS = ("pphi", "mu")

# Scaling case is fixed so timings remain comparable across backends and runs.
SCALING_E1 = 10.0
SCALING_DT = 1.28e-7
SCALING_FINAL_TIME = 5.0e-4
SCALING_TOTAL_PARTICLES = 512
SCALING_REPEATS = 3


def _bogacki_shampine5_state_step(rhs, state, time, dt):
    """Fixed-step adapter matching the legacy ``step(rhs, state, t, dt)`` API."""

    state_5, _, _ = bogacki_shampine5_step(rhs, state, time, dt)
    return state_5


def _a_omega_ce_over_c(b0_t=RAMC_B0_T, a_m=RAMC_A_M):
    omega_ce = E_CHARGE_C * b0_t / M_E_KG
    return a_m * omega_ce / C_LIGHT_M_S


def make_case(
    n_particles: int = 32,
    q2: float = 0.0,
    e1: float = RAMC_E1,
    seed: int = 17,
):
    """Construct a deterministic ensemble spanning trapped and passing pitches."""

    # Constant q is represented exactly by the diagnostic flux quadrature. A
    # fine grid is used for the optional RAMc-like q profile so interpolation
    # error stays below orbit error over most of the convergence scan.
    grid = jnp.linspace(0.0, 1.0, 4097 if q2 else 257)
    profiles = CircularFieldProfiles(
        r=grid,
        e1=jnp.full_like(grid, e1),
        q=RAMC_Q0 + q2 * grid * grid,
    )
    norm = OrbitNormalization(
        epsilon=RAMC_EPSILON,
        c_tau_over_a=RAMC_C_TAU_OVER_A,
        a_omega_ce_over_c=_a_omega_ce_over_c(),
        alpha_syn=0.0,
    )

    rng = np.random.default_rng(seed)
    r = np.linspace(0.10, 0.70, n_particles)
    theta = rng.uniform(0.0, 2.0 * np.pi, n_particles)
    gamma = rng.uniform(2.0, 20.0, n_particles)

    n_passing = n_particles // 2
    xi = np.concatenate(
        (
            rng.uniform(-0.95, -0.55, n_passing),
            rng.uniform(-0.25, 0.25, n_particles - n_passing),
        )
    )
    phi = rng.uniform(-np.pi, np.pi, n_particles)

    kin = KinematicState(
        gamma=jnp.asarray(gamma),
        xi=jnp.asarray(xi),
        x=jnp.asarray(r * np.cos(theta)),
        y=jnp.asarray(r * np.sin(theta)),
        phi=jnp.asarray(phi),
    )
    return kin, profiles, norm


def _make_measure_kernel(step, *, adaptive=False):
    """Build one JIT-compiled kernel that measures both invariants.

    ``lax.while_loop`` keeps the number of integration steps dynamic, so a
    timestep/electric-field scan reuses one XLA compilation per integrator.
    """

    @jax.jit
    def kernel(dt_actual, n_steps, initial_dt, kin, profiles, norm):
        pphi0 = toroidal_canonical_momentum_circular(kin, 0.0, profiles, norm)
        mu0 = magnetic_moment_circular(kin, profiles, norm)
        pphi_scale = jnp.maximum(jnp.abs(pphi0), 1.0e-12)
        mu_scale = jnp.maximum(jnp.abs(mu0), 1.0e-12)

        def rhs(y, t):
            return ramc_circular_rhs(y, t, profiles, norm)

        def cond(carry):
            _, istep, _, _, _, _, _, converged = carry
            return (istep < n_steps) & converged

        def body(carry):
            (
                state,
                istep,
                max_pphi_error,
                max_mu_error,
                proposed_dt,
                accepted_total,
                rejected_total,
                converged,
            ) = carry
            time = dt_actual * istep
            if adaptive:
                result = jax.vmap(
                    lambda marker, marker_dt: adaptive_bogacki_shampine5_interval(
                        rhs,
                        marker,
                        time,
                        dt_actual,
                        atol=1.0e-9,
                        rtol=1.0e-9,
                        initial_dt=marker_dt,
                    )
                )(state, proposed_dt)
                next_state = result.state
                next_proposed_dt = result.next_dt
                next_accepted_total = accepted_total + jnp.sum(result.accepted_steps).astype(jnp.int32)
                next_rejected_total = rejected_total + jnp.sum(result.rejected_steps).astype(jnp.int32)
                next_converged = converged & jnp.all(result.converged)
            else:
                next_state = step(rhs, state, time, dt_actual)
                next_proposed_dt = proposed_dt
                next_accepted_total = accepted_total
                next_rejected_total = rejected_total
                next_converged = converged
            next_istep = istep + 1

            # Compute time from the integer step index rather than repeatedly
            # adding dt; this prevents diagnostic-clock roundoff from masking
            # the orbit-integrator convergence.
            next_time = dt_actual * next_istep
            pphi = toroidal_canonical_momentum_circular(
                next_state, next_time, profiles, norm
            )
            mu = magnetic_moment_circular(next_state, profiles, norm)

            pphi_error = jnp.max(jnp.abs(pphi - pphi0) / pphi_scale)
            mu_error = jnp.max(jnp.abs(mu - mu0) / mu_scale)
            return (
                next_state,
                next_istep,
                jnp.maximum(max_pphi_error, pphi_error),
                jnp.maximum(max_mu_error, mu_error),
                next_proposed_dt,
                next_accepted_total,
                next_rejected_total,
                next_converged,
            )

        result = jax.lax.while_loop(
            cond,
            body,
            (
                kin,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
                jnp.full(kin.gamma.shape, initial_dt),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(True),
            ),
        )
        final_state, _, pphi_error, mu_error, _, accepted_total, rejected_total, converged = result
        return final_state, pphi_error, mu_error, accepted_total, rejected_total, converged

    return kernel


def _integrate_and_measure(
    kernel,
    dt: float,
    final_time: float,
    kin,
    profiles,
    norm,
    *,
    adaptive=False,
):
    """Integrate one ensemble and return maximum relative errors in both invariants."""

    if adaptive:
        dt_actual = final_time
        n_steps = 1
        initial_dt = min(dt, final_time)
    else:
        n_steps = max(1, int(round(final_time / dt)))
        dt_actual = final_time / n_steps
        initial_dt = dt_actual
    started = time.perf_counter()
    final_state, pphi_error, mu_error, accepted, rejected, converged = kernel(
        jnp.asarray(dt_actual),
        jnp.asarray(n_steps, dtype=jnp.int32),
        jnp.asarray(initial_dt),
        kin,
        profiles,
        norm,
    )
    jax.block_until_ready(
        (final_state, pphi_error, mu_error, accepted, rejected, converged)
    )
    elapsed = time.perf_counter() - started

    state_finite = jnp.all(jnp.isfinite(jnp.stack(final_state)))
    finite = bool(state_finite & jnp.isfinite(pphi_error) & jnp.isfinite(mu_error))
    return {
        "dt": float(dt if adaptive else dt_actual),
        "n_steps": int(accepted) if adaptive else n_steps,
        "output_interval": float(dt_actual),
        "pphi_max_rel_error": float(pphi_error),
        "mu_max_rel_error": float(mu_error),
        "finite": finite and bool(converged),
        "accepted_internal_steps": int(accepted),
        "rejected_internal_steps": int(rejected),
        "adaptive_converged": bool(converged),
        "runtime_seconds": elapsed,
    }


def run_convergence(
    step,
    dts,
    *,
    final_time: float = 2.0e-5,
    n_particles: int = 32,
    q2: float = 0.0,
    e1: float = RAMC_E1,
):
    """Run a fixed-E convergence scan with shared orbit pushes for both invariants."""

    kin, profiles, norm = make_case(n_particles=n_particles, q2=q2, e1=e1)
    kernel = _make_measure_kernel(step)
    return [
        _integrate_and_measure(kernel, dt, final_time, kin, profiles, norm)
        for dt in dts
    ]


def run_electric_field_scan(
    step,
    dts,
    electric_fields,
    *,
    final_time: float = 1.0e-5,
    n_particles: int = 8,
    q2: float = 0.0,
    adaptive: bool = False,
):
    """Run one compiled integrator kernel across all E1 and dt values."""

    electric_fields = tuple(float(value) for value in electric_fields)
    dts = tuple(float(value) for value in dts)
    kernel = _make_measure_kernel(step, adaptive=adaptive)
    # Compile and synchronize once before recording timings. JAX compilation
    # is a one-time cost and must not be conflated with case runtime.
    warmup_kin, warmup_profiles, warmup_norm = make_case(
        n_particles=n_particles,
        q2=q2,
        e1=electric_fields[0],
    )
    _integrate_and_measure(
        kernel,
        dts[0],
        final_time,
        warmup_kin,
        warmup_profiles,
        warmup_norm,
        adaptive=adaptive,
    )
    scan = {}
    for e1 in electric_fields:
        kin, profiles, norm = make_case(
            n_particles=n_particles,
            q2=q2,
            e1=float(e1),
        )
        scan[float(e1)] = [
            _integrate_and_measure(
                kernel,
                dt,
                final_time,
                kin,
                profiles,
                norm,
                adaptive=adaptive,
            )
            for dt in dts
        ]
    return scan


def observed_orders(rows, invariant: str):
    """Return pairwise observed order for ``pphi`` or ``mu``."""

    key = f"{invariant}_max_rel_error"
    orders = [math.nan]
    for coarse, fine in zip(rows[:-1], rows[1:]):
        if coarse[key] <= 0.0 or fine[key] <= 0.0:
            orders.append(math.nan)
            continue
        orders.append(
            math.log(coarse[key] / fine[key])
            / math.log(coarse["dt"] / fine["dt"])
        )
    return orders


def check_midpoint_guiding_center_invariants_converge():
    rows = run_convergence(
        midpoint_step,
        [1.28e-7, 6.4e-8, 3.2e-8, 1.6e-8, 8.0e-9],
        final_time=1.0e-5,
        n_particles=8,
    )
    assert all(row["finite"] for row in rows)

    pphi_errors = np.array([row["pphi_max_rel_error"] for row in rows])
    mu_errors = np.array([row["mu_max_rel_error"] for row in rows])
    assert pphi_errors[-1] < 1.0e-5
    assert mu_errors[-1] < 1.0e-5
    assert pphi_errors[0] / pphi_errors[-1] > 100.0
    assert mu_errors[0] / mu_errors[-1] > 100.0
    assert np.nanmedian(observed_orders(rows, "pphi")[-3:]) > 1.7
    assert np.nanmedian(observed_orders(rows, "mu")[-3:]) > 1.7


def check_rk4_guiding_center_invariants_converge():
    rows = run_convergence(
        rk4_step,
        [2.56e-7, 1.28e-7, 6.4e-8, 3.2e-8, 1.6e-8],
        final_time=1.0e-5,
        n_particles=8,
    )
    assert all(row["finite"] for row in rows)

    pphi_errors = np.array([row["pphi_max_rel_error"] for row in rows])
    mu_errors = np.array([row["mu_max_rel_error"] for row in rows])
    assert pphi_errors[-1] < 1.0e-9
    assert mu_errors[-1] < 1.0e-9
    assert pphi_errors[0] / pphi_errors[-1] > 1.0e3
    assert mu_errors[0] / mu_errors[-1] > 1.0e3


def check_rk4_guiding_center_invariants_extreme_electric_field_scan_remains_finite():
    """Guard the full extreme-field scan against non-finite orbit states."""

    scans = run_electric_field_scan(
        rk4_step,
        [3.2e-8, 8.0e-9, 2.0e-9],
        [0.0, 1.0e4, 1.0e6, 1.0e8],
        final_time=2.0e-6,
        n_particles=4,
    )
    for rows in scans.values():
        assert all(row["finite"] for row in rows)
        assert rows[-1]["pphi_max_rel_error"] < rows[0]["pphi_max_rel_error"]
        assert rows[-1]["mu_max_rel_error"] < rows[0]["mu_max_rel_error"]


def _write_csv(path: Path, datasets):
    with path.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(
            [
                "integrator",
                "E1_over_Ec",
                "dt",
                "n_steps",
                "finite",
                "pphi_max_rel_error",
                "pphi_observed_order",
                "mu_max_rel_error",
                "mu_observed_order",
                "output_interval",
                "accepted_internal_steps",
                "rejected_internal_steps",
                "adaptive_converged",
                "runtime_seconds",
            ]
        )
        for integrator, field_scan in datasets.items():
            for e1, rows in field_scan.items():
                pphi_orders = observed_orders(rows, "pphi")
                mu_orders = observed_orders(rows, "mu")
                for row, pphi_order, mu_order in zip(rows, pphi_orders, mu_orders):
                    writer.writerow(
                        [
                            integrator,
                            e1,
                            row["dt"],
                            row["n_steps"],
                            row["finite"],
                            row["pphi_max_rel_error"],
                            pphi_order,
                            row["mu_max_rel_error"],
                            mu_order,
                            row["output_interval"],
                            row["accepted_internal_steps"],
                            row["rejected_internal_steps"],
                            row["adaptive_converged"],
                            row["runtime_seconds"],
                        ]
                    )


def _field_label(e1):
    if e1 == 0.0:
        return r"$E_1/E_c=0$"
    exponent = int(round(math.log10(e1)))
    return rf"$E_1/E_c=10^{{{exponent}}}$"


def _reference_line(rows, invariant: str, expected_order: int):
    key = f"{invariant}_max_rel_error"
    dt = np.asarray([row["dt"] for row in rows])
    error = np.asarray([row[key] for row in rows])
    target_dt = 3.2e-8 if expected_order == 2 else 1.28e-7
    anchor = int(np.argmin(np.abs(dt - target_dt)))
    return dt, error[anchor] * (dt / dt[anchor]) ** expected_order


def _write_integrator_plot(path: Path, integrator: str, field_scan, expected_order=None):
    """Write one two-panel convergence plot for P_phi and mu."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    specs = (
        ("pphi", r"max $|\Delta P_\phi/P_{\phi,0}|$", r"Toroidal $P_\phi$"),
        ("mu", r"max $|\Delta\mu/\mu_0|$", r"Magnetic moment $\mu$"),
    )

    anchor_rows = field_scan.get(10.0, field_scan[next(iter(field_scan))])
    for ax, (invariant, ylabel, title) in zip(axes, specs):
        key = f"{invariant}_max_rel_error"
        for e1, rows in field_scan.items():
            dt = np.asarray([row["dt"] for row in rows])
            error = np.asarray([row[key] for row in rows])
            ax.loglog(dt, error, "o-", label=_field_label(e1))

        if expected_order is not None:
            dt_ref, ref = _reference_line(anchor_rows, invariant, expected_order)
            ax.loglog(
                dt_ref,
                ref,
                "--",
                linewidth=1.4,
                label=rf"expected $\Delta t^{{{expected_order}}}$",
            )
        ax.set_xlabel(r"fixed timestep $\Delta t/\tau_c$")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(1.0e-15, 1.0)

    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle(f"Guiding-center invariants: {integrator} electric-field scan")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_floor_plot(path: Path, datasets):
    """Plot the best invariant errors reached in the timestep scan."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6))
    specs = (
        ("pphi", r"minimum $P_\phi$ error"),
        ("mu", r"minimum $\mu$ error"),
    )
    for ax, (invariant, ylabel) in zip(axes, specs):
        key = f"{invariant}_max_rel_error"
        for integrator, field_scan in datasets.items():
            e_nonzero = []
            best = []
            for e1, rows in field_scan.items():
                if e1 == 0.0:
                    continue
                e_nonzero.append(e1)
                finite_errors = [
                    row[key]
                    for row in rows
                    if row["finite"] and math.isfinite(row[key])
                ]
                best.append(min(finite_errors) if finite_errors else math.nan)
            ax.loglog(e_nonzero, best, "o-", label=integrator)
        ax.set_xlabel(r"$E_1/E_c$")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.25)

    axes[0].legend()
    fig.suptitle("Guiding-center invariant error floors versus electric field")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_runtime_plot(path: Path, datasets):
    """Plot synchronized wall time for every field/timestep case."""

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    integrators = tuple(datasets)
    fig, axes = plt.subplots(
        1,
        len(integrators),
        figsize=(5.0 * len(integrators), 4.8),
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    axes = axes[0]
    for axis, (integrator, field_scan) in zip(axes, datasets.items()):
        fields = tuple(field_scan)
        colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(fields)))
        all_runtime = []
        for color, (e1, rows) in zip(colors, field_scan.items()):
            dt = np.asarray([row["dt"] for row in rows])
            runtime = np.maximum(
                np.asarray([row["runtime_seconds"] for row in rows]),
                np.finfo(float).tiny,
            )
            all_runtime.append(runtime)
            axis.loglog(dt, runtime, "o-", color=color, ms=3, label=_field_label(e1))
        median_runtime = np.median(np.asarray(all_runtime), axis=0)
        axis.loglog(
            np.asarray([row["dt"] for row in field_scan[fields[0]]]),
            median_runtime,
            "k--",
            linewidth=1.5,
            label="field median",
        )
        axis.set_title(integrator)
        axis.set_xlabel(r"requested timestep $\Delta t/\tau_c$")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
    axes[0].set_ylabel("synchronized case runtime [s]")
    fig.suptitle("Guiding-center invariant benchmark: runtime by case")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _shard_state(kin, n_devices: int):
    """Reshape marker state to ``(device, marker_per_device, ...)`` for pmap."""

    leaves, treedef = jax.tree_util.tree_flatten(kin)
    if any(leaf.shape[0] % n_devices for leaf in leaves):
        raise ValueError("particle count must divide evenly across scaling devices")
    n_local = leaves[0].shape[0] // n_devices
    sharded = [leaf.reshape((n_devices, n_local) + leaf.shape[1:]) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, sharded)


def _resolve_local_devices(platform):
    if platform == "auto":
        return tuple(jax.local_devices())
    backend = "gpu" if platform == "cuda" else platform
    return tuple(jax.local_devices(backend=backend))


def _slice_state(kin, start, stop):
    return jax.tree_util.tree_map(lambda value: value[start:stop], kin)


def _run_scaling_case(
    n_devices: int,
    platform: str,
    distributed: bool,
    total_particles: int,
    final_time: float,
    dt_requested: float,
    repeats: int,
):
    """Time fixed RK4 orbit work across selected local or global devices."""

    devices = _resolve_local_devices(platform)
    if n_devices > len(devices):
        raise RuntimeError(
            f"requested {n_devices} {platform} devices, but only "
            f"{len(devices)} local devices are available"
        )
    process_count = jax.process_count() if distributed else 1
    process_index = jax.process_index() if distributed else 0
    global_devices = n_devices * process_count
    if total_particles % global_devices:
        raise ValueError(
            f"scaling workload ({total_particles}) must divide evenly "
            f"across {global_devices} devices"
        )

    kin, profiles, norm = make_case(
        n_particles=total_particles,
        e1=SCALING_E1,
        seed=17,
    )
    if process_count > 1:
        particles_per_process = total_particles // process_count
        start = process_index * particles_per_process
        kin = _slice_state(kin, start, start + particles_per_process)
    particles_per_device = total_particles // global_devices
    sharded_kin = _shard_state(kin, n_devices)
    n_steps = max(1, int(round(final_time / dt_requested)))
    dt_actual = final_time / n_steps
    kernel = _make_measure_kernel(rk4_step)

    def mapped_fn(dt, steps, state, field_profiles, normalization):
        return kernel(dt, steps, dt, state, field_profiles, normalization)

    mapped_kernel = jax.pmap(
        mapped_fn,
        in_axes=(None, None, 0, None, None),
        devices=devices[:n_devices],
    )

    global_barrier(distributed, f"invariants-warmup-{n_devices}")
    warmup = mapped_kernel(
        jnp.asarray(dt_actual),
        jnp.asarray(n_steps, dtype=jnp.int32),
        sharded_kin,
        profiles,
        norm,
    )
    jax.block_until_ready(warmup)
    global_barrier(distributed, f"invariants-warmup-done-{n_devices}")

    timings = []
    result = None
    for _ in range(repeats):
        global_barrier(distributed, f"invariants-start-{n_devices}")
        start = time.perf_counter()
        result = mapped_kernel(
            jnp.asarray(dt_actual),
            jnp.asarray(n_steps, dtype=jnp.int32),
            sharded_kin,
            profiles,
            norm,
        )
        jax.block_until_ready(result)
        elapsed = time.perf_counter() - start
        global_barrier(distributed, f"invariants-done-{n_devices}")
        timings.append(float(np.max(global_values(elapsed, distributed))))

    final_state, _, pphi_error, mu_error, _, _ = result
    state_finite = jnp.all(jnp.stack(jax.tree_util.tree_leaves(final_state)))
    finite = bool(
        state_finite
        & jnp.all(jnp.isfinite(pphi_error))
        & jnp.all(jnp.isfinite(mu_error))
    )
    global_pphi = np.max(global_values(float(jnp.max(pphi_error)), distributed))
    global_mu = np.max(global_values(float(jnp.max(mu_error)), distributed))
    global_finite = bool(np.all(global_values(int(finite), distributed)))
    return {
        "devices": global_devices,
        "local_devices": n_devices,
        "processes": process_count,
        "platform": platform,
        "particles_per_device": particles_per_device,
        "total_particles": total_particles,
        "wall_time_s": float(np.median(timings)),
        "timing_trials_s": [float(value) for value in timings],
        "pphi_max_rel_error": float(global_pphi),
        "mu_max_rel_error": float(global_mu),
        "finite": global_finite,
    }


def run_scaling(
    device_counts,
    platform,
    distributed,
    total_particles,
    final_time,
    dt_requested,
    repeats,
    weak_scaling=False,
    particles_per_device=None,
):
    """Run strong particle scaling on selected local or global devices."""

    counts = [int(value) for value in device_counts]
    if not counts or counts[0] != 1:
        raise ValueError("--scaling-devices must be non-empty and start with 1")
    if any(value < 1 for value in counts) or len(set(counts)) != len(counts):
        raise ValueError("--scaling-devices must contain unique positive counts")

    process_count = jax.process_count() if distributed else 1
    rows = []
    for count in counts:
        case_particles = total_particles
        if weak_scaling:
            if particles_per_device is None or particles_per_device < 1:
                raise ValueError("weak scaling requires positive particles_per_device")
            case_particles = particles_per_device * count * process_count
        rows.append(
            _run_scaling_case(
                count,
                platform,
                distributed,
                case_particles,
                final_time,
                dt_requested,
                repeats,
            )
        )
    serial_time = rows[0]["wall_time_s"]
    for row in rows:
        row["speedup"] = serial_time / row["wall_time_s"]
        row["ideal_speedup"] = 1.0 if weak_scaling else float(row["devices"])
        row["parallel_efficiency"] = row["speedup"] / row["ideal_speedup"]
        row["weak_scaling"] = weak_scaling
    return rows


def run_particle_sweep(particle_counts, platform, final_time, dt_requested, repeats):
    """Measure fixed RK4 case over marker counts on one local device."""

    rows = []
    for count in particle_counts:
        if count < 1:
            raise ValueError("particle counts must be positive")
        row = _run_scaling_case(
            1,
            platform,
            False,
            int(count),
            final_time,
            dt_requested,
            repeats,
        )
        row["particle_count"] = int(count)
        row["markers_per_second"] = count / row["wall_time_s"]
        rows.append(row)
    return rows


def _write_particle_sweep(path: Path, rows):
    fields = (
        "particle_count",
        "wall_time_s",
        "markers_per_second",
        "pphi_max_rel_error",
        "mu_max_rel_error",
        "finite",
        "platform",
    )
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _write_particle_sweep_plot(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    particles = np.asarray([row["particle_count"] for row in rows])
    runtime = np.asarray([row["wall_time_s"] for row in rows])
    throughput = np.asarray([row["markers_per_second"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)
    axes[0].loglog(particles, runtime, "o-")
    axes[0].set_xlabel("markers")
    axes[0].set_ylabel("runtime [s]")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[1].semilogx(particles, throughput, "o-")
    axes[1].set_xlabel("markers")
    axes[1].set_ylabel("markers/s")
    axes[1].grid(True, which="both", alpha=0.25)
    fig.suptitle("Guiding-center invariant benchmark: single-GPU particle sweep")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_scaling_csv(path: Path, rows):
    fields = (
        "devices",
        "local_devices",
        "processes",
        "platform",
        "weak_scaling",
        "particles_per_device",
        "total_particles",
        "wall_time_s",
        "speedup",
        "ideal_speedup",
        "parallel_efficiency",
        "pphi_max_rel_error",
        "mu_max_rel_error",
        "finite",
    )
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)


def _write_scaling_plot(path: Path, rows, efficiency_ymin=0.0):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    devices = np.asarray([row["devices"] for row in rows])
    speedup = np.asarray([row["speedup"] for row in rows])
    ideal = np.asarray([row["ideal_speedup"] for row in rows])
    efficiency = np.asarray([row["parallel_efficiency"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axes[0].plot(devices, speedup, "o-", label="measured")
    axes[0].plot(devices, ideal, "--", label="ideal")
    axes[0].set_ylabel("speedup relative to baseline")
    axes[0].set_xlabel("devices")
    axes[0].set_xticks(devices)
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()
    axes[1].plot(devices, efficiency, "o-")
    axes[1].set_ylabel("parallel efficiency")
    axes[1].set_xlabel("devices")
    axes[1].set_xticks(devices)
    axes[1].set_ylim(bottom=efficiency_ymin)
    axes[1].grid(True, alpha=0.25)
    fig.suptitle("Guiding-center invariant benchmark: particle scaling")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_scaling_metadata(path: Path, args, rows):
    """Write fixed-case scaling configuration and measured rows."""

    repo_root = Path(__file__).resolve().parents[3]
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
        git_dirty = None

    payload = {
        "benchmark": "guiding_center_invariants",
        "mode": "particle_scaling",
        "command": " ".join([sys.executable, *sys.argv]),
        "backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "requested_device_counts": [int(value) for value in args.scaling_devices],
        "fixed_case": {
            "integrator": "RK4",
            "E1_over_Ec": SCALING_E1,
            "dt_over_tau_c": args.scaling_dt,
            "final_time_over_tau_c": args.scaling_final_time,
            "total_particles": args.scaling_particles,
            "particles_per_device": args.particles_per_device,
            "timing_repeats_after_warmup": args.scaling_repeats,
        },
        "rows": rows,
        "platform": args.platform,
        "distributed": args.distributed,
        "process_count": jax.process_count() if args.distributed else 1,
        "interpretation": (
            "Device counts partition a fixed total marker workload with pmap. "
            "Speedup uses first requested device count as baseline. In distributed "
            "mode, each process maps local devices and host barriers report global time."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _print_scan_summary(datasets):
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    for integrator, field_scan in datasets.items():
        print(f"\n{integrator}")
        print(
            "E1/Ec        best Pphi      dt(Pphi)       best mu        dt(mu)"
        )
        for e1, rows in field_scan.items():
            values = {}
            for invariant in INVARIANTS:
                key = f"{invariant}_max_rel_error"
                errors = np.asarray([row[key] for row in rows], dtype=float)
                finite = np.asarray([row["finite"] for row in rows]) & np.isfinite(errors)
                if np.any(finite):
                    indices = np.flatnonzero(finite)
                    idx = int(indices[np.argmin(errors[finite])])
                    values[invariant] = (errors[idx], rows[idx]["dt"])
                else:
                    values[invariant] = (math.nan, math.nan)
            print(
                f"{e1:10.1e}  {values['pphi'][0]:12.4e}  {values['pphi'][1]:11.4e}  "
                f"{values['mu'][0]:12.4e}  {values['mu'][1]:11.4e}"
            )


def _write_metadata(path: Path, args, q2: float):
    """Write the exact requested scan so result directories are self-describing."""

    repo_root = Path(__file__).resolve().parents[3]
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        git_dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        git_commit = "unknown"
        git_dirty = None

    def package_version(name):
        try:
            from importlib.metadata import version

            return version(name)
        except Exception:  # pragma: no cover - depends on installation context
            return "unknown"

    payload = {
        "benchmark": "guiding_center_invariants",
        "scan_type": (
            "full_default_grid"
            if tuple(args.electric_fields) == DEFAULT_EFIELD_SCAN
            and tuple(args.dts) == DEFAULT_DT_SCAN
            else "custom_grid"
        ),
        "electric_fields_E1_over_Ec": [float(value) for value in args.electric_fields],
        "timesteps_dt_over_tau_c_requested": [float(value) for value in args.dts],
        "final_time_over_tau_c": float(args.final_time),
        "n_particles": int(args.n_particles),
        "q_profile": "q(r)=2.1+2r^2" if q2 else "constant q=2.1",
        "integrators": ["RK4", "BS5", "BS5-adaptive"],
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": package_version("jax"),
            "jaxlib": package_version("jaxlib"),
            "numpy": package_version("numpy"),
            "jax_enable_x64": bool(jax.config.read("jax_enable_x64")),
        },
        "timing": {
            "per_case_field": True,
            "compilation_excluded": True,
            "synchronization": "jax.block_until_ready",
            "runtime_field": "runtime_seconds",
        },
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "interpretation": (
            "E1/Ec=0 is a non-logarithmic anchor; positive fields are decade-spaced. "
            "The timestep sequence is a four-fold refinement scan followed by a "
            "1e-10 tau_c floating-point-floor probe."
        ),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--scaling",
        action="store_true",
        help="run fixed particle scaling case instead of convergence scan",
    )
    parser.add_argument(
        "--scaling-devices",
        nargs="+",
        type=int,
        default=[1],
        help="local device counts for --scaling; must start with 1",
    )
    parser.add_argument(
        "--platform",
        choices=("auto", "cpu", "gpu", "cuda"),
        default="auto",
        help="device backend for scaling; auto uses JAX default backend",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="enable multi-process scaling using JAX distributed initialization",
    )
    parser.add_argument(
        "--scaling-particles",
        type=int,
        default=SCALING_TOTAL_PARTICLES,
        help="total fixed marker workload for scaling",
    )
    parser.add_argument(
        "--particles-per-device",
        type=int,
        default=None,
        help="per-device workload for --weak-scaling",
    )
    parser.add_argument(
        "--weak-scaling",
        action="store_true",
        help="grow total workload with global device count",
    )
    parser.add_argument(
        "--scaling-final-time",
        type=float,
        default=SCALING_FINAL_TIME,
        help="fixed-case final time for scaling/throughput",
    )
    parser.add_argument(
        "--scaling-dt",
        type=float,
        default=SCALING_DT,
        help="fixed-case timestep for scaling/throughput",
    )
    parser.add_argument(
        "--scaling-repeats",
        type=int,
        default=SCALING_REPEATS,
        help="timed repeats after warmup",
    )
    parser.add_argument(
        "--particle-sweep",
        nargs="+",
        type=int,
        default=None,
        help="single-device marker counts for runtime/throughput sweep",
    )
    parser.add_argument(
        "--ramc-q-profile",
        action="store_true",
        help="use q(r)=2.1+2r^2 instead of constant q",
    )
    parser.add_argument(
        "--electric-fields",
        nargs="+",
        type=float,
        default=list(DEFAULT_EFIELD_SCAN),
        help="E1/Ec values; default is 0 plus 10^0 through 10^8",
    )
    parser.add_argument(
        "--dts",
        nargs="+",
        type=float,
        default=list(DEFAULT_DT_SCAN),
        help="fixed dt/tau_c values; default is the full refinement/floor grid",
    )
    parser.add_argument("--final-time", type=float, default=1.0e-5)
    parser.add_argument("--n-particles", type=int, default=8)
    args = parser.parse_args()
    if args.distributed:
        initialize_from_environment()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.particle_sweep is not None:
        if args.scaling or args.distributed:
            raise ValueError("--particle-sweep cannot combine with scaling/distributed")
        rows = run_particle_sweep(
            args.particle_sweep,
            args.platform,
            args.scaling_final_time,
            args.scaling_dt,
            args.scaling_repeats,
        )
        csv_path = args.output_dir / "guiding_center_invariants_particle_sweep.csv"
        plot_path = args.output_dir / "guiding_center_invariants_particle_sweep.png"
        _write_particle_sweep(csv_path, rows)
        _write_particle_sweep_plot(plot_path, rows)
        metadata = {
            "benchmark": "guiding_center_invariants",
            "mode": "single_device_particle_sweep",
            "platform": args.platform,
            "backend": jax.default_backend(),
            "jax_devices": [str(device) for device in jax.devices()],
            "particle_counts": [int(value) for value in args.particle_sweep],
            "fixed_case": {
                "integrator": "RK4",
                "E1_over_Ec": SCALING_E1,
                "dt_over_tau_c": args.scaling_dt,
                "final_time_over_tau_c": args.scaling_final_time,
                "timing_repeats": args.scaling_repeats,
            },
            "rows": rows,
        }
        metadata_path = args.output_dir / "guiding_center_invariants_particle_sweep.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        print(f"CSV: {csv_path}")
        print(f"Plot: {plot_path}")
        print(f"Metadata: {metadata_path}")
        return

    if args.scaling:
        if args.scaling_particles < 1:
            raise ValueError("--scaling-particles must be positive")
        if args.scaling_final_time <= 0.0 or args.scaling_dt <= 0.0:
            raise ValueError("scaling final time and timestep must be positive")
        if args.scaling_repeats < 1:
            raise ValueError("--scaling-repeats must be positive")
        if args.weak_scaling and args.particles_per_device is None:
            raise ValueError("--weak-scaling requires --particles-per-device")
        rows = run_scaling(
            args.scaling_devices,
            args.platform,
            args.distributed,
            args.scaling_particles,
            args.scaling_final_time,
            args.scaling_dt,
            args.scaling_repeats,
            args.weak_scaling,
            args.particles_per_device,
        )
        if args.distributed and jax.process_index() != 0:
            return
        csv_path = args.output_dir / "guiding_center_invariants_scaling.csv"
        plot_path = args.output_dir / "guiding_center_invariants_scaling.png"
        metadata_path = args.output_dir / "guiding_center_invariants_scaling.json"
        _write_scaling_csv(csv_path, rows)
        _write_scaling_plot(
            plot_path,
            rows,
            efficiency_ymin=0.5 if args.weak_scaling else 0.0,
        )
        _write_scaling_metadata(metadata_path, args, rows)
        print("\nParticle scaling")
        print("devices  wall_time_s  speedup  ideal  efficiency")
        for row in rows:
            print(
                f"{row['devices']:7d}  {row['wall_time_s']:11.4f}  "
                f"{row['speedup']:7.3f}  {row['ideal_speedup']:5.1f}  "
                f"{row['parallel_efficiency']:10.3f}"
            )
        print(f"\nCSV: {csv_path}")
        print(f"Plot: {plot_path}")
        print(f"Metadata: {metadata_path}")
        return

    q2 = RAMC_Q2 if args.ramc_q_profile else 0.0
    datasets = {
        "RK4": run_electric_field_scan(
            rk4_step,
            args.dts,
            args.electric_fields,
            q2=q2,
            final_time=args.final_time,
            n_particles=args.n_particles,
        ),
        "BS5": run_electric_field_scan(
            _bogacki_shampine5_state_step,
            args.dts,
            args.electric_fields,
            q2=q2,
            final_time=args.final_time,
            n_particles=args.n_particles,
        ),
        "BS5-adaptive": run_electric_field_scan(
            adaptive_bogacki_shampine5_interval,
            args.dts,
            args.electric_fields,
            q2=q2,
            final_time=args.final_time,
            n_particles=args.n_particles,
            adaptive=True,
        ),
    }

    _print_scan_summary(datasets)

    csv_path = args.output_dir / "guiding_center_invariants_efield_scan.csv"
    rk4_path = args.output_dir / "guiding_center_invariants_efield_scan_rk4.png"
    bs5_path = args.output_dir / "guiding_center_invariants_efield_scan_bs5.png"
    adaptive_path = args.output_dir / "guiding_center_invariants_efield_scan_bs5_adaptive.png"
    floor_path = args.output_dir / "guiding_center_invariants_efield_floor.png"
    runtime_path = args.output_dir / "guiding_center_invariants_runtime.png"
    metadata_path = args.output_dir / "guiding_center_invariants_efield_scan.json"
    _write_csv(csv_path, datasets)
    _write_metadata(metadata_path, args, q2)
    _write_integrator_plot(rk4_path, "RK4", datasets["RK4"], 4)
    _write_integrator_plot(bs5_path, "BS5", datasets["BS5"], 5)
    _write_integrator_plot(
        adaptive_path,
        "BS5-adaptive",
        datasets["BS5-adaptive"],
    )
    _write_floor_plot(floor_path, datasets)
    _write_runtime_plot(runtime_path, datasets)

    print(f"\nCSV: {csv_path}")
    print(f"RK4 plot: {rk4_path}")
    print(f"BS5 plot: {bs5_path}")
    print(f"Adaptive BS5 plot: {adaptive_path}")
    print(f"Floor plot: {floor_path}")
    print(f"Runtime plot: {runtime_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
