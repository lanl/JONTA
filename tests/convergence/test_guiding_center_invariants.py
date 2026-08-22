"""Fixed-step convergence of axisymmetric guiding-center invariants.

This is JONTA's primary deterministic-orbit conservation/convergence test. It
combines the two invariants used by RAMc for verification of collisionless,
radiation-free guiding-center trajectories:

* toroidal canonical momentum P_phi, from axisymmetry;
* magnetic moment mu, the guiding-center adiabatic invariant.

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
short smoke test. Both invariants are measured in the *same* orbit
integration, avoiding duplicated particle pushes. One JIT compilation is
reused for every timestep and electric-field value for a given integrator, so
the same code path is suitable for CPU or GPU execution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.config import OrbitNormalization
from core.constants import C_LIGHT_M_S, E_CHARGE_C, M_E_KG
from core.state import CircularFieldProfiles, KinematicState
from diagnostics.invariants import (
    magnetic_moment_circular,
    toroidal_canonical_momentum_circular,
)
from integrators import midpoint_step, rk4_step
from orbits.ramc_circular import ramc_circular_rhs

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


def _make_measure_kernel(step):
    """Build one JIT-compiled kernel that measures both invariants.

    ``lax.while_loop`` keeps the number of integration steps dynamic, so a
    timestep/electric-field scan reuses one XLA compilation per integrator.
    """

    @jax.jit
    def kernel(dt_actual, n_steps, kin, profiles, norm):
        pphi0 = toroidal_canonical_momentum_circular(kin, 0.0, profiles, norm)
        mu0 = magnetic_moment_circular(kin, profiles, norm)
        pphi_scale = jnp.maximum(jnp.abs(pphi0), 1.0e-12)
        mu_scale = jnp.maximum(jnp.abs(mu0), 1.0e-12)

        def rhs(y, t):
            return ramc_circular_rhs(y, t, profiles, norm)

        def cond(carry):
            _, istep, _, _ = carry
            return istep < n_steps

        def body(carry):
            state, istep, max_pphi_error, max_mu_error = carry
            time = dt_actual * istep
            next_state = step(rhs, state, time, dt_actual)
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
            )

        return jax.lax.while_loop(
            cond,
            body,
            (
                kin,
                jnp.asarray(0, dtype=jnp.int32),
                jnp.asarray(0.0),
                jnp.asarray(0.0),
            ),
        )

    return kernel


def _integrate_and_measure(kernel, dt: float, final_time: float, kin, profiles, norm):
    """Integrate one ensemble and return maximum relative errors in both invariants."""

    n_steps = max(1, int(round(final_time / dt)))
    dt_actual = final_time / n_steps
    final_state, _, pphi_error, mu_error = kernel(
        jnp.asarray(dt_actual),
        jnp.asarray(n_steps, dtype=jnp.int32),
        kin,
        profiles,
        norm,
    )

    state_finite = jnp.all(jnp.isfinite(jnp.stack(final_state)))
    finite = bool(state_finite & jnp.isfinite(pphi_error) & jnp.isfinite(mu_error))
    return {
        "dt": float(dt_actual),
        "n_steps": n_steps,
        "pphi_max_rel_error": float(pphi_error),
        "mu_max_rel_error": float(mu_error),
        "finite": finite,
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
):
    """Run one compiled integrator kernel across all E1 and dt values."""

    kernel = _make_measure_kernel(step)
    scan = {}
    for e1 in electric_fields:
        kin, profiles, norm = make_case(
            n_particles=n_particles,
            q2=q2,
            e1=float(e1),
        )
        scan[float(e1)] = [
            _integrate_and_measure(kernel, dt, final_time, kin, profiles, norm)
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


@pytest.mark.slow
def test_midpoint_guiding_center_invariants_converge():
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


@pytest.mark.slow
def test_rk4_guiding_center_invariants_converge():
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


@pytest.mark.slow
def test_rk4_guiding_center_invariants_extreme_electric_field_scan_remains_finite():
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


def _write_integrator_plot(path: Path, integrator: str, field_scan, expected_order: int):
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

    repo_root = Path(__file__).resolve().parents[2]
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
        "integrators": ["midpoint/RK2", "RK4"],
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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    q2 = RAMC_Q2 if args.ramc_q_profile else 0.0
    datasets = {
        "midpoint/RK2": run_electric_field_scan(
            midpoint_step,
            args.dts,
            args.electric_fields,
            q2=q2,
            final_time=args.final_time,
            n_particles=args.n_particles,
        ),
        "RK4": run_electric_field_scan(
            rk4_step,
            args.dts,
            args.electric_fields,
            q2=q2,
            final_time=args.final_time,
            n_particles=args.n_particles,
        ),
    }

    _print_scan_summary(datasets)

    csv_path = args.output_dir / "guiding_center_invariants_efield_scan.csv"
    rk2_path = args.output_dir / "guiding_center_invariants_efield_scan_rk2.png"
    rk4_path = args.output_dir / "guiding_center_invariants_efield_scan_rk4.png"
    floor_path = args.output_dir / "guiding_center_invariants_efield_floor.png"
    metadata_path = args.output_dir / "guiding_center_invariants_efield_scan.json"
    _write_csv(csv_path, datasets)
    _write_metadata(metadata_path, args, q2)
    _write_integrator_plot(rk2_path, "midpoint/RK2", datasets["midpoint/RK2"], 2)
    _write_integrator_plot(rk4_path, "RK4", datasets["RK4"], 4)
    _write_floor_plot(floor_path, datasets)

    print(f"\nCSV: {csv_path}")
    print(f"RK2 plot: {rk2_path}")
    print(f"RK4 plot: {rk4_path}")
    print(f"Floor plot: {floor_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
