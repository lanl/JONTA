"""Collisionless trapped/passing circular-tokamak orbit benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import jax
import jax.numpy as jnp
import numpy as np
import yaml

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
    rk4_step,
)
from orbits.ramc_circular import ramc_circular_rhs

jax.config.update("jax_enable_x64", True)

def _make_profiles(config):
    case = config["case"]
    grid = jnp.linspace(0.0, 1.0, 257)
    return CircularFieldProfiles(
        grid,
        jnp.zeros_like(grid),
        float(case["q0"]) + float(case["q2"]) * grid * grid,
    )


def _make_norm(config):
    case = config["case"]
    b0_t = float(case["b0_t"])
    a_m = float(case["a_m"])
    omega = a_m * E_CHARGE_C * b0_t / (M_E_KG * C_LIGHT_M_S)
    return OrbitNormalization(
        float(case["epsilon"]),
        float(case["c_tau_over_a"]),
        omega,
        0.0,
    )


def make_case(n_particles: int = 2, config=None):
    """Return paired near-trapped and passing markers, repeated if needed."""

    if n_particles < 2 or n_particles % 2:
        raise ValueError("n_particles must be an even number >= 2")
    config = config or {
        "case": {
            "gamma_trapped": 10.0,
            "xi_trapped": -0.08,
            "gamma_passing": 10.0,
            "xi_passing": -0.80,
            "r0": 0.40,
            "theta0": 0.0,
            "phi0": 0.0,
        }
    }
    case = config["case"]
    n_each = n_particles // 2
    r = np.full(n_each, float(case["r0"]))
    theta = np.full(n_each, float(case["theta0"]))
    phi = np.full(n_each, float(case["phi0"]))
    gamma_trapped = np.full(n_each, float(case["gamma_trapped"]))
    gamma_passing = np.full(n_each, float(case["gamma_passing"]))
    kin = KinematicState(
        jnp.asarray(np.concatenate((gamma_trapped, gamma_passing))),
        jnp.asarray(
            np.concatenate(
                (
                    np.full(n_each, float(case["xi_trapped"])),
                    np.full(n_each, float(case["xi_passing"])),
                )
            )
        ),
        jnp.asarray(np.concatenate((r * np.cos(theta), r * np.cos(theta)))),
        jnp.asarray(np.concatenate((r * np.sin(theta), r * np.sin(theta)))),
        jnp.asarray(np.concatenate((phi, phi))),
    )
    labels = np.asarray(["trapped"] * n_each + ["passing"] * n_each)
    return kin, labels, _make_profiles(config), _make_norm(config)


def _make_trajectory_kernel(
    dt: float,
    n_steps: int,
    integrator: str = "rk4",
):
    if integrator not in (
        "rk4",
        "bogacki_shampine5",
        "bogacki_shampine5_adaptive",
    ):
        raise ValueError(f"unknown tokamak orbit integrator: {integrator}")

    @jax.jit
    def kernel(kin, profiles, norm):
        def rhs(state, t):
            return ramc_circular_rhs(state, t, profiles, norm)

        def body(carry, istep):
            if integrator == "bogacki_shampine5_adaptive":
                state, proposed_dt = carry
            else:
                state = carry
            t = dt * istep
            if integrator == "rk4":
                nxt = rk4_step(rhs, state, t, dt)
                accepted = jnp.zeros(state.gamma.shape, dtype=jnp.int32)
                rejected = jnp.zeros(state.gamma.shape, dtype=jnp.int32)
                converged = jnp.ones(state.gamma.shape, dtype=bool)
            elif integrator == "bogacki_shampine5":
                nxt, _embedded, _error = bogacki_shampine5_step(rhs, state, t, dt)
                accepted = jnp.ones(state.gamma.shape, dtype=jnp.int32)
                rejected = jnp.zeros(state.gamma.shape, dtype=jnp.int32)
                converged = jnp.ones(state.gamma.shape, dtype=bool)
            else:
                def adaptive_marker(marker, marker_dt):
                    return adaptive_bogacki_shampine5_interval(
                        rhs,
                        marker,
                        t,
                        dt,
                        atol=1.0e-9,
                        rtol=1.0e-9,
                        initial_dt=marker_dt,
                    )

                adaptive = jax.vmap(adaptive_marker)(state, proposed_dt)
                nxt = adaptive.state
                accepted = adaptive.accepted_steps
                rejected = adaptive.rejected_steps
                converged = adaptive.converged
            r = jnp.sqrt(nxt.x * nxt.x + nxt.y * nxt.y)
            theta = jnp.arctan2(nxt.y, nxt.x)
            pphi = toroidal_canonical_momentum_circular(
                nxt, t + dt, profiles, norm
            )
            mu = magnetic_moment_circular(nxt, profiles, norm)
            diagnostics = jnp.stack((r, theta, nxt.xi, nxt.gamma, pphi, mu))
            if integrator == "bogacki_shampine5_adaptive":
                return (nxt, adaptive.next_dt), (diagnostics, accepted, rejected, converged)
            return nxt, diagnostics

        initial_r = jnp.sqrt(kin.x * kin.x + kin.y * kin.y)
        initial_theta = jnp.arctan2(kin.y, kin.x)
        initial_pphi = toroidal_canonical_momentum_circular(
            kin, 0.0, profiles, norm
        )
        initial_mu = magnetic_moment_circular(kin, profiles, norm)
        initial = jnp.stack(
            (initial_r, initial_theta, kin.xi, kin.gamma, initial_pphi, initial_mu)
        )
        initial_carry = (
            (kin, jnp.full(kin.gamma.shape, dt))
            if integrator == "bogacki_shampine5_adaptive"
            else kin
        )
        final, history = jax.lax.scan(body, initial_carry, jnp.arange(n_steps))
        del final
        if integrator == "bogacki_shampine5_adaptive":
            trajectory, accepted, rejected, converged = history
            return (
                jnp.concatenate((initial[None, ...], trajectory), axis=0),
                accepted,
                rejected,
                converged,
            )
        return jnp.concatenate((initial[None, ...], history), axis=0)

    return kernel


def _crossing_periods(xi: np.ndarray, dt: float):
    indices = np.flatnonzero(xi[1:] * xi[:-1] <= 0)
    if indices.size < 3:
        return np.empty(0)
    left = xi[indices]
    right = xi[indices + 1]
    fraction = np.divide(
        -left,
        right - left,
        out=np.zeros_like(left, dtype=float),
        where=np.abs(right - left) > 0.0,
    )
    crossing_times = (indices.astype(float) + fraction) * dt
    return np.diff(crossing_times[::2])


def _summarize(history: np.ndarray, labels: np.ndarray, dt: float):
    rows = []
    for i, label in enumerate(labels):
        r = history[:, 0, i]
        xi = history[:, 2, i]
        pphi = history[:, 4, i]
        mu = history[:, 5, i]
        pphi0 = max(abs(pphi[0]), 1.0e-30)
        mu0 = max(abs(mu[0]), 1.0e-30)
        periods = _crossing_periods(xi, dt)
        sign_changes = int(np.count_nonzero(xi[1:] * xi[:-1] <= 0))
        if sign_changes >= 2:
            observed = "trapped"
        elif sign_changes == 1:
            observed = "ambiguous"
        else:
            observed = "passing"
        rows.append(
            {
                "marker": i,
                "label": label,
                "expected_classification": label,
                "observed_classification": observed,
                "classification_match": observed == label,
                "r_min": float(np.min(r)),
                "r_max": float(np.max(r)),
                "banana_width": float(np.max(r) - np.min(r)),
                "xi_sign_changes": sign_changes,
                "bounce_period": (
                    float(np.mean(periods)) if len(periods) else math.nan
                ),
                "max_rel_pphi_error": float(
                    np.max(np.abs(pphi - pphi[0]) / pphi0)
                ),
                "max_rel_mu_error": float(np.max(np.abs(mu - mu[0]) / mu0)),
                "finite": bool(np.all(np.isfinite(history[:, :, i]))),
            }
        )
    return rows


def _git_metadata():
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    return commit, dirty


def _write_csv(path: Path, rows):
    fields = list(rows[0])
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_plots(
    outdir: Path,
    history: np.ndarray,
    labels: np.ndarray,
    dt: float,
    epsilon: float,
):
    import matplotlib.pyplot as plt

    times = np.arange(history.shape[0]) * dt
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    for i, label in enumerate(labels):
        name = f"{label} {i}"
        axes[0, 0].plot(times, history[:, 0, i], label=name)
        axes[0, 1].plot(history[:, 1, i], history[:, 0, i], label=name)
        axes[1, 0].plot(times, history[:, 2, i], label=name)
        axes[1, 1].plot(
            times,
            history[:, 4, i] - history[0, 4, i],
            label=name,
        )
    axes[0, 0].set(xlabel=r"$t/\tau_c$", ylabel=r"$r/a$", title="Radial orbit")
    axes[0, 1].set(xlabel=r"$\theta$", ylabel=r"$r/a$", title="Orbit topology")
    axes[1, 0].set(xlabel=r"$t/\tau_c$", ylabel=r"$\xi$", title="Trapped/passing pitch")
    axes[1, 1].set(
        xlabel=r"$t/\tau_c$",
        ylabel=r"$P_\phi-P_{\phi,0}$",
        title="Toroidal momentum drift",
    )
    for ax in axes.flat:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(outdir / "tokamak_orbit_trajectories.png", dpi=220)
    plt.close(fig)

    # Circular-tokamak embedding: R/R0 = 1 + epsilon*r*cos(theta),
    # Z/R0 = epsilon*r*sin(theta), with r normalized to the minor radius.
    radius = history[:, 0, :]
    theta = history[:, 1, :]
    major_radius = 1.0 + epsilon * radius * np.cos(theta)
    vertical = epsilon * radius * np.sin(theta)
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    for i, label in enumerate(labels):
        ax.plot(major_radius[:, i], vertical[:, i], label=f"{label} {i}")
        ax.plot(major_radius[0, i], vertical[0, i], "o", ms=4)
    ax.set(
        xlabel=r"$R/R_0$",
        ylabel=r"$Z/R_0$",
        title="Trapped/passing orbits in the poloidal R--Z plane",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(outdir / "tokamak_orbit_RZ.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), constrained_layout=True)
    for i, label in enumerate(labels):
        name = f"{label} {i}"
        axes[0].plot(
            times,
            (history[:, 4, i] - history[0, 4, i])
            / max(abs(history[0, 4, i]), 1.0e-30),
            label=name,
        )
        axes[1].plot(
            times,
            (history[:, 5, i] - history[0, 5, i])
            / max(abs(history[0, 5, i]), 1.0e-30),
            label=name,
        )
    axes[0].set(xlabel=r"$t/\tau_c$", ylabel=r"$\Delta P_\phi/P_{\phi,0}$")
    axes[1].set(xlabel=r"$t/\tau_c$", ylabel=r"$\Delta\mu/\mu_0$")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
    fig.savefig(outdir / "tokamak_orbit_invariants.png", dpi=220)
    plt.close(fig)


def run_trajectory(
    output_dir: Path,
    dt: float,
    final_time: float,
    n_particles: int,
    config,
    integrator: str,
):
    n_steps = int(round(final_time / dt))
    if n_steps < 1:
        raise ValueError("final_time must be at least one timestep")
    dt_actual = final_time / n_steps
    kin, labels, profiles, norm = make_case(n_particles, config)
    kernel = _make_trajectory_kernel(dt_actual, n_steps, integrator)
    # Compile and synchronize once before recording the case runtime. The
    # reported timing is execution time for the resolved trajectory, not JIT
    # compilation overhead.
    warmup_output = kernel(kin, profiles, norm)
    jax.block_until_ready(warmup_output)
    started = time.perf_counter()
    kernel_output = kernel(kin, profiles, norm)
    jax.block_until_ready(kernel_output)
    elapsed = time.perf_counter() - started
    adaptive_stats = None
    if integrator == "bogacki_shampine5_adaptive":
        history, accepted_history, rejected_history, convergence_history = kernel_output
        adaptive_stats = {
            "accepted_internal_steps": int(np.asarray(accepted_history).sum()),
            "rejected_internal_steps": int(np.asarray(rejected_history).sum()),
            "adaptive_converged": bool(np.asarray(convergence_history).all()),
        }
    else:
        history = kernel_output
    history_np = np.asarray(history)
    rows = _summarize(history_np, labels, dt_actual)
    for row in rows:
        row["runtime_seconds"] = elapsed
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tokamak_orbit_summary.csv", rows)
    np.savez_compressed(
        output_dir / "tokamak_orbit_trajectories.npz",
        time=np.arange(history_np.shape[0]) * dt_actual,
        labels=labels,
        history=history_np,
    )
    _write_plots(
        output_dir,
        history_np,
        labels,
        dt_actual,
        float(config["case"]["epsilon"]),
    )
    commit, dirty = _git_metadata()
    manifest = {
        "benchmark": "one_d.tokamak_orbits",
        "mode": "trajectory",
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "integrator": integrator,
        "dt": dt_actual,
        "final_time": final_time,
        "n_steps": n_steps,
        "n_particles": n_particles,
        "runtime_seconds": elapsed,
        "timing": {
            "compilation_excluded": True,
            "synchronization": "jax.block_until_ready",
        },
        "adaptive_controls": (
            {"atol": 1.0e-9, "rtol": 1.0e-9, **adaptive_stats}
            if adaptive_stats is not None
            else None
        ),
        "git_commit": commit,
        "git_dirty": dirty,
        "interpretation": (
            "analytical and internal-provenance orbit topology; "
            "no published numerical target yet"
        ),
        "outputs": [
            "tokamak_orbit_summary.csv",
            "tokamak_orbit_trajectories.npz",
            "tokamak_orbit_trajectories.png",
            "tokamak_orbit_RZ.png",
            "tokamak_orbit_invariants.png",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return rows, manifest


def _shard_case(n_particles: int, n_devices: int, config):
    kin, _labels, profiles, norm = make_case(n_particles, config)
    leaves, treedef = jax.tree_util.tree_flatten(kin)
    if any(leaf.shape[0] % n_devices for leaf in leaves):
        raise ValueError("total_particles must divide evenly across devices")
    local = leaves[0].shape[0] // n_devices
    sharded = [leaf.reshape((n_devices, local) + leaf.shape[1:]) for leaf in leaves]
    return jax.tree_util.tree_unflatten(treedef, sharded), profiles, norm


def run_scaling(
    output_dir: Path,
    devices: list[int],
    total_particles: int,
    repeats: int,
    config,
    integrator: str,
):
    if jax.default_backend() != "cpu":
        raise RuntimeError("--scaling requires JAX_PLATFORMS=cpu")
    available = len(jax.devices("cpu"))
    if any(count < 1 or count > available for count in devices):
        raise ValueError(f"requested devices {devices}, available CPU devices: {available}")
    if devices != sorted(set(devices)) or devices[0] != 1:
        raise ValueError("scaling devices must be unique, sorted, and begin with 1")
    if any(total_particles % count for count in devices):
        raise ValueError("every scaling device count must divide total_particles")
    case = config["case"]
    final_time = float(case["final_time"])
    dt = float(case["dt"])
    n_steps = int(round(final_time / dt))
    dt = final_time / n_steps
    rows = []
    for count in devices:
        kin, profiles, norm = _shard_case(total_particles, count, config)
        local_kernel = _make_trajectory_kernel(dt, n_steps, integrator)
        mapped = jax.pmap(
            local_kernel,
            in_axes=(0, None, None),
            axis_name="device",
            devices=jax.devices("cpu")[:count],
        )
        jax.block_until_ready(mapped(kin, profiles, norm))
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            jax.block_until_ready(mapped(kin, profiles, norm))
            samples.append(time.perf_counter() - started)
        rows.append({"devices": count, "seconds": float(np.median(samples))})
    baseline = rows[0]["seconds"]
    for row in rows:
        row["speedup"] = baseline / row["seconds"]
        row["ideal_speedup"] = row["devices"]
        row["efficiency"] = row["speedup"] / row["devices"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "tokamak_orbit_scaling.csv", rows)
    import matplotlib.pyplot as plt

    counts = np.asarray([row["devices"] for row in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(counts, [row["speedup"] for row in rows], "o-", label="measured")
    ax.plot(counts, counts, "--", label="ideal")
    ax.set(
        xlabel="logical CPU devices",
        ylabel="speedup",
        title="Tokamak-orbit CPU scaling",
    )
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "tokamak_orbit_scaling.png", dpi=220)
    plt.close(fig)
    commit, dirty = _git_metadata()
    manifest = {
        "benchmark": "one_d.tokamak_orbits",
        "mode": "strong_scaling",
        "backend": "cpu",
        "devices": devices,
        "available_cpu_devices": available,
        "platform": platform.platform(),
        "jax_version": jax.__version__,
        "x64_enabled": bool(jax.config.read("jax_enable_x64")),
        "total_particles": total_particles,
        "repeats": repeats,
        "integrator": integrator,
        "dt": dt,
        "final_time": final_time,
        "git_commit": commit,
        "git_dirty": dirty,
        "outputs": ["tokamak_orbit_scaling.csv", "tokamak_orbit_scaling.png"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return rows


def _load_config(path: Path):
    with path.open() as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("benchmark") != "one_d.tokamak_orbits":
        raise ValueError(f"invalid tokamak-orbit config: {path}")
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--dt", type=float)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--n-particles", type=int, default=2)
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--scaling-devices", type=int, nargs="+")
    parser.add_argument("--scaling-total-particles", type=int)
    parser.add_argument("--scaling-repeats", type=int)
    parser.add_argument(
        "--integrator",
        choices=(
            "rk4",
            "bogacki_shampine5",
            "bogacki_shampine5_adaptive",
        ),
    )
    args = parser.parse_args()
    config = _load_config(args.config)
    case = config["case"]
    selected_integrator = args.integrator or config["integrator"].get("baseline", "rk4")
    if args.scaling:
        scaling = config["scaling"]
        rows = run_scaling(
            args.output_dir,
            args.scaling_devices or [int(value) for value in scaling["devices"]],
            args.scaling_total_particles or int(scaling["total_particles"]),
            args.scaling_repeats or int(scaling["repeats"]),
            config,
            selected_integrator,
        )
        for row in rows:
            print(row)
    else:
        rows, manifest = run_trajectory(
            args.output_dir,
            args.dt if args.dt is not None else float(case["dt"]),
            args.final_time
            if args.final_time is not None
            else float(case["final_time"]),
            args.n_particles,
            config,
            selected_integrator,
        )
        print(f"completed {manifest['n_steps']} steps in {manifest['runtime_seconds']:.3f} s")
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
