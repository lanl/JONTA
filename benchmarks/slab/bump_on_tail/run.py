"""Reproducible Guo et al. (2017) bump-on-tail benchmark.

The evolution kernel is the production 0D particle pusher from
``benchmarks.test_runaway_vortex``.  This module owns only benchmark
orchestration: warm-up/timing, CPU sharding, convergence scans, and output.
"""

from __future__ import annotations

import argparse
import csv
import json
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

from benchmarks.test_runaway_vortex import (
    DEFAULT_SAMPLE_EVERY,
    DEFAULT_VORTEX_BINS,
    DEFAULT_VORTEX_BURN_TIME,
    DEFAULT_VORTEX_DT,
    DEFAULT_VORTEX_TOTAL_TIME,
    _make_vortex_kernel,
    _write_particle_vortex_results,
    published_bump_points,
    run_particle_vortex_case,
)
from parallel import merge_particle_partitions, partition_particles

jax.config.update("jax_enable_x64", True)

DEFAULT_FIELDS = (2.0, 2.25, 2.5)
CONVERGENCE_MARKERS = (1024, 2048, 4096)
CONVERGENCE_DT = (2.0e-3, 4.0e-3, 8.0e-3)
CONVERGENCE_SEEDS = (41, 43, 47)


def _make_execution_kernel(mode: str, n_devices: int, *, n_bins: int, phase_bins):
    """Build serial or particle-sharded execution around the production kernel."""

    if mode == "serial":
        if n_devices != 1:
            raise ValueError("serial execution requires n_devices=1")
        return _make_vortex_kernel(n_bins=n_bins, phase_bins=phase_bins)
    if mode != "parallel":
        raise ValueError(f"unknown execution mode: {mode}")

    devices = tuple(jax.devices("cpu"))
    if n_devices < 2 or n_devices > len(devices):
        raise RuntimeError(
            f"requested {n_devices} CPU devices, available={len(devices)}"
        )
    local = _make_vortex_kernel(n_bins=n_bins, phase_bins=phase_bins)
    mapped = jax.pmap(
        local,
        in_axes=(0, None, None, None, None, None, None, 0),
        devices=devices[:n_devices],
    )

    def kernel(particles, e_over_ec, dt, n_steps, burn_steps, sample_every, phase_sample_every, base_key):
        partitioned = partition_particles(particles, n_devices)
        keys = jax.random.split(base_key, n_devices)
        final, histogram, phase_hist, n_samples, n_phase_samples = mapped(
            partitioned,
            e_over_ec,
            dt,
            n_steps,
            burn_steps,
            sample_every,
            phase_sample_every,
            keys,
        )
        return (
            merge_particle_partitions(final),
            jnp.sum(histogram, axis=0),
            jnp.sum(phase_hist, axis=0),
            jnp.sum(n_samples),
            jnp.sum(n_phase_samples),
        )

    return kernel


def _run_case(kernel, e_over_ec: float, *, n_markers: int, seed: int, dt: float,
              total_time: float, burn_time: float, phase_sample_every: int):
    """Run one case and add a synchronized post-warm-up wall time."""

    kwargs = dict(
        n_markers=n_markers,
        seed=seed,
        dt=dt,
        total_time=total_time,
        burn_time=burn_time,
        sample_every=DEFAULT_SAMPLE_EVERY,
        phase_sample_every=phase_sample_every,
    )
    run_particle_vortex_case(kernel, e_over_ec, **kwargs)  # compile/warm-up
    start = time.perf_counter()
    result = run_particle_vortex_case(kernel, e_over_ec, **kwargs)
    result["runtime_seconds"] = time.perf_counter() - start
    result["seed"] = seed
    return result


def _write_runtime_plot(path: Path, rows):
    import matplotlib.pyplot as plt

    fields = [row["e_over_ec"] for row in rows]
    runtime = [row["runtime_seconds"] for row in rows]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar([f"{value:g}" for value in fields], runtime, color="#4472c4")
    ax.set_xlabel(r"$E/E_c$")
    ax.set_ylabel("wall time after JIT warm-up [s]")
    ax.set_title("Bump-on-tail runtime")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _write_scaling(path_csv: Path, path_png: Path, rows):
    with path_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("mode", "n_devices", "markers", "e_over_ec", "seconds", "speedup"))
        writer.writeheader()
        serial = rows[0]["seconds"]
        for row in rows:
            row["speedup"] = serial / row["seconds"]
            writer.writerow(row)

    import matplotlib.pyplot as plt

    devices = np.asarray([row["n_devices"] for row in rows], dtype=float)
    seconds = np.asarray([row["seconds"] for row in rows], dtype=float)
    speedup = seconds[0] / seconds
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].loglog(devices, seconds, "o-", label="measured")
    axes[0].loglog(devices, seconds[0] / devices, "--", label="ideal $1/N$")
    axes[0].set(xlabel="CPU devices", ylabel="wall time [s]", title="Bump-on-tail scaling")
    axes[1].plot(devices, speedup, "o-", label="measured")
    axes[1].plot(devices, devices, "--", label="ideal")
    axes[1].set(xlabel="CPU devices", ylabel="speedup", title="Particle-sharded speedup")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def _write_convergence(path_csv: Path, path_png: Path, rows):
    fields = ("study", "parameter", "seed", "e_over_ec", "n_markers", "dt_tau_c", "total_time_tau_c", "burn_time_tau_c", "has_bump", "p_bump", "published_p_bump", "relative_error", "runtime_seconds")
    with path_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), constrained_layout=True)
    for axis, study, xkey, xlabel in (
        (axes[0], "markers", "n_markers", "markers"),
        (axes[1], "timestep", "dt_tau_c", r"$\Delta t/\tau_c$"),
        (axes[2], "seeds", "seed", "seed"),
    ):
        for field in (2.25, 2.5):
            subset = [r for r in rows if r["study"] == study and r["e_over_ec"] == field and np.isfinite(r["p_bump"])]
            subset.sort(key=lambda r: r[xkey])
            if subset:
                axis.plot([r[xkey] for r in subset], [r["p_bump"] for r in subset], "o-", label=f"E/Ec={field:g}")
                ref = published_bump_points()[field]
                axis.axhline(ref, color="k", linestyle=":", alpha=0.35)
        axis.set(xlabel=xlabel, ylabel=r"bump momentum $p_b/(m_ec)$", title=study.capitalize())
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
    fig.savefig(path_png, dpi=200)
    plt.close(fig)


def _manifest(path: Path, args, outputs, *, interpretation):
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    payload = {
        "benchmark": "slab.bump_on_tail",
        "status": "reproducibility_record",
        "interpretation": interpretation,
        "command_arguments": vars(args),
        "backend": {"platform": jax.default_backend(), "devices": [str(d) for d in jax.devices()], "precision": "float64"},
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "jax": getattr(jax, "__version__", "unknown"), "git_commit": commit},
        "timing": {"compilation_excluded": True, "synchronization": "host conversion of JAX results", "runtime_field": "runtime_seconds"},
        "outputs": outputs,
        "reference": "Guo, McDevitt & Tang, PPCF 59, 044003 (2017), Figs. 9-12",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--fields", type=float, nargs="+", default=list(DEFAULT_FIELDS))
    parser.add_argument("--markers", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--dt", type=float, default=DEFAULT_VORTEX_DT)
    parser.add_argument("--total-time", type=float, default=DEFAULT_VORTEX_TOTAL_TIME)
    parser.add_argument("--burn-time", type=float, default=DEFAULT_VORTEX_BURN_TIME)
    parser.add_argument("--bins", type=int, default=DEFAULT_VORTEX_BINS)
    parser.add_argument("--phase-bins", type=int, nargs=2, default=(120, 72))
    parser.add_argument("--phase-sample-every", type=int, default=25)
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--scaling-devices", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--scaling-markers", type=int, default=4096)
    parser.add_argument("--convergence", action="store_true")
    return parser.parse_args()


def main():
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    phase_bins = tuple(args.phase_bins)

    if args.scaling:
        rows = []
        for n_devices in args.scaling_devices:
            mode = "serial" if n_devices == 1 else "parallel"
            kernel = _make_execution_kernel(mode, n_devices, n_bins=args.bins, phase_bins=(1, 1))
            result = _run_case(kernel, 2.25, n_markers=args.scaling_markers, seed=args.seed, dt=args.dt, total_time=args.total_time, burn_time=args.burn_time, phase_sample_every=args.phase_sample_every)
            rows.append({"mode": mode, "n_devices": n_devices, "markers": args.scaling_markers, "e_over_ec": 2.25, "seconds": result["runtime_seconds"]})
            print(f"{mode:8s} devices={n_devices}: {result['runtime_seconds']:.4f} s")
        _write_scaling(args.output_dir / "scalability.csv", args.output_dir / "scalability.png", rows)
        _manifest(args.output_dir / "manifest.json", args, {"csv": "scalability.csv", "plot": "scalability.png"}, interpretation="CPU particle-sharding timing; physics result is the same fixed-capacity kernel.")
        return

    if args.convergence:
        rows = []
        for markers in CONVERGENCE_MARKERS:
            for field in (2.25, 2.5):
                result = _run_case(_make_execution_kernel("serial", 1, n_bins=args.bins, phase_bins=(1, 1)), field, n_markers=markers, seed=args.seed, dt=args.dt, total_time=min(args.total_time, 35.0), burn_time=min(args.burn_time, 17.5), phase_sample_every=args.phase_sample_every)
                rows.append(_row("markers", markers, result))
        for dt in CONVERGENCE_DT:
            for field in (2.25, 2.5):
                result = _run_case(_make_execution_kernel("serial", 1, n_bins=args.bins, phase_bins=(1, 1)), field, n_markers=2048, seed=args.seed, dt=dt, total_time=min(args.total_time, 35.0), burn_time=min(args.burn_time, 17.5), phase_sample_every=args.phase_sample_every)
                rows.append(_row("timestep", dt, result))
        for seed in CONVERGENCE_SEEDS:
            for field in (2.25, 2.5):
                result = _run_case(_make_execution_kernel("serial", 1, n_bins=args.bins, phase_bins=(1, 1)), field, n_markers=2048, seed=seed, dt=args.dt, total_time=min(args.total_time, 35.0), burn_time=min(args.burn_time, 17.5), phase_sample_every=args.phase_sample_every)
                rows.append(_row("seeds", seed, result))
        _write_convergence(args.output_dir / "convergence.csv", args.output_dir / "convergence.png", rows)
        _manifest(args.output_dir / "manifest.json", args, {"csv": "convergence.csv", "plot": "convergence.png"}, interpretation="Resolution and independent-seed study around the Guo bump diagnostic; not a universal tolerance claim.")
        return

    kernel = _make_execution_kernel(args.mode, args.devices, n_bins=args.bins, phase_bins=phase_bins)
    results = [_run_case(kernel, field, n_markers=args.markers, seed=args.seed + i, dt=args.dt, total_time=args.total_time, burn_time=args.burn_time, phase_sample_every=args.phase_sample_every) for i, field in enumerate(args.fields)]
    summary = _write_particle_vortex_results(results, args.output_dir)
    _write_runtime_plot(args.output_dir / "runtime.png", results)
    _manifest(args.output_dir / "manifest.json", args, {"csv": summary.name, "distribution_plot": "runaway_vortex_energy_distribution.png", "bump_plot": "runaway_vortex_bump_comparison.png", "phase_plot": "runaway_vortex_phase_space_flux.png", "runtime_plot": "runtime.png"}, interpretation="Particle Monte-Carlo reproduction record using the production linearized small-angle operator, electric acceleration, and synchrotron reaction.")
    for result in results:
        ref = published_bump_points().get(result["e_over_ec"], np.nan)
        value = f"p_b={result['p_bump']:.5f}, published={ref:.8g}" if result["has_bump"] else "no pitch-integrated bump"
        print(f"E/Ec={result['e_over_ec']:.2f}: {value}; runtime={result['runtime_seconds']:.3f} s")


def _row(study, parameter, result):
    ref = published_bump_points().get(result["e_over_ec"], np.nan)
    return {"study": study, "parameter": parameter, "seed": result.get("seed", ""), "e_over_ec": result["e_over_ec"], "n_markers": result["n_markers"], "dt_tau_c": result["dt_tau_c"], "total_time_tau_c": result["total_time_tau_c"], "burn_time_tau_c": result["burn_time_tau_c"], "has_bump": result["has_bump"], "p_bump": result["p_bump"], "published_p_bump": ref, "relative_error": (result["p_bump"] - ref) / ref if result["has_bump"] and np.isfinite(ref) else np.nan, "runtime_seconds": result["runtime_seconds"]}


if __name__ == "__main__":
    main()
