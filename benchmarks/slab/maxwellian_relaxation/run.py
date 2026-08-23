"""Reproducible slab Maxwellian-relaxation benchmark.

The physical kernel is shared with the small-angle collision implementation;
this driver adds execution-mode selection, timing, and scalability output.
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
import numpy as np

from benchmarks.slab.maxwellian_relaxation.model import (
    DEFAULT_DT_THERMAL,
    DEFAULT_RELAXATION_THERMAL,
    DEFAULT_TEMPERATURES_EV,
    INITIAL_DISTRIBUTIONS,
    _make_relaxation_kernel,
    _write_distribution_plot,
    _write_runtime_plot,
    _write_summary_plot,
    run_temperature_case,
)
from parallel import merge_particle_partitions, partition_particles

jax.config.update("jax_enable_x64", True)

CONVERGENCE_MARKERS = (512, 1024, 2048, 4096)
CONVERGENCE_DT_THERMAL = (2.5e-3, 5.0e-3, 1.0e-2)
CONVERGENCE_SEEDS = (41, 43, 47)


def _make_execution_kernel(mode: str, n_devices: int, seed: int):
    """Return a kernel with the requested serial or CPU-pmap execution."""

    if mode == "serial":
        if n_devices != 1:
            raise ValueError("serial execution requires n_devices=1")
        return _make_relaxation_kernel(jax.random.key(seed))

    if mode != "parallel":
        raise ValueError(f"unknown execution mode: {mode}")

    devices = tuple(jax.devices("cpu"))
    if n_devices < 2:
        raise ValueError("parallel execution requires at least two CPU devices")
    if n_devices > len(devices):
        raise RuntimeError(
            f"requested {n_devices} CPU devices, but only {len(devices)} are available"
        )

    local_kernel = _make_relaxation_kernel(jax.random.key(seed))
    mapped = jax.pmap(
        local_kernel,
        in_axes=(0, None, None, None, None),
        devices=devices[:n_devices],
    )

    def kernel(particles, background, dt, n_steps, config):
        partitioned = partition_particles(particles, n_devices)
        evolved = mapped(partitioned, background, dt, n_steps, config)
        return merge_particle_partitions(evolved)

    return kernel


def _write_scaling_csv(path: Path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("mode", "n_devices", "markers", "temperature_ev", "seconds"),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_scaling_plot(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    rows = sorted(rows, key=lambda row: row["n_devices"])
    devices = np.asarray([row["n_devices"] for row in rows], dtype=float)
    seconds = np.asarray([row["seconds"] for row in rows], dtype=float)
    serial_seconds = seconds[0]
    speedup = serial_seconds / seconds
    ideal = devices

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    axes[0].loglog(devices, seconds, "o-", label="measured")
    axes[0].loglog(devices, serial_seconds / ideal, "--", label="ideal $1/N$")
    axes[0].set_xlabel("CPU devices")
    axes[0].set_ylabel("wall time [s]")
    axes[0].set_title("Maxwellian relaxation scaling")
    axes[0].grid(True, which="both", alpha=0.25)
    axes[0].legend()

    axes[1].plot(devices, speedup, "o-", label="measured")
    axes[1].plot(devices, ideal, "--", label="ideal")
    axes[1].set_xlabel("CPU devices")
    axes[1].set_ylabel("speedup")
    axes[1].set_title("Parallel speedup")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _write_convergence_plot(path: Path, rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), constrained_layout=True)
    for kind in INITIAL_DISTRIBUTIONS:
        marker_rows = [
            row
            for row in rows
            if row["study"] == "markers" and row["initial_distribution"] == kind
        ]
        marker_groups = {}
        for row in marker_rows:
            marker_groups.setdefault(row["n_markers"], []).append(row["final_ks_energy"])
        marker_x = sorted(marker_groups)
        axes[0].loglog(
            marker_x,
            [np.median(marker_groups[x]) for x in marker_x],
            "o-",
            label=kind,
        )

        dt_rows = [
            row
            for row in rows
            if row["study"] == "timestep" and row["initial_distribution"] == kind
        ]
        dt_groups = {}
        for row in dt_rows:
            dt_groups.setdefault(row["dt_tau_c"], []).append(row["final_ks_energy"])
        dt_x = sorted(dt_groups)
        axes[1].loglog(
            dt_x,
            [np.median(dt_groups[x]) for x in dt_x],
            "o-",
            label=kind,
        )

        seed_rows = [
            row
            for row in rows
            if row["study"] == "seeds" and row["initial_distribution"] == kind
        ]
        axes[2].plot(
            [row["seed"] for row in seed_rows],
            [row["final_ks_energy"] for row in seed_rows],
            "o-",
            label=kind,
        )

    axes[0].set_xlabel("markers per initial distribution")
    axes[0].set_ylabel("final energy KS distance")
    axes[0].set_title("Marker convergence")
    axes[1].set_xlabel(r"collision timestep $\Delta t/\tau_{\rm th}$")
    axes[1].set_ylabel("final energy KS distance")
    axes[1].set_title("Timestep convergence")
    axes[2].set_xlabel("seed")
    axes[2].set_ylabel("final energy KS distance")
    axes[2].set_title("Independent-seed spread")
    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("Maxwellian-relaxation convergence")
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def _run_warmed_case(*, temperature_ev, markers, seed, dt_thermal, relaxation_thermal):
    """Run one serial case after excluding compilation from its timing."""

    kernel = _make_execution_kernel("serial", 1, seed)
    run_temperature_case(
        kernel,
        temperature_ev,
        n_per_distribution=markers,
        seed=seed,
        dt_thermal=dt_thermal,
        relaxation_thermal=relaxation_thermal,
    )
    return run_temperature_case(
        kernel,
        temperature_ev,
        n_per_distribution=markers,
        seed=seed,
        dt_thermal=dt_thermal,
        relaxation_thermal=relaxation_thermal,
    )


def run_convergence(*, temperature_ev, relaxation_thermal):
    """Run marker, timestep, and independent-seed convergence studies."""

    rows = []
    for markers in CONVERGENCE_MARKERS:
        for row in _run_warmed_case(
            temperature_ev=temperature_ev,
            markers=markers,
            seed=CONVERGENCE_SEEDS[0],
            dt_thermal=DEFAULT_DT_THERMAL,
            relaxation_thermal=relaxation_thermal,
        ):
            rows.append({"study": "markers", "parameter": markers, "seed": CONVERGENCE_SEEDS[0], **row})
    for dt_thermal in CONVERGENCE_DT_THERMAL:
        for row in _run_warmed_case(
            temperature_ev=temperature_ev,
            markers=CONVERGENCE_MARKERS[-1],
            seed=CONVERGENCE_SEEDS[0],
            dt_thermal=dt_thermal,
            relaxation_thermal=relaxation_thermal,
        ):
            rows.append({"study": "timestep", "parameter": dt_thermal, "seed": CONVERGENCE_SEEDS[0], **row})
    for seed in CONVERGENCE_SEEDS:
        for row in _run_warmed_case(
            temperature_ev=temperature_ev,
            markers=CONVERGENCE_MARKERS[-1],
            seed=seed,
            dt_thermal=DEFAULT_DT_THERMAL,
            relaxation_thermal=relaxation_thermal,
        ):
            rows.append({"study": "seeds", "parameter": seed, "seed": seed, **row})
    return rows


def run_scalability(
    device_counts,
    *,
    temperature_ev: float,
    markers: int,
    seed: int,
    dt_thermal: float,
    relaxation_thermal: float,
):
    """Measure steady-state execution after one compilation warm-up."""

    device_counts = tuple(int(value) for value in device_counts)
    if not device_counts or device_counts[0] != 1 or len(set(device_counts)) != len(device_counts):
        raise ValueError("--scaling-devices must contain unique counts and start with 1")

    rows = []
    for n_devices in device_counts:
        mode = "serial" if n_devices == 1 else "parallel"
        kernel = _make_execution_kernel(mode, n_devices, seed)

        # Compile and warm the exact kernel before measuring execution.
        run_temperature_case(
            kernel,
            temperature_ev,
            n_per_distribution=markers,
            seed=seed,
            dt_thermal=dt_thermal,
            relaxation_thermal=relaxation_thermal,
        )
        start = time.perf_counter()
        run_temperature_case(
            kernel,
            temperature_ev,
            n_per_distribution=markers,
            seed=seed,
            dt_thermal=dt_thermal,
            relaxation_thermal=relaxation_thermal,
        )
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "mode": mode,
                "n_devices": int(n_devices),
                "markers": int(markers * len(INITIAL_DISTRIBUTIONS)),
                "temperature_ev": float(temperature_ev),
                "seconds": float(elapsed),
            }
        )
        print(f"{mode:8s} devices={n_devices}: {elapsed:.4f} s")
    return rows


def _write_manifest(path: Path, args, *, outputs):
    """Record exact CLI/backend provenance for reproducibility."""

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    try:
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        dirty = None
    payload = {
        "benchmark": "slab.maxwellian_relaxation",
        "status": "reproducibility_record",
        "interpretation": "Analytical-equilibrium comparison; not paper-level acceptance evidence.",
        "command_arguments": vars(args),
        "backend": {
            "platform": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "precision": "float64" if jax.config.read("jax_enable_x64") else "float32",
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "jax": getattr(jax, "__version__", "unknown"),
            "git_commit": commit,
            "git_dirty": dirty,
        },
        "timing": {
            "per_temperature_case": True,
            "compilation_excluded": True,
            "synchronization": "jax.block_until_ready",
            "runtime_field": "runtime_seconds",
        },
        "outputs": outputs,
        "configuration_source": "CLI arguments; typed YAML integration pending",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("serial", "parallel"), default="serial")
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--temperatures", type=float, nargs="+", default=list(DEFAULT_TEMPERATURES_EV))
    parser.add_argument("--markers", type=int, default=4096, help="markers per initial distribution")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--dt-thermal", type=float, default=DEFAULT_DT_THERMAL)
    parser.add_argument("--relaxation-thermal", type=float, default=DEFAULT_RELAXATION_THERMAL)
    parser.add_argument("--scaling", action="store_true")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--scaling-devices", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--scaling-temperature", type=float, default=1.0e3)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.scaling:
        scaling = run_scalability(
            args.scaling_devices,
            temperature_ev=args.scaling_temperature,
            markers=args.markers,
            seed=args.seed,
            dt_thermal=args.dt_thermal,
            relaxation_thermal=args.relaxation_thermal,
        )
        scaling_csv = args.output_dir / "scalability.csv"
        scaling_plot = args.output_dir / "scalability.png"
        _write_scaling_csv(scaling_csv, scaling)
        _write_scaling_plot(scaling_plot, scaling)
        _write_manifest(
            args.output_dir / "manifest.json",
            args,
            outputs={"csv": str(scaling_csv.name), "plot": str(scaling_plot.name)},
        )
        print(f"CSV: {args.output_dir / 'scalability.csv'}")
        print(f"Plot: {args.output_dir / 'scalability.png'}")
        print(f"Manifest: {args.output_dir / 'manifest.json'}")
        return

    if args.convergence:
        convergence_rows = run_convergence(
            temperature_ev=args.scaling_temperature,
            relaxation_thermal=args.relaxation_thermal,
        )
        convergence_csv = args.output_dir / "convergence.csv"
        convergence_fields = (
            "study",
            "parameter",
            "seed",
            "temperature_ev",
            "initial_distribution",
            "n_markers",
            "dt_tau_c",
            "n_steps",
            "final_ks_energy",
            "final_mean_energy_over_target",
            "final_mean_xi",
            "final_mean_xi2",
            "runtime_seconds",
        )
        with convergence_csv.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=convergence_fields)
            writer.writeheader()
            writer.writerows(
                {field: row[field] for field in convergence_fields}
                for row in convergence_rows
            )
        convergence_plot = args.output_dir / "convergence.png"
        _write_convergence_plot(convergence_plot, convergence_rows)
        _write_manifest(
            args.output_dir / "manifest.json",
            args,
            outputs={"csv": convergence_csv.name, "plot": convergence_plot.name},
        )
        print(f"CSV: {convergence_csv}")
        print(f"Plot: {convergence_plot}")
        print(f"Manifest: {args.output_dir / 'manifest.json'}")
        return

    kernel = _make_execution_kernel(args.mode, args.devices, args.seed)
    if args.temperatures:
        run_temperature_case(
            kernel,
            float(args.temperatures[0]),
            n_per_distribution=args.markers,
            seed=args.seed,
            dt_thermal=args.dt_thermal,
            relaxation_thermal=args.relaxation_thermal,
        )
    rows = []
    for temperature_ev in args.temperatures:
        rows.extend(
            run_temperature_case(
                kernel,
                temperature_ev,
                n_per_distribution=args.markers,
                seed=args.seed,
                dt_thermal=args.dt_thermal,
                relaxation_thermal=args.relaxation_thermal,
            )
        )

    csv_path = args.output_dir / "maxwellian_relaxation.csv"
    fields = (
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
        "initial_mean_xi2",
        "final_mean_xi",
        "final_mean_xi2",
        "runtime_seconds",
    )
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    _write_distribution_plot(args.output_dir / "relaxation_distributions.png", rows)
    _write_summary_plot(args.output_dir / "relaxation_summary.png", rows)
    _write_runtime_plot(args.output_dir / "runtime.png", rows)
    _write_manifest(
        args.output_dir / "manifest.json",
        args,
        outputs={
            "csv": str(csv_path.name),
            "distribution_plot": "relaxation_distributions.png",
            "summary_plot": "relaxation_summary.png",
            "runtime_plot": "runtime.png",
        },
    )
    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"CSV: {csv_path}")
    print(f"Distribution plot: {args.output_dir / 'relaxation_distributions.png'}")
    print(f"Summary plot: {args.output_dir / 'relaxation_summary.png'}")
    print(f"Runtime plot: {args.output_dir / 'runtime.png'}")
    print(f"Manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
