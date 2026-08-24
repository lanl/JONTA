"""Reproduce McDevitt et al. (2019) Appendix Fig. B3(a)."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from benchmarks.test_large_angle_avalanche import (
    load_reference_data,
    run_growth_replicates,
)
from core.configuration import load_config
from core.constants import ME_C2_EV

jax.config.update("jax_enable_x64", True)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    default_config = Path(__file__).with_name("config.yaml")
    parser.add_argument("--config", type=Path, default=default_config)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--markers", type=int, default=None)
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--total-time", type=float, default=None)
    parser.add_argument("--dt", type=float, default=None)
    parser.add_argument("--large-angle-dt", type=float, default=None)
    parser.add_argument("--sample-dt", type=float, default=None)
    parser.add_argument("--fit-start-fraction", type=float, default=None)
    parser.add_argument("--fields", type=float, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--mode", choices=("serial", "parallel"), default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="skip fields already present in output CSV")
    return parser.parse_args()


def _reference_rows(path):
    return [row for row in load_reference_data(Path(path)) if row["figure"] == "B3a"]


def _reference_lookup(path):
    return {float(row["e_over_ec"]): float(row["value"]) for row in _reference_rows(path)}


def _write_tables(output, rows, histories):
    if rows:
        with (output / "b3a_growth.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
    if histories:
        with (output / "b3a_histories.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(histories[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(histories)


def _normalise_growth_row(row):
    result = dict(row)
    for key in (
        "E_over_Ec", "JONTA_growth", "JONTA_SEM", "JONTA_TOTAL_SIGMA", "fit_window_sigma", "paper_growth", "difference",
        "fit_R2_mean", "max_event_probability", "total_time_tau_c", "dt_tau_c",
        "large_angle_dt_tau_c", "sample_dt_tau_c", "fit_start_fraction",
        "small_angle_p_min", "runtime_seconds",
    ):
        if key in result:
            result[key] = float(result[key])
    for key in ("markers", "replicas", "seed_start", "devices", "small_angle_substeps", "small_angle_n_sa"):
        if key in result:
            result[key] = int(result[key])
    return result


def _write_growth_plot(output, rows, reference_rows):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(
        [row["e_over_ec"] for row in reference_rows],
        [row["value"] for row in reference_rows],
        "o-",
        label="McDevitt 2019 Fig. B3(a)",
    )
    if rows:
        ax.errorbar(
            [row["E_over_Ec"] for row in rows],
            [row["JONTA_growth"] for row in rows],
            yerr=[row.get("JONTA_TOTAL_SIGMA", row["JONTA_SEM"]) for row in rows],
            fmt="s",
            capsize=3,
            label="JONTA (total 1-sigma)",
        )
    ax.axhline(0.0, color="0.3", linewidth=0.8)
    ax.set(xlabel=r"$E/E_c$", ylabel=r"$\gamma_{\rm av}\tau_c$", title="Slab avalanche growth")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "b3a_growth.png", dpi=240)
    plt.close(fig)


def _write_history_plot(output, histories):
    if not histories:
        return
    import matplotlib.pyplot as plt

    fields = sorted({float(row["E_over_Ec"]) for row in histories})
    ncols = 2
    nrows = (len(fields) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12.0, 3.8 * nrows), squeeze=False)
    for axis, field in zip(axes.flat, fields):
        field_rows = [row for row in histories if np.isclose(float(row["E_over_Ec"]), field)]
        positive_weights = np.asarray(
            [float(row["weight"]) for row in field_rows if float(row["weight"]) > 0.0]
        )
        lower = max(float(np.min(positive_weights)) * 0.5, 1.0e-6) if positive_weights.size else 1.0e-6
        upper = max(float(np.max(positive_weights)) * 2.0, lower * 10.0) if positive_weights.size else 10.0
        for seed in sorted({int(row["seed"]) for row in field_rows}):
            series = sorted((row for row in field_rows if int(row["seed"]) == seed), key=lambda row: float(row["time_tau_c"]))
            t = np.asarray([float(row["time_tau_c"]) for row in series])
            w = np.asarray([float(row["weight"]) for row in series])
            growth = float(series[0]["fit_growth"])
            intercept = float(series[0]["fit_intercept"])
            axis.plot(t, np.where(w > 0.0, w, np.nan), color="tab:blue", alpha=0.22, linewidth=0.8)
            fit_fraction = float(series[0].get("fit_start_fraction", 0.35))
            fit_start = t[min(int(np.floor(fit_fraction * len(t))), len(t) - 1)]
            fit_mask = t >= fit_start
            fit_y = np.exp(intercept + growth * t[fit_mask])
            fit_valid = np.isfinite(fit_y) & (fit_y >= lower) & (fit_y <= upper)
            axis.plot(t[fit_mask][fit_valid], fit_y[fit_valid], color="tab:red", alpha=0.65, linewidth=1.0)
        axis.set_yscale("log")
        axis.set_ylim(lower, upper)
        axis.set_title(fr"$E/E_c={field:g}$")
        axis.set_xlabel(r"$t/\tau_c$")
        axis.set_ylabel(r"$n_{\rm RE}/n_{\rm RE,0}$")
        axis.grid(alpha=0.25, which="both")
    for axis in axes.flat[len(fields):]:
        axis.remove()
    handles = [
        plt.Line2D([], [], color="tab:blue", alpha=0.6, label="replica history"),
        plt.Line2D([], [], color="tab:red", label="exponential fit"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output / "b3a_histories.png", dpi=220)
    plt.close(fig)


def main():
    args = _args()
    config = load_config(args.config)
    mode = config.execution.mode if args.mode is None else args.mode
    devices = (config.execution.n_devices or 1) if args.devices is None else args.devices
    if mode == "parallel" and devices < 2:
        raise ValueError("parallel mode requires at least two CPU devices")
    output = args.output_dir or Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    markers = config.particles.markers if args.markers is None else args.markers
    replicas = config.benchmark.replicas if args.replicas is None else args.replicas
    total_time = config.benchmark.total_time if args.total_time is None else args.total_time
    dt = config.timesteps.dt_particle if args.dt is None else args.dt
    large_angle_dt = (
        config.timesteps.large_angle_every * dt
        if args.large_angle_dt is None
        else args.large_angle_dt
    )
    sample_dt = (
        config.diagnostics.sample_every * dt
        if args.sample_dt is None
        else args.sample_dt
    )
    fit_start_fraction = (
        config.diagnostics.fit_start_fraction
        if args.fit_start_fraction is None
        else args.fit_start_fraction
    )
    configured_burn_fraction = config.benchmark.burn_time / total_time
    if args.fit_start_fraction is None and not np.isclose(
        fit_start_fraction, configured_burn_fraction
    ):
        raise ValueError(
            "benchmark.burn_time/total_time must match diagnostics.fit_start_fraction"
        )
    fields = tuple(config.field.e_scan) if args.fields is None else tuple(args.fields)
    vte_over_c = float(np.sqrt(2.0 * config.background.te_ev / ME_C2_EV))
    p_min = float(config.collisions.small_angle.p_min)
    reference_path = config.reference.data_file
    if reference_path is None:
        raise ValueError("reference.data_file is required for B3(a) benchmark")
    reference_rows = _reference_rows(reference_path)
    reference = _reference_lookup(reference_path)
    seed_start = config.particles.seed if args.seed_start is None else args.seed_start
    seeds = tuple(seed_start + i for i in range(replicas))

    rows = []
    histories = []
    if args.resume and (output / "b3a_growth.csv").exists():
        with (output / "b3a_growth.csv").open(newline="") as stream:
            rows = [_normalise_growth_row(row) for row in csv.DictReader(stream)]
        if (output / "b3a_histories.csv").exists():
            with (output / "b3a_histories.csv").open(newline="") as stream:
                histories = list(csv.DictReader(stream))
        completed_fields = {float(row["E_over_Ec"]) for row in rows}
    else:
        completed_fields = set()
    for field in fields:
        if args.resume and any(np.isclose(field, completed) for completed in completed_fields):
            print(f"E/Ec={field:g}: already present; skipped")
            continue
        start = time.perf_counter()
        result = run_growth_replicates(
            field,
            alpha=config.geometry.alpha_syn,
            zeff=config.background.zeff,
            coulog0=config.background.coulomb_log,
            gamma_min=config.collisions.large_angle.gamma_min,
            conservative=config.collisions.large_angle.conservative,
            large_angle_enabled=config.collisions.large_angle.enabled,
            n_markers=markers,
            total_time=total_time,
            dt=dt,
            large_angle_dt=large_angle_dt,
            sample_dt=sample_dt,
            seeds=seeds,
            fit_start_fraction=fit_start_fraction,
            vte_over_c=vte_over_c,
            p_min=p_min,
            n_sa=config.collisions.small_angle.n_sa,
            energy_scattering=config.collisions.small_angle.energy_scattering,
            initial_gamma_range=config.particles.gamma_range,
            initial_xi_range=config.particles.xi_range,
            execution_mode=mode,
            n_devices=devices,
        )
        runtime = time.perf_counter() - start
        paper_value = min(reference.items(), key=lambda pair: abs(pair[0] - field))[1]
        robust_sigma = float(np.sqrt(result["std"] ** 2 + result["fit_window_std"] ** 2))
        rows.append({
            "E_over_Ec": field,
            "JONTA_growth": result["growth"],
            "JONTA_SEM": result["sem"],
            "JONTA_TOTAL_SIGMA": robust_sigma,
            "fit_window_sigma": result["fit_window_std"],
            "paper_growth": paper_value,
            "difference": result["growth"] - paper_value,
            "fit_R2_mean": result["r2_mean"],
            "max_event_probability": result["max_q"],
            "markers": markers,
            "replicas": replicas,
            "total_time_tau_c": total_time,
            "dt_tau_c": dt,
            "large_angle_dt_tau_c": large_angle_dt,
            "sample_dt_tau_c": sample_dt,
            "fit_start_fraction": fit_start_fraction,
            "seed_start": seed_start,
            "mode": mode,
            "devices": devices,
            "small_angle_substeps": result["small_angle_substeps"],
            "small_angle_n_sa": result["small_angle_n_sa"],
            "small_angle_p_min": result["small_angle_p_min"],
            "runtime_seconds": runtime,
        })
        for seed, t, weight, growth, intercept, r2, fit_window_std in result["histories"]:
            for ti, wi in zip(t, weight):
                histories.append({
                    "E_over_Ec": field,
                    "seed": seed,
                    "time_tau_c": ti,
                    "weight": wi,
                    "fit_growth": growth,
                    "fit_intercept": intercept,
                    "fit_R2": r2,
                    "fit_window_std": fit_window_std,
                    "fit_start_fraction": fit_start_fraction,
                })
        _write_tables(output, rows, histories)
        _write_growth_plot(output, rows, reference_rows)
        _write_history_plot(output, histories)
        completed_fields.add(float(field))
        print(f"E/Ec={field:g}: growth={result['growth']:.8g} +/- {result['sem']:.3g}; runtime={runtime:.3f}s")

    _write_tables(output, rows, histories)
    _write_growth_plot(output, rows, reference_rows)
    _write_history_plot(output, histories)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = {
        "benchmark": "slab.avalanche_decay.b3a",
        "reference": "McDevitt et al. 2019 PPCF 61 054008, Fig. B3(a)",
        "parameters": {
            "alpha_syn": config.geometry.alpha_syn,
            "zeff": config.background.zeff,
            "coulomb_log": config.background.coulomb_log,
            "vte_over_c": vte_over_c,
            "gamma_min": config.collisions.large_angle.gamma_min,
            "large_angle_enabled": config.collisions.large_angle.enabled,
            "source_only_moller": not config.collisions.large_angle.conservative,
            "energy_scattering": config.collisions.small_angle.energy_scattering,
            "fields": list(fields),
            "markers": markers,
            "replicas": replicas,
            "total_time_tau_c": total_time,
            "dt_tau_c": dt,
            "large_angle_dt_tau_c": large_angle_dt,
            "sample_dt_tau_c": sample_dt,
            "fit_start_fraction": fit_start_fraction,
            "seed_start": seed_start,
        },
        "backend": {
            "jax": jax.__version__,
            "platform": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "precision": "float64",
        },
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "git_commit": commit},
        "resolution": {
            "small_angle_n_sa": rows[0]["small_angle_n_sa"],
            "small_angle_substeps": rows[0]["small_angle_substeps"],
            "small_angle_p_min": rows[0]["small_angle_p_min"],
            "max_event_probability": max(row["max_event_probability"] for row in rows),
        },
        "uncertainty": {
            "stochastic": "replica sample SEM",
            "fit_window": "sample RMS across fit-start offsets; not divided by sqrt(replicas)",
            "reported_plot_error": "sqrt(replica SD^2 + fit-window RMS^2)",
        },
        "outputs": ["b3a_growth.csv", "b3a_histories.csv", "b3a_growth.png", "b3a_histories.png"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
