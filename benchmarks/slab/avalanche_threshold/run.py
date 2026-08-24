"""Reproduce McDevitt et al. (2019) Fig. B3(b) avalanche thresholds."""

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

from benchmarks.test_large_angle_avalanche import mcdevitt_threshold_fit, run_growth_replicates
from core.configuration import load_config
from core.constants import ME_C2_EV

jax.config.update("jax_enable_x64", True)


def _args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--markers", type=int, default=None)
    parser.add_argument("--replicas", type=int, default=None)
    parser.add_argument("--total-time", type=float, default=None)
    parser.add_argument("--inv-alpha", type=float, nargs="+", default=None)
    parser.add_argument("--zeff", type=float, nargs="+", default=None)
    parser.add_argument("--seed-start", type=int, default=None)
    parser.add_argument("--mode", choices=("serial", "parallel"), default=None)
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reconcile",
        action="store_true",
        help="drop checkpoint rows outside the requested Zeff and 1/alpha grid",
    )
    return parser.parse_args()


def _read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _reference(path: Path):
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "zeff": float(row["zeff"]),
                    "inv_alpha": float(row["inv_alpha"]),
                    "value": float(row["eav_over_ec"]),
                }
            )
    return rows


def _threshold_sigma(values):
    values = sorted(values)
    (e0, g0, s0), (e1, g1, s1) = values
    dg = float(g1 - g0)
    if abs(dg) < 1.0e-30:
        return float("nan")
    de = float(e1 - e0)
    d0 = -de * float(g1) / (dg * dg)
    d1 = de * float(g0) / (dg * dg)
    return float(np.sqrt((d0 * s0) ** 2 + (d1 * s1) ** 2))


def _growth_kwargs(config, *, alpha, zeff, markers, total_time, seeds, mode, devices):
    dt = float(config.timesteps.dt_particle)
    return dict(
        alpha=alpha,
        zeff=zeff,
        coulog0=float(config.background.coulomb_log),
        gamma_min=float(config.collisions.large_angle.gamma_min),
        conservative=bool(config.collisions.large_angle.conservative),
        large_angle_enabled=bool(config.collisions.large_angle.enabled),
        n_markers=markers,
        total_time=total_time,
        dt=dt,
        large_angle_dt=dt * int(config.timesteps.large_angle_every),
        sample_dt=dt * int(config.diagnostics.sample_every),
        seeds=seeds,
        fit_start_fraction=float(config.diagnostics.fit_start_fraction),
        vte_over_c=float(np.sqrt(2.0 * config.background.te_ev / ME_C2_EV)),
        p_min=float(config.collisions.small_angle.p_min),
        n_sa=int(config.collisions.small_angle.n_sa),
        energy_scattering=bool(config.collisions.small_angle.energy_scattering),
        initial_gamma_range=config.particles.gamma_range,
        initial_xi_range=config.particles.xi_range,
        execution_mode=mode,
        n_devices=devices,
    )


def _history_rows(zeff, inv_alpha, alpha, iteration, field, result):
    rows = []
    for seed, times, weights, growth, intercept, r2, fit_window_std in result["histories"]:
        for time_tau_c, weight in zip(times, weights):
            rows.append(
                {
                    "Zeff": zeff,
                    "inv_alpha": inv_alpha,
                    "alpha": alpha,
                    "bracket_iteration": iteration,
                    "E_over_Ec": field,
                    "seed": seed,
                    "time_tau_c": time_tau_c,
                    "weight": weight,
                    "fit_growth": growth,
                    "fit_intercept": intercept,
                    "fit_R2": r2,
                    "fit_window_sigma": fit_window_std,
                }
            )
    return rows


def _solve_threshold(config, *, zeff, inv_alpha, markers, total_time, seeds, mode, devices):
    alpha = 1.0 / float(inv_alpha)
    estimate = float(mcdevitt_threshold_fit(alpha, zeff))
    span = max(0.08, 0.08 * estimate)
    sampled = []
    histories = []
    cache = {}
    start = time.perf_counter()

    for iteration in range(4):
        fields = (max(1.001, estimate - span), estimate + span)
        bracket = []
        for field in fields:
            key = round(float(field), 12)
            if key not in cache:
                result = run_growth_replicates(
                    field,
                    **_growth_kwargs(
                        config,
                        alpha=alpha,
                        zeff=zeff,
                        markers=markers,
                        total_time=total_time,
                        seeds=seeds,
                        mode=mode,
                        devices=devices,
                    ),
                )
                total_sigma = float(np.sqrt(result["sem"] ** 2 + result["fit_window_std"] ** 2))
                cache[key] = (result, total_sigma)
                histories.extend(_history_rows(zeff, inv_alpha, alpha, iteration, field, result))
                sampled.append(
                    {
                        "Zeff": zeff,
                        "inv_alpha": inv_alpha,
                        "alpha": alpha,
                        "bracket_iteration": iteration,
                        "E_over_Ec": field,
                        "growth": result["growth"],
                        "growth_SEM": result["sem"],
                        "growth_total_sigma": total_sigma,
                        "fit_R2_mean": result["r2_mean"],
                        "max_event_probability": result["max_q"],
                    }
                )
            result, total_sigma = cache[key]
            bracket.append((float(field), float(result["growth"]), total_sigma))

        (e0, g0, s0), (e1, g1, s1) = bracket
        if (g0 <= 0.0 <= g1) or (g1 <= 0.0 <= g0):
            threshold = e0 - g0 * (e1 - e0) / (g1 - g0)
            return {
                "Zeff": zeff,
                "inv_alpha": inv_alpha,
                "alpha": alpha,
                "JONTA_Eav_over_Ec": threshold,
                "JONTA_sigma": _threshold_sigma(bracket),
                "Eq_B15_Eav_over_Ec": estimate,
                "bracket_low": min(e0, e1),
                "bracket_high": max(e0, e1),
                "bracket_growth_low": g0 if e0 < e1 else g1,
                "bracket_growth_high": g1 if e0 < e1 else g0,
                "runtime_seconds": time.perf_counter() - start,
            }, sampled, histories
        span *= 1.7

    raise RuntimeError(
        f"could not bracket B3(b) threshold for Zeff={zeff:g}, 1/alpha={inv_alpha:g}; "
        f"sampled={[(row['E_over_Ec'], row['growth']) for row in sampled]}"
    )


def _write_threshold_plot(output, rows, reference):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    grid = np.geomspace(min(row["inv_alpha"] for row in reference), 80.0, 400)
    for zeff, color, marker in ((1.0, "tab:blue", "o"), (5.0, "tab:red", "s")):
        curve = [float(mcdevitt_threshold_fit(1.0 / x, zeff)) for x in grid]
        ax.plot(grid, curve, color=color, label=fr"Eq. (B15), $Z_{{\rm eff}}={int(zeff)}$")
        ref_rows = [row for row in reference if np.isclose(row["zeff"], zeff)]
        ax.plot(
            [row["inv_alpha"] for row in ref_rows],
            [row["value"] for row in ref_rows],
            marker=marker,
            linestyle="none",
            markerfacecolor="none",
            color=color,
            label=fr"McDevitt MC, $Z_{{\rm eff}}={int(zeff)}$",
        )
        sim_rows = [row for row in rows if np.isclose(float(row["Zeff"]), zeff)]
        ax.errorbar(
            [float(row["inv_alpha"]) for row in sim_rows],
            [float(row["JONTA_Eav_over_Ec"]) for row in sim_rows],
            yerr=[float(row["JONTA_sigma"]) for row in sim_rows],
            fmt=".",
            color=color,
            capsize=3,
            label=fr"JONTA, $Z_{{\rm eff}}={int(zeff)}$",
        )
    ax.set_xscale("log")
    ax.set(xlabel=r"$1/\alpha$", ylabel=r"$E_{\rm av}/E_c$", title="McDevitt Fig. B3(b) avalanche threshold")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "b3b_threshold.png", dpi=240)
    plt.close(fig)


def _write_history_plot(output, history_rows, fit_start_fraction=0.6):
    if not history_rows:
        return
    import matplotlib.pyplot as plt

    cases = sorted({(float(row["Zeff"]), float(row["inv_alpha"])) for row in history_rows})
    ncols = 2
    nrows = (len(cases) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 3.4 * nrows), squeeze=False)
    for axis, (zeff, inv_alpha) in zip(axes.flat, cases):
        case_rows = [
            row
            for row in history_rows
            if np.isclose(float(row["Zeff"]), zeff)
            and np.isclose(float(row["inv_alpha"]), inv_alpha)
        ]
        fields = sorted({float(row["E_over_Ec"]) for row in case_rows})
        for field in fields:
            field_rows = [row for row in case_rows if np.isclose(float(row["E_over_Ec"]), field)]
            for seed in sorted({int(row["seed"]) for row in field_rows}):
                series = sorted(
                    (row for row in field_rows if int(row["seed"]) == seed),
                    key=lambda row: float(row["time_tau_c"]),
                )
                t = np.asarray([float(row["time_tau_c"]) for row in series])
                w = np.asarray([float(row["weight"]) for row in series])
                color = "tab:blue" if field == fields[0] else "tab:red"
                axis.plot(t, np.where(w > 0.0, w, np.nan), color=color, alpha=0.18, linewidth=0.7)
                growth = float(series[0]["fit_growth"])
                intercept = float(series[0]["fit_intercept"])
                fit_start = min(max(float(fit_start_fraction), 0.0), 1.0) * t[-1]
                fit_mask = t >= fit_start
                if np.any(fit_mask):
                    axis.plot(
                        t[fit_mask],
                        np.exp(intercept + growth * t[fit_mask]),
                        color=color,
                        alpha=0.75,
                        linewidth=1.0,
                        linestyle="--",
                    )
        axis.set_yscale("log")
        axis.set_title(fr"$Z_{{\rm eff}}={zeff:g}$, $1/\alpha={inv_alpha:g}$")
        axis.set_xlabel(r"$t/\tau_c$")
        axis.set_ylabel(r"$n_{\rm RE}/n_{\rm RE,0}$")
        axis.grid(alpha=0.25, which="both")
    for axis in axes.flat[len(cases):]:
        axis.remove()
    handles = [
        plt.Line2D([], [], color="tab:blue", alpha=0.7, label="lower bracket field"),
        plt.Line2D([], [], color="tab:red", alpha=0.7, label="upper bracket field"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output / "b3b_histories.png", dpi=200)
    plt.close(fig)


def main():
    args = _args()
    config = load_config(args.config)
    mode = config.execution.mode if args.mode is None else args.mode
    devices = (config.execution.n_devices or 1) if args.devices is None else args.devices
    markers = config.particles.markers if args.markers is None else args.markers
    replicas = config.benchmark.replicas if args.replicas is None else args.replicas
    total_time = config.benchmark.total_time if args.total_time is None else args.total_time
    if mode == "parallel" and (devices < 2 or replicas != devices):
        raise ValueError("parallel B3(b) requires replicas == devices for fixed-shape pmap")
    output = args.output_dir or Path(config.output.directory)
    output.mkdir(parents=True, exist_ok=True)
    fields = tuple(config.field.e_scan if args.inv_alpha is None else args.inv_alpha)
    zeffs = (1.0, 5.0) if args.zeff is None else tuple(args.zeff)
    reference_path = Path(config.reference.data_file)
    reference = _reference(reference_path)
    seed_start = config.particles.seed if args.seed_start is None else args.seed_start
    seeds = tuple(seed_start + i for i in range(replicas))

    threshold_path = output / "b3b_threshold.csv"
    scan_path = output / "b3b_threshold_scan.csv"
    history_path = output / "b3b_histories.csv"
    rows = _read_csv(threshold_path) if args.resume else []
    scan_rows = _read_csv(scan_path) if args.resume else []
    history_rows = _read_csv(history_path) if args.resume else []
    if args.reconcile:
        valid = {(float(zeff), float(inv_alpha)) for zeff in zeffs for inv_alpha in fields}
        rows = [
            row for row in rows
            if (float(row["Zeff"]), float(row["inv_alpha"])) in valid
        ]
        scan_rows = [
            row for row in scan_rows
            if (float(row["Zeff"]), float(row["inv_alpha"])) in valid
        ]
        history_rows = [
            row for row in history_rows
            if (float(row["Zeff"]), float(row["inv_alpha"])) in valid
        ]
    completed = {(float(row["Zeff"]), float(row["inv_alpha"])) for row in rows}
    for zeff in zeffs:
        for inv_alpha in fields:
            key = (float(zeff), float(inv_alpha))
            if args.resume and key in completed:
                print(f"Zeff={zeff:g}, 1/alpha={inv_alpha:g}: already present; skipped")
                continue
            result, samples, histories = _solve_threshold(
                config,
                zeff=float(zeff),
                inv_alpha=float(inv_alpha),
                markers=markers,
                total_time=total_time,
                seeds=seeds,
                mode=mode,
                devices=devices,
            )
            paper = min(
                (
                    row
                    for row in reference
                    if np.isclose(row["zeff"], zeff)
                    and np.isclose(row["inv_alpha"], inv_alpha)
                ),
                key=lambda row: abs(row["inv_alpha"] - inv_alpha),
            )["value"]
            result.update(
                {
                    "paper_MC_Eav_over_Ec": paper,
                    "difference_from_paper": result["JONTA_Eav_over_Ec"] - paper,
                    "relative_difference_from_B15": (
                        result["JONTA_Eav_over_Ec"] - result["Eq_B15_Eav_over_Ec"]
                    )
                    / result["Eq_B15_Eav_over_Ec"],
                    "markers": markers,
                    "replicas": replicas,
                    "total_time_tau_c": total_time,
                    "dt_tau_c": config.timesteps.dt_particle,
                    "large_angle_dt_tau_c": config.timesteps.dt_particle * config.timesteps.large_angle_every,
                    "fit_start_fraction": config.diagnostics.fit_start_fraction,
                }
            )
            rows.append(result)
            scan_rows.extend(samples)
            history_rows.extend(histories)
            _write_csv(threshold_path, rows)
            _write_csv(scan_path, scan_rows)
            _write_csv(history_path, history_rows)
            _write_threshold_plot(output, rows, reference)
            _write_history_plot(output, history_rows, config.diagnostics.fit_start_fraction)
            print(
                f"Zeff={zeff:g}, 1/alpha={inv_alpha:g}: "
                f"Eav/Ec={result['JONTA_Eav_over_Ec']:.8g} +/- {result['JONTA_sigma']:.3g}; "
                f"runtime={result['runtime_seconds']:.1f}s"
            )

    _write_threshold_plot(output, rows, reference)
    _write_history_plot(output, history_rows, config.diagnostics.fit_start_fraction)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    manifest = {
        "benchmark": "slab.avalanche_threshold.b3b",
        "reference": "McDevitt et al. 2019 PPCF 61 054008, Fig. B3(b)",
        "parameters": {
            "zeff": list(zeffs),
            "inv_alpha": list(fields),
            "coulomb_log": config.background.coulomb_log,
            "gamma_min_la": config.collisions.large_angle.gamma_min,
            "markers": markers,
            "replicas": replicas,
            "total_time_tau_c": total_time,
            "dt_tau_c": config.timesteps.dt_particle,
            "large_angle_dt_tau_c": config.timesteps.dt_particle * config.timesteps.large_angle_every,
            "fit_start_fraction": config.diagnostics.fit_start_fraction,
        },
        "backend": {
            "jax": jax.__version__,
            "platform": jax.default_backend(),
            "devices": [str(device) for device in jax.devices()],
            "precision": "float64",
        },
        "runtime": {"python": sys.version.split()[0], "platform": platform.platform(), "git_commit": commit},
        "uncertainty": {
            "growth_rates": "replica SEM plus fit-window RMS in quadrature",
            "threshold": "linear zero-crossing propagation using total growth-rate sigma",
        },
        "outputs": ["b3b_threshold.csv", "b3b_threshold_scan.csv", "b3b_histories.csv", "b3b_threshold.png", "b3b_histories.png"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
