"""Analytic Rosenbluth--Legendre warm-start test for avalanche Monte Carlo.

The test remains particle Monte Carlo only.  The initializer is an inexpensive
analytic approximation to the dominant avalanche eigenmode:

* Rosenbluth-Putvinski high-field growth gives the exponential energy scale;
* Guo-McDevitt-Tang synchrotron balance limits that scale at moderate field;
* momentum-dependent pitch structure is projected onto a Legendre basis.

The production kinetic solve is unchanged.  The only question tested here is
whether the analytic seed reduces burn-in before fitting the avalanche growth
rate.  McDevitt et al. (2019) Appendix Fig. B3(a) provides the numerical target.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from diagnostics.avalanche import replicate_mean_sem
try:
    from benchmarks.test_large_angle_avalanche import (
        b3_growth_reference,
        run_growth_replicates,
    )
except ModuleNotFoundError:  # direct execution from benchmark tree
    from benchmarks.test_large_angle_avalanche import b3_growth_reference, run_growth_replicates


def _b3_reference(e_over_ec: float = 3.23210) -> float:
    rows = b3_growth_reference()
    row = min(rows, key=lambda r: abs(r["e_over_ec"] - e_over_ec))
    return float(row["value"])


def run_warm_start_comparison(
    *,
    e_over_ec: float = 3.23210,
    n_markers: int = 512,
    total_time: float = 6.0,
    seeds=(3, 11, 29, 47, 71, 89, 101, 131),
):
    """Compare the legacy energetic seed with the analytic warm start."""

    common = dict(
        alpha=0.5,
        zeff=2.0,
        coulog0=20.0,
        n_markers=int(n_markers),
        total_time=float(total_time),
        dt=0.005,
        large_angle_dt=0.1,
        seeds=tuple(seeds),
        # The purpose of the warm start is to make this early fit reliable.
        fit_start_fraction=0.05,
    )
    uniform = run_growth_replicates(e_over_ec, **common)
    warm = run_growth_replicates(
        e_over_ec,
        **common,
        initialization="rosenbluth_legendre",
        rosenbluth_lmax=24,
        rosenbluth_pitch_width=None,
        rosenbluth_seed_floor_fraction=0.15,
        rosenbluth_energy_extent=1.7,
    )
    return {
        "reference": _b3_reference(e_over_ec),
        "uniform": uniform,
        "warm": warm,
        "e_over_ec": float(e_over_ec),
        "n_markers": int(n_markers),
        "total_time": float(total_time),
    }


def test_rosenbluth_legendre_warm_start_reduces_b3_burn_in():
    """A short CPU run is closer to Fig. B3 when warm-started analytically."""

    result = run_warm_start_comparison(seeds=(3, 11, 29, 47))
    reference = result["reference"]
    uniform = result["uniform"]
    warm = result["warm"]

    warm_error = abs(warm["growth"] - reference)
    uniform_error = abs(uniform["growth"] - reference)

    # With only six collision times available, the warm start should already
    # recover the published B3 growth within independent-replica uncertainty.
    assert warm_error < max(2.5 * warm["sem"], 0.003)
    assert warm["r2_mean"] > 0.90

    # This is the efficiency claim: for the same markers, RNG replicas and
    # integration length, the analytic seed must reduce transient bias.
    assert warm_error < uniform_error


def _write_results(result, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    reference = result["reference"]

    summary_path = outdir / "b3_warm_start_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("initialization", "growth", "std", "sem", "r2_mean", "reference", "error"),
        )
        writer.writeheader()
        for name in ("uniform", "warm"):
            r = result[name]
            writer.writerow(
                {
                    "initialization": name,
                    "growth": r["growth"],
                    "std": r["std"],
                    "sem": r["sem"],
                    "r2_mean": r["r2_mean"],
                    "reference": reference,
                    "error": r["growth"] - reference,
                }
            )

    replicate_path = outdir / "b3_warm_start_replicates.csv"
    with replicate_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("initialization", "replica", "growth"))
        for name in ("uniform", "warm"):
            for i, growth in enumerate(result[name]["replicates"]):
                writer.writerow((name, i, float(growth)))

    import matplotlib.pyplot as plt

    names = ("uniform RE seed", "Rosenbluth-Legendre")
    values = np.asarray([result["uniform"]["growth"], result["warm"]["growth"]])
    errors = np.asarray([result["uniform"]["sem"], result["warm"]["sem"]])
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(6.4, 4.5))
    ax.errorbar(x, values, yerr=errors, fmt="o", capsize=4, label="JONTA, replica SEM")
    ax.axhline(reference, linestyle="--", label="McDevitt Fig. B3(a)")
    ax.set_xticks(x, names)
    ax.set_ylabel(r"$\gamma_{av}\tau_c$")
    ax.set_title(rf"Short-run B3 warm start, $E/E_c={result['e_over_ec']:.3f}$")
    ax.legend()
    fig.tight_layout()
    comparison_path = outdir / "b3_warm_start_comparison.png"
    fig.savefig(comparison_path, dpi=300)
    plt.close(fig)

    # Plot mean log-population histories with independent-replica SEM bands.
    # Remove the deliberately short source-insertion startup jump by anchoring
    # all curves at 0.5 tau_c; the remaining slope is the avalanche eigenvalue.
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    anchor_time = 0.5
    warm_mean = None
    for name, label in (("uniform", "uniform RE seed"), ("warm", "Rosenbluth-Legendre")):
        histories = result[name]["histories"]
        time = histories[0][1]
        logw = np.stack([np.log(h[2]) for h in histories])
        i_anchor = int(np.argmin(np.abs(time - anchor_time)))
        logw = logw - logw[:, [i_anchor]]
        mean = np.mean(logw, axis=0)
        sem = np.std(logw, axis=0, ddof=1) / np.sqrt(logw.shape[0])
        if name == "warm":
            warm_mean = mean
        line = ax.plot(time[i_anchor:], mean[i_anchor:], label=label)[0]
        ax.fill_between(
            time[i_anchor:],
            mean[i_anchor:] - sem[i_anchor:],
            mean[i_anchor:] + sem[i_anchor:],
            alpha=0.18,
            color=line.get_color(),
        )
    reference_line = reference * (time - time[i_anchor])
    ax.plot(time[i_anchor:], reference_line[i_anchor:], linestyle="--", label="published B3 slope")
    ax.set_xlabel(r"$t/\tau_c$")
    ax.set_ylabel(r"$\ln[W(t)/W(0.5\tau_c)]$")
    ax.set_title("Post-startup capacity-controlled avalanche growth")
    ax.legend()
    fig.tight_layout()
    history_path = outdir / "b3_warm_start_history.png"
    fig.savefig(history_path, dpi=300)
    plt.close(fig)

    return summary_path, replicate_path, comparison_path, history_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, default=Path("warm_start_results"))
    parser.add_argument("--markers", type=int, default=512)
    parser.add_argument("--time", type=float, default=6.0)
    parser.add_argument("--replicas", type=int, default=8)
    args = parser.parse_args()
    seeds = (3, 11, 29, 47, 71, 89, 101, 131, 149, 173, 197, 211)[: args.replicas]
    result = run_warm_start_comparison(
        n_markers=args.markers,
        total_time=args.time,
        seeds=seeds,
    )
    paths = _write_results(result, args.outdir)
    print(f"reference={result['reference']:.8f}")
    for name in ("uniform", "warm"):
        r = result[name]
        print(
            f"{name}: growth={r['growth']:.8f} +/- {r['sem']:.8f} SEM, "
            f"R2={r['r2_mean']:.5f}"
        )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
