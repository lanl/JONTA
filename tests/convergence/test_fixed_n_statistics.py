"""Statistical validation of JONTA's fixed-N branching representation.

The production avalanche algorithm keeps a fixed number of marker slots while
physical population growth is carried by marker weight.  Candidate gain/loss
ensembles are randomly thinned with the production stratified resampler.

This file validates the statistical design at two levels:

1. a cheap two-type linear branching process with a known dominant eigenvalue;
2. the published McDevitt-2019 Appendix Fig. B3(a) avalanche growth rate.

Uncertainties are always estimated from independent replica ensembles.  Marker
samples inside one fixed-N realization are not treated as independent after
resampling.
"""

from __future__ import annotations

import argparse
import csv
from functools import lru_cache
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from core.state import KinematicState, ParticleState
from diagnostics.avalanche import monte_carlo_convergence_slope, replicate_mean_sem
from resampling.multinomial import stratified_resample
try:
    from tests.convergence.test_large_angle_avalanche import (
        b3_growth_reference,
        run_growth_replicates,
    )
except ModuleNotFoundError:  # direct execution from tests/convergence
    from test_large_angle_avalanche import b3_growth_reference, run_growth_replicates

jax.config.update("jax_enable_x64", True)

# A positive two-type branching matrix.  M[i, j] is the expected weight of
# type-i children produced by one unit of type-j parent weight per macrostep.
# The two parent types have different total reproduction factors, so finite-N
# composition fluctuations produce genuine population-growth noise.
_BRANCH_MATRIX = np.asarray([[0.96, 0.06], [0.04, 1.04]], dtype=float)


def _dominant_branching_mode():
    values, vectors = np.linalg.eig(_BRANCH_MATRIX)
    i = int(np.argmax(values.real))
    lam = float(values[i].real)
    vec = np.abs(vectors[:, i].real)
    fractions = vec / np.sum(vec)
    return lam, fractions


@lru_cache(maxsize=16)
def _build_branching_kernel(n_markers: int, n_steps: int = 200):
    """Return a JIT/vmapped production-resampling branching kernel."""

    lam, fractions = _dominant_branching_mode()
    del lam
    n0 = int(round(float(fractions[0]) * n_markers))
    xi0 = jnp.concatenate(
        [
            -jnp.ones((n0,), dtype=jnp.float64),
            jnp.ones((n_markers - n0,), dtype=jnp.float64),
        ]
    )
    z = jnp.zeros((n_markers,), dtype=jnp.float64)
    p0 = ParticleState(
        KinematicState(2.0 * jnp.ones_like(z), xi0, z, z, z),
        jnp.full((n_markers,), 1.0 / n_markers, dtype=jnp.float64),
        jnp.ones((n_markers,), dtype=jnp.bool_),
        jnp.arange(n_markers, dtype=jnp.int64),
    )
    matrix = jnp.asarray(_BRANCH_MATRIX, dtype=jnp.float64)

    def one_replica(base_key):
        def body(i, p):
            parent_type = (p.kin.xi > 0.0).astype(jnp.int32)
            w0 = p.weight * matrix[0, parent_type]
            w1 = p.weight * matrix[1, parent_type]
            wc = jnp.concatenate([w0, w1])
            n2 = 2 * n_markers
            z2 = jnp.zeros((n2,), dtype=jnp.float64)
            kin = KinematicState(
                2.0 * jnp.ones((n2,), dtype=jnp.float64),
                jnp.concatenate(
                    [
                        -jnp.ones((n_markers,), dtype=jnp.float64),
                        jnp.ones((n_markers,), dtype=jnp.float64),
                    ]
                ),
                z2,
                z2,
                z2,
            )
            candidates = ParticleState(
                kin,
                wc,
                jnp.ones((n2,), dtype=jnp.bool_),
                jnp.arange(n2, dtype=jnp.int64),
            )
            key = jax.random.fold_in(base_key, jnp.asarray(i, dtype=jnp.uint32))
            return stratified_resample(candidates, key, n_markers)

        pf = jax.lax.fori_loop(0, n_steps, body, p0)
        return jnp.log(jnp.sum(pf.weight)) / n_steps

    return jax.jit(jax.vmap(one_replica))


def run_branching_replicates(n_markers, seeds=tuple(range(16)), n_steps=200):
    """Run the analytic branching test and return independent-replica stats."""

    # Construct independent keys explicitly while keeping the expensive time
    # evolution vectorized across replicas.
    keys = jnp.stack([jax.random.key(int(seed)) for seed in seeds])
    samples = np.asarray(_build_branching_kernel(n_markers, n_steps)(keys))
    mean, std, sem = replicate_mean_sem(samples)
    exact = float(np.log(_dominant_branching_mode()[0]))
    return {
        "n_markers": int(n_markers),
        "mean": mean,
        "std": std,
        "sem": sem,
        "exact": exact,
        "bias": mean - exact,
        "replicates": samples,
    }


def _b3_reference_at(e_over_ec=3.23210):
    rows = b3_growth_reference()
    return min(rows, key=lambda r: abs(r["e_over_ec"] - e_over_ec))


def run_b3_marker_scan(
    marker_counts=(256, 512, 1024),
    seeds=tuple(range(8)),
    *,
    e_over_ec=3.23210,
):
    """Marker-count scan at a representative Fig. B3(a) point."""

    rows = []
    for n_markers in marker_counts:
        result = run_growth_replicates(
            e_over_ec,
            alpha=0.5,
            zeff=2.0,
            coulog0=20.0,
            n_markers=int(n_markers),
            total_time=10.0,
            dt=0.005,
            large_angle_dt=0.1,
            seeds=tuple(seeds),
            fit_start_fraction=0.35,
        )
        rows.append(
            {
                "n_markers": int(n_markers),
                "growth": result["growth"],
                "std": result["std"],
                "sem": result["sem"],
                "r2_mean": result["r2_mean"],
                "max_q": result["max_q"],
            }
        )
    return rows


def run_b3_macrostep_scan(
    large_angle_steps=(0.1, 0.05, 0.025),
    seeds=(3, 11, 29, 47, 71, 89),
    *,
    e_over_ec=3.23210,
    n_markers=768,
):
    """Large-angle macrostep scan at a representative Fig. B3(a) point."""

    rows = []
    for dt_la in large_angle_steps:
        result = run_growth_replicates(
            e_over_ec,
            alpha=0.5,
            zeff=2.0,
            coulog0=20.0,
            n_markers=int(n_markers),
            total_time=10.0,
            dt=0.005,
            large_angle_dt=float(dt_la),
            seeds=tuple(seeds),
            fit_start_fraction=0.35,
        )
        rows.append(
            {
                "large_angle_dt": float(dt_la),
                "growth": result["growth"],
                "std": result["std"],
                "sem": result["sem"],
                "r2_mean": result["r2_mean"],
                "max_q": result["max_q"],
            }
        )
    return rows


def test_fixed_n_branching_recovers_exact_growth_and_mc_scaling():
    """Fixed-N random thinning is unbiased and converges statistically."""

    counts = np.asarray([128, 256, 512, 1024])
    rows = [run_branching_replicates(int(n)) for n in counts]
    std = np.asarray([row["std"] for row in rows])
    slope, _ = monte_carlo_convergence_slope(counts, std)

    # The finest ensemble should recover the known dominant eigenvalue well
    # within independent-replica uncertainty plus a small finite-N allowance.
    fine = rows[-1]
    assert abs(fine["bias"]) < 5.0 * fine["sem"] + 2.0e-5

    # Ordinary Monte-Carlo behavior is sigma ~ N^-1/2. Stratification can make
    # the observed slope somewhat steeper; allow finite-sample scatter without
    # accepting a non-convergent or increasing-noise method.
    assert -0.9 < slope < -0.25
    scaled = std * np.sqrt(counts)
    assert np.max(scaled) / np.min(scaled) < 2.5


@pytest.mark.slow
def test_b3_fixed_n_statistics_converge_with_marker_count():
    """Published Fig. B3(a) is recovered as fixed-N statistics improve."""

    rows = run_b3_marker_scan()
    counts = np.asarray([row["n_markers"] for row in rows], dtype=float)
    std = np.asarray([row["std"] for row in rows])
    slope, _ = monte_carlo_convergence_slope(counts, std)
    reference = float(_b3_reference_at()["value"])
    fine = rows[-1]

    assert abs(fine["growth"] - reference) < max(3.0 * fine["sem"], 0.004)
    assert np.all(np.asarray([row["max_q"] for row in rows]) < 0.20)
    assert -0.9 < slope < -0.15
    scaled = std * np.sqrt(counts)
    assert np.max(scaled) / np.min(scaled) < 2.0


@pytest.mark.slow
def test_b3_large_angle_macrostep_is_resolved():
    """Fig. B3 growth is insensitive to the large-angle update cadence."""

    rows = run_b3_macrostep_scan()
    fine = rows[-1]
    for row in rows[:-1]:
        combined = np.sqrt(row["sem"] ** 2 + fine["sem"] ** 2)
        assert abs(row["growth"] - fine["growth"]) < 3.0 * combined + 0.003
    # The finest macrostep keeps the maximum one-step collision fraction small.
    assert fine["max_q"] < 0.05
    # No scan point reaches the production clipping limit.
    assert max(row["max_q"] for row in rows) < 0.20


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_synthetic(output_dir: Path):
    import matplotlib.pyplot as plt

    counts = np.asarray([128, 256, 512, 1024])
    rows = [run_branching_replicates(int(n), seeds=tuple(range(32))) for n in counts]
    exact = rows[0]["exact"]
    mean = np.asarray([row["mean"] for row in rows])
    sem = np.asarray([row["sem"] for row in rows])
    std = np.asarray([row["std"] for row in rows])
    slope, _ = monte_carlo_convergence_slope(counts, std)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.errorbar(counts, mean, yerr=sem, marker="o", capsize=3, label="fixed-N JONTA")
    ax.axhline(exact, linestyle="--", label="exact branching eigenvalue")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("marker count N")
    ax.set_ylabel("growth per macrostep")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_n_branching_growth.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.loglog(counts, std, marker="o", label=f"replica std, fitted slope={slope:.2f}")
    ref = std[0] * np.sqrt(counts[0] / counts)
    ax.loglog(counts, ref, linestyle="--", label=r"$N^{-1/2}$")
    ax.set_xlabel("marker count N")
    ax.set_ylabel("replica standard deviation")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fixed_n_branching_noise.png", dpi=300)
    plt.close(fig)

    csv_rows = [
        {
            "n_markers": row["n_markers"],
            "mean_growth": row["mean"],
            "replica_std": row["std"],
            "sem": row["sem"],
            "exact_growth": row["exact"],
            "bias": row["bias"],
        }
        for row in rows
    ]
    _write_csv(output_dir / "fixed_n_branching_statistics.csv", csv_rows)
    return csv_rows, slope


def _plot_b3_markers(output_dir: Path):
    import matplotlib.pyplot as plt

    rows = run_b3_marker_scan(seeds=tuple(range(12)))
    reference_row = _b3_reference_at()
    reference = float(reference_row["value"])
    counts = np.asarray([row["n_markers"] for row in rows], dtype=float)
    growth = np.asarray([row["growth"] for row in rows])
    sem = np.asarray([row["sem"] for row in rows])
    std = np.asarray([row["std"] for row in rows])
    slope, _ = monte_carlo_convergence_slope(counts, std)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.errorbar(counts, growth, yerr=sem, marker="o", capsize=3, label="JONTA MC mean +/- SEM")
    ax.axhline(reference, linestyle="--", label="McDevitt Fig. B3(a)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("marker count N")
    ax.set_ylabel(r"$\gamma_{av}\tau_c$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "b3_marker_convergence.png", dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.loglog(counts, std, marker="o", label=f"replica std, fitted slope={slope:.2f}")
    ref = std[0] * np.sqrt(counts[0] / counts)
    ax.loglog(counts, ref, linestyle="--", label=r"$N^{-1/2}$")
    ax.set_xlabel("marker count N")
    ax.set_ylabel(r"std($\gamma_{av}\tau_c$)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "b3_marker_noise.png", dpi=300)
    plt.close(fig)

    csv_rows = []
    for row in rows:
        csv_rows.append({**row, "paper_growth": reference, "paper_e_over_ec": reference_row["e_over_ec"]})
    _write_csv(output_dir / "b3_marker_convergence.csv", csv_rows)
    return csv_rows, slope


def _plot_b3_macrostep(output_dir: Path):
    import matplotlib.pyplot as plt

    rows = run_b3_macrostep_scan()
    reference = float(_b3_reference_at()["value"])
    x = np.asarray([row["large_angle_dt"] for row in rows])
    growth = np.asarray([row["growth"] for row in rows])
    sem = np.asarray([row["sem"] for row in rows])

    fig, ax = plt.subplots(figsize=(6.2, 4.3))
    ax.errorbar(x, growth, yerr=sem, marker="o", capsize=3, label="JONTA MC mean +/- SEM")
    ax.axhline(reference, linestyle="--", label="McDevitt Fig. B3(a)")
    ax.set_xscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"large-angle interval $\Delta t_{LA}/\tau_c$")
    ax.set_ylabel(r"$\gamma_{av}\tau_c$")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "b3_macrostep_convergence.png", dpi=300)
    plt.close(fig)

    _write_csv(output_dir / "b3_macrostep_convergence.csv", rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("synthetic", "b3-markers", "b3-macrostep"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.case == "synthetic":
        rows, slope = _plot_synthetic(args.output_dir)
        print(f"fixed-N synthetic fitted noise slope: {slope:.3f}")
        for row in rows:
            print(row)
    elif args.case == "b3-markers":
        rows, slope = _plot_b3_markers(args.output_dir)
        print(f"Fig. B3 marker-count fitted noise slope: {slope:.3f}")
        for row in rows:
            print(row)
    else:
        rows = _plot_b3_macrostep(args.output_dir)
        for row in rows:
            print(row)


if __name__ == "__main__":
    main()
