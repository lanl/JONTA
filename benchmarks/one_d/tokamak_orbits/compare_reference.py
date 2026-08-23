"""Compare fixed-step trajectories against the analytic-equation reference."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")


def _load_method(path: Path):
    data = np.load(path / "tokamak_orbit_trajectories.npz")
    manifest = json.loads((path / "manifest.json").read_text())
    return data, manifest


def compare(reference_dir: Path, methods, output_dir: Path):
    reference = np.load(reference_dir / "analytic_reference_trajectories.npz")
    rows = []
    for name, path in methods:
        data, manifest = _load_method(path)
        indices = np.rint(data["time"] / reference["time"][1]).astype(int)
        error = np.abs(data["history"] - reference["history"][indices])
        for i, label in enumerate(data["labels"]):
            rows.append(
                {
                    "method": name,
                    "marker": i,
                    "label": str(label),
                    "dt": manifest["dt"],
                    "steps": manifest["n_steps"],
                    "runtime_seconds": manifest["runtime_seconds"],
                    "max_abs_r_error": float(error[:, 0, i].max()),
                    "max_abs_xi_error": float(error[:, 2, i].max()),
                    "max_abs_gamma_error": float(error[:, 3, i].max()),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "reference_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    styles = ("-", "--", ":", "-.")
    for (name, path), style in zip(methods, styles):
        data, _ = _load_method(path)
        indices = np.rint(data["time"] / reference["time"][1]).astype(int)
        error = data["history"] - reference["history"][indices]
        for i, label in enumerate(data["labels"]):
            axes[0, 0].plot(data["time"], error[:, 0, i], style, label=f"{name} {label}")
            axes[0, 1].plot(data["time"], error[:, 2, i], style, label=f"{name} {label}")
            axes[1, 0].plot(data["time"], error[:, 3, i], style, label=f"{name} {label}")
            axes[1, 1].plot(data["time"], error[:, 4, i], style, label=f"{name} {label}")
    axes[0, 0].set_ylabel(r"$r-r_{\rm ref}$")
    axes[0, 1].set_ylabel(r"$\xi-\xi_{\rm ref}$")
    axes[1, 0].set_ylabel(r"$\gamma-\gamma_{\rm ref}$")
    axes[1, 1].set_ylabel(r"$P_\phi-P_{\phi,\rm ref}$")
    for axis in axes.flat:
        axis.set_xlabel(r"$t/\tau_c$")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
    fig.savefig(output_dir / "reference_comparison.png", dpi=220)
    plt.close(fig)

    # Runtime is one synchronized scalar per method, not a time-history
    # quantity. Keep it in a separate figure instead of using a misleading
    # secondary axis on the trajectory-error plots.
    runtimes = {}
    for row in rows:
        runtimes.setdefault(row["method"], []).append(row["runtime_seconds"])
    names = list(runtimes)
    values = [float(np.median(runtimes[name])) for name in names]
    fig, ax = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
    bars = ax.bar(names, values, color=("#4c78a8", "#f58518", "#54a24b")[: len(names)])
    ax.set_ylabel("synchronized runtime [s]")
    ax.set_title("Toroidal orbit benchmark runtime")
    ax.grid(True, axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.3f} s",
            (bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
    fig.savefig(output_dir / "reference_runtime.png", dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        action="append",
        help="method specification NAME=DIRECTORY; repeat for each method",
    )
    parser.add_argument("--rk4-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    methods = []
    for specification in args.method or []:
        try:
            name, directory = specification.split("=", 1)
        except ValueError as exc:
            raise ValueError("--method must have the form NAME=DIRECTORY") from exc
        methods.append((name, Path(directory)))
    if args.rk4_dir is not None:
        methods.append(("rk4", args.rk4_dir))
    if not methods:
        parser.error("provide at least one --method NAME=DIRECTORY")
    compare(args.reference_dir, methods, args.output_dir)


if __name__ == "__main__":
    main()
