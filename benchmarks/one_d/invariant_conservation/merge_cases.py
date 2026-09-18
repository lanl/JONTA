"""Merge independent guiding-center invariant case results."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmarks.one_d.invariant_conservation.run import (
    _write_csv,
    _write_floor_plot,
    _write_integrator_plot,
    _write_runtime_plot,
)


def _parse_row(row):
    integer_fields = {
        "n_steps",
        "accepted_internal_steps",
        "rejected_internal_steps",
    }
    float_fields = {
        "E1_over_Ec",
        "dt",
        "pphi_max_rel_error",
        "mu_max_rel_error",
        "output_interval",
        "runtime_seconds",
    }
    boolean_fields = {"finite", "adaptive_converged"}
    parsed = dict(row)
    for field in integer_fields:
        if field in parsed and parsed[field] != "":
            parsed[field] = int(float(parsed[field]))
    for field in float_fields:
        if field in parsed and parsed[field] != "":
            parsed[field] = float(parsed[field])
    for field in boolean_fields:
        if field in parsed:
            parsed[field] = parsed[field].lower() == "true"
    return parsed


def _read_case(path):
    with (path / "guiding_center_invariants_efield_scan.csv").open(newline="") as stream:
        return [_parse_row(row) for row in csv.DictReader(stream)]


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, default=None)
    return parser.parse_args()


def main():
    args = _parse_args()
    rows = []
    manifests = []
    for input_path in args.inputs:
        rows.extend(_read_case(input_path))
        with (input_path / "guiding_center_invariants_efield_scan.json").open() as stream:
            manifests.append(json.load(stream))
    if not rows:
        raise ValueError("no invariant case rows found")

    rows.sort(key=lambda row: (row["integrator"], row["E1_over_Ec"], -row["dt"]))
    datasets = {}
    for row in rows:
        datasets.setdefault(row["integrator"], {}).setdefault(row["E1_over_Ec"], []).append(row)
    for field_scan in datasets.values():
        for e1 in field_scan:
            field_scan[e1].sort(key=lambda row: row["dt"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "guiding_center_invariants_efield_scan.csv", datasets)
    _write_integrator_plot(
        args.output_dir / "guiding_center_invariants_efield_scan_rk4.png",
        "RK4",
        datasets["RK4"],
        4,
    )
    _write_integrator_plot(
        args.output_dir / "guiding_center_invariants_efield_scan_bs5.png",
        "BS5",
        datasets["BS5"],
        5,
    )
    _write_integrator_plot(
        args.output_dir / "guiding_center_invariants_efield_scan_bs5_adaptive.png",
        "BS5-adaptive",
        datasets["BS5-adaptive"],
    )
    _write_floor_plot(args.output_dir / "guiding_center_invariants_efield_floor.png", datasets)
    _write_runtime_plot(args.output_dir / "guiding_center_invariants_runtime.png", datasets)

    first = manifests[0]
    manifest = dict(first)
    manifest.update(
        {
            "scan_type": "case_merged",
            "case_directories": [str(path) for path in args.inputs],
            "electric_fields_E1_over_Ec": sorted(
                {float(row["E1_over_Ec"]) for row in rows}
            ),
            "timesteps_dt_over_tau_c_requested": sorted(
                {float(row["dt"]) for row in rows}, reverse=True
            ),
            "backend_records": [item.get("jax_backend") for item in manifests],
            "timing": {
                "per_case_field": True,
                "compilation_excluded": True,
                "synchronization": "jax.block_until_ready",
                "runtime_field": "runtime_seconds",
            },
        }
    )
    if args.reference_dir is not None:
        reference_rows = _read_case(args.reference_dir)

        def key(row):
            return row["integrator"], row["E1_over_Ec"], row["dt"]

        reference = {key(row): row for row in reference_rows}
        current = {key(row): row for row in rows}
        common = set(reference) & set(current)
        differences = {}
        for field in ("pphi_max_rel_error", "mu_max_rel_error"):
            differences[field] = max(
                abs(float(current[item][field]) - float(reference[item][field]))
                for item in common
            )
        manifest["cpu_reference"] = {
            "directory": str(args.reference_dir),
            "matching_rows": len(common),
            "missing_gpu_rows": len(set(reference) - set(current)),
            "missing_cpu_rows": len(set(current) - set(reference)),
            "max_absolute_error_difference": differences,
            "finite_mismatch_count": sum(
                current[item]["finite"] != reference[item]["finite"] for item in common
            ),
        }
    (args.output_dir / "guiding_center_invariants_efield_scan.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"merged cases: {len(args.inputs)}")
    print(f"merged rows: {len(rows)}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
