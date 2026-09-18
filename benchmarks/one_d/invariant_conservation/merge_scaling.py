"""Merge local and distributed invariant-scaling result directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from benchmarks.one_d.invariant_conservation.run import (
    _write_scaling_csv,
    _write_scaling_plot,
)


def _read_rows(path: Path):
    with (path / "guiding_center_invariants_scaling.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for field in (
            "devices",
            "local_devices",
            "processes",
            "particles_per_device",
            "total_particles",
        ):
            row[field] = int(row[field])
        for field in (
            "wall_time_s",
            "speedup",
            "ideal_speedup",
            "parallel_efficiency",
            "pphi_max_rel_error",
            "mu_max_rel_error",
        ):
            row[field] = float(row[field])
        row["finite"] = row["finite"].lower() == "true"
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    manifests = []
    for path in args.inputs:
        rows.extend(_read_rows(path))
        with (path / "guiding_center_invariants_scaling.json").open() as stream:
            manifests.append(json.load(stream))
    if not rows:
        raise ValueError("no scaling rows found")

    rows.sort(key=lambda row: row["devices"])
    devices = [row["devices"] for row in rows]
    if len(devices) != len(set(devices)):
        raise ValueError("duplicate device counts in scaling inputs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_scaling_csv(
        args.output_dir / "guiding_center_invariants_scaling.csv",
        rows,
    )
    _write_scaling_plot(
        args.output_dir / "guiding_center_invariants_scaling.png",
        rows,
    )
    manifest = {
        "benchmark": "guiding_center_invariants",
        "mode": "particle_scaling_merged",
        "platform": "cuda",
        "input_directories": [str(path) for path in args.inputs],
        "rows": rows,
        "input_manifests": [manifest.get("mode") for manifest in manifests],
    }
    (args.output_dir / "guiding_center_invariants_scaling.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(f"merged rows: {len(rows)}")
    print(f"devices: {devices}")
    print(f"wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
