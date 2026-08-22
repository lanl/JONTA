# Benchmark output directory

Benchmark drivers write plots, raw tables, manifests, and resolved
configurations here. Generated outputs are intentionally not tracked in Git;
they are reproducible from the benchmark scripts and checked-in YAML templates.

Every local result directory should include a manifest with the exact command,
code commit, backend/device, precision, parameter grid, and interpretation.
Results without a manifest must not be cited as paper-level acceptance evidence.

The canonical drivers and acceptance criteria are documented in
[`docs/benchmarks.md`](../docs/benchmarks.md) and
[`docs/validation.md`](../docs/validation.md). Keep digitized paper reference
data separate under the consuming benchmark's `reference/` directory.
