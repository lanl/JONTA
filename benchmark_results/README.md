# Benchmark output directory

Benchmark drivers write plots, raw tables, manifests, and resolved
configurations here. Canonical outputs for the current validation pass are kept
alongside the drivers so the repository shows the results reproduced by the
documented commands. Exploratory runs remain local and should not be added
here.

Every local result directory should include a manifest with the exact command,
code commit, backend/device, precision, parameter grid, and interpretation.
Results without a manifest must not be cited as paper-level acceptance evidence.

During migration of legacy drivers, the manifest must explicitly state that
the run used CLI arguments rather than a resolved YAML configuration. Such a
result is a reproducibility record only; it cannot be promoted to paper-level
acceptance evidence until the consuming driver writes and archives its
resolved configuration.

The canonical drivers and acceptance criteria are documented in
[`docs/benchmarks.md`](../docs/benchmarks.md), with the physics hierarchy in
[`docs/validation.md`](../docs/validation.md). Keep digitized paper reference
data separate under the consuming benchmark's `reference/` directory.
