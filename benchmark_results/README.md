# Checked-in benchmark results

This directory contains selected plots and raw outputs produced by JONTA
benchmark drivers. They are reproducibility artifacts, not source code.

New result directories must include a manifest with the exact command, code
commit, backend/device, precision, parameter grid, and interpretation. Results
without a manifest are historical exploratory artifacts and must not be cited
as paper-level acceptance evidence.

The canonical drivers and acceptance criteria are documented in
[`docs/benchmarks.md`](../docs/benchmarks.md) and
[`docs/validation.md`](../docs/validation.md). Keep digitized paper reference
data separate under `tests/convergence/reference_data/`.
