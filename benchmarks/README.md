# JONTA benchmark drivers

Benchmark drivers validate physical observables against analytical relations,
RAMc behavior, or published numerical data. They are separate from coding
tests and are not collected by the default `pytest` command.

Reference data lives in `reference_data/`. Reproduction commands, parameter
grids, uncertainty rules, and result policy are documented in
[`../docs/benchmarks.md`](../docs/benchmarks.md).

Run a driver explicitly, for example:

```bash
PYTHONPATH=src python -m benchmarks.test_mcdevitt_2019_avalanche_figures \
  --figure B3 --output-dir benchmark_results/mcdevitt_B3
```
