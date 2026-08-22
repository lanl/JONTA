# JONTA benchmark drivers

Benchmark drivers validate physical observables against analytical relations,
RAMc behavior, or published numerical data. They are separate from coding
tests and are not collected by the default `pytest` command.

Reference data lives beside each benchmark under its `reference/` directory.
The top-level `reference_data/` directory is only a migration index.
Reproduction commands, parameter grids, uncertainty rules, and result policy are documented in
[`../docs/benchmarks.md`](../docs/benchmarks.md).

Each benchmark configuration is recorded as `config.yaml` inside its benchmark
directory. The schema and validation rules are documented in
[`../docs/configuration.md`](../docs/configuration.md). Load a configuration
before constructing JAX arrays, and archive its resolved form with the result.

Current benchmark families:

- [`slab/`](slab/): Maxwellian relaxation, bump-on-tail, Dreicer generation,
  and avalanche/decay.
- [`one_d/`](one_d/): invariant conservation, radial transport, and radial
  avalanche.

The supplied project PDFs are not redistributed. Each benchmark's
`reference/metadata.yaml` records the paper DOI, local source-document name,
figure targets, and digitization status.

Run a driver explicitly, for example:

```bash
PYTHONPATH=src python -m benchmarks.test_mcdevitt_2019_avalanche_figures \
  --figure B3 --output-dir benchmark_results/mcdevitt_B3
```
