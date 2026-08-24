# JONTA benchmark drivers

Benchmark drivers validate physical observables against analytical relations,
published numerical data, and (where explicitly labelled) internal RAMc
implementation behavior. They are separate from coding tests and are not
collected by the default `pytest` command.

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
- [`one_d/`](one_d/): invariant conservation, trapped/passing tokamak orbits,
  radial transport, and radial avalanche.

The supplied project PDFs are not redistributed. Each benchmark's
`reference/metadata.yaml` records the paper DOI, local source-document name,
figure targets, and digitization status.

Run a benchmark driver explicitly, for example:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=6 \
PYTHONPATH=src:. JAX_PLATFORMS=cpu \
  .venv/bin/python benchmarks/slab/avalanche_decay/run.py \
  --mode parallel --devices 6 --resume
```

The dedicated avalanche driver documents the accepted B3(a) scan. The legacy
multi-figure McDevitt driver remains available for follow-on figure studies.
