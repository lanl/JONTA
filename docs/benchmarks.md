# Reproducible benchmarks

Checked-in benchmark results are evidence attached to a specific code version,
not a substitute for the benchmark drivers. Every new result directory should
contain:

- the exact command used;
- a JSON manifest with backend, device, precision, parameters, and code commit;
- raw CSV/JSON output;
- plots only when they are generated from the raw output;
- a short statement of whether the result is preview, regression, or paper-level
  acceptance evidence.

Reference data digitized from papers belongs under
`benchmarks/reference_data/` with provenance and licensing information.
Generated results belong under `benchmark_results/`; exploratory results may be
kept locally or archived separately when they are not part of the public
record.

## Guiding-center invariant scan

The primary deterministic-orbit scan is:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python \
  tests/validation/test_guiding_center_invariants.py \
  --output-dir benchmark_results/adiabatic_invariants_cpu_full
```

This uses the documented full electric-field and timestep grids in
[`tests/validation/README.md`](../tests/validation/README.md). It is an
expensive validation job, not a smoke test. A bounded CPU preview must override
both axes explicitly:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python \
  tests/validation/test_guiding_center_invariants.py \
  --electric-fields 0 10 10000 \
  --dts 2.56e-7 6.4e-8 1.6e-8 \
  --final-time 1e-5 --n-particles 8 \
  --output-dir benchmark_results/adiabatic_invariants_cpu_preview
```

The driver writes RK2/RK4 plots, an error-floor plot, a CSV containing the
actual adjusted timestep and finite-state flag, and a JSON scan manifest.

## Published McDevitt figures

Paper-scale figure reproduction is exposed one figure per process and is
intended for a supported GPU:

```bash
PYTHONPATH=src python benchmarks/test_mcdevitt_2019_avalanche_figures.py \
  --figure 2 --paper --output-dir benchmark_results/mcdevitt_fig2
```

The driver documents which figures have reduced CPU paths and which require
paper-scale statistics. Do not label a reduced run as a paper reproduction.
The transport family is similarly exposed through:

```bash
PYTHONPATH=src python tests/validation/test_spatial_transport.py \
  --output-dir benchmark_results/transport_cpu_preview
```

Use `--paper` only for the published parameter set on an appropriate backend.

## Result review

Before adding a result to Git, verify:

1. the command succeeds from a clean environment;
2. the output manifest identifies the code commit and backend;
3. the result is compared with the documented acceptance target;
4. preview, exploratory, and acceptance outputs are named distinctly.
