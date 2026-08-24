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

Reference data digitized from papers belongs beside the consuming benchmark
under `benchmarks/slab/*/reference/` or `benchmarks/one_d/*/reference/`, with
provenance and licensing information.
Generated results belong under `benchmark_results/`; exploratory results may be
kept locally or archived separately when they are not part of the public
record.

## Slab Maxwellian relaxation

The production small-angle Fokker--Planck operator can be tested independently
of fields, radiation, large-angle collisions, and plasma coupling:

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cpu python \
  benchmarks/slab/maxwellian_relaxation/run.py \
  --output-dir benchmark_results/slab/maxwellian_relaxation/serial
```

The driver records relaxation metrics and synchronized runtime for each
temperature case. Its convergence mode scans marker count, collision timestep,
and independent seeds:

```bash
PYTHONPATH=src:. JAX_PLATFORMS=cpu python \
  benchmarks/slab/maxwellian_relaxation/run.py \
  --convergence \
  --output-dir benchmark_results/slab/maxwellian_relaxation/convergence
```

CPU decomposition scaling uses the same fixed-shape collision kernel and
logical XLA CPU devices. See the benchmark README for the exact scaling
command and interpretation.

## Guiding-center invariant scan

## Tokamak trapped/passing orbits

The collisionless circular-tokamak orbit benchmark is:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python \
  benchmarks/one_d/tokamak_orbits/run.py \
  --output-dir benchmark_results/one_d/tokamak_orbits/rk4
```

It records trapped and passing trajectories, banana width, bounce-period
estimates, and \(P_\phi\)/\(\mu\) conservation. The driver supports RK4,
fixed Bogacki--Shampine 5, and adaptive Bogacki--Shampine 5(4). The adaptive
method is the RAMc-compatible production comparison; the fixed methods are
used for timestep-convergence studies.

The primary deterministic-orbit scan is:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python \
  benchmarks/one_d/invariant_conservation/run.py \
  --output-dir benchmark_results/one_d/invariant_conservation_cpu
```

This uses the documented full electric-field and timestep grids in
[`benchmarks/one_d/invariant_conservation/README.md`](../benchmarks/one_d/invariant_conservation/README.md).
It is an expensive validation job. A bounded CPU development preview must override
both axes explicitly:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python \
  benchmarks/one_d/invariant_conservation/run.py \
  --electric-fields 0 10 10000 \
  --dts 2.56e-7 6.4e-8 1.6e-8 \
  --final-time 1e-5 --n-particles 8 \
  --output-dir benchmark_results/one_d/invariant_conservation_preview
```

The driver writes RK4, fixed BS5, and adaptive BS5(4) convergence plots, an
error-floor plot, a CSV containing actual adjusted timesteps and adaptive
accept/reject counts, and a JSON scan manifest.

Its CPU decomposition scaling case is:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=8 \
PYTHONPATH=src JAX_PLATFORMS=cpu python \
  benchmarks/one_d/invariant_conservation/run.py \
  --scaling --scaling-devices 1 2 4 8 \
  --output-dir benchmark_results/one_d/invariant_conservation_scaling_cpu
```

This keeps total marker work fixed and reports measured versus ideal speedup;
devices are logical XLA CPU devices, not physical-core measurements.

## Published McDevitt figures

The accepted source-limit B3(a) scan has its own reproducible driver and
documentation at
[`benchmarks/slab/avalanche_decay/`](../benchmarks/slab/avalanche_decay/).
Use that driver for the slab growth-rate benchmark; the multi-figure command
below is reserved for the remaining paper figures.

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
PYTHONPATH=src python benchmarks/one_d/radial_transport/run.py \
  --output-dir benchmark_results/transport_cpu_preview
```

Use `--paper` only for the published parameter set on an appropriate backend.

## Result review

Before adding a result to Git, verify:

1. the command succeeds from a clean environment;
2. the output manifest identifies the code commit and backend;
3. the result is compared with the documented acceptance target;
4. preview, exploratory, and acceptance outputs are named distinctly.
