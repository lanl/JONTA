# JONTA

Just anOther fuNcTionAl pusher

Approved for open source under #O5194.

JONTA is a JAX-based, GPU-accelerated kinetic Monte Carlo code for runaway electrons.

JONTA is released under the [MIT License](LICENSE). Public contribution,
security, citation, and reproducibility guidance is linked below.

## Goals

JONTA is designed for:

- **GPU-first FP64 execution** on NVIDIA accelerators, with RTX PRO 6000 Blackwell and B200 as primary targets;
- **multi-GPU particle parallelism** using fixed-shape weighted marker ensembles;
- **geometry- and field-representation agnostic physics**, with 0-D and axisymmetric circular RAMc-like models supplied as the first reference implementations;
- **analytic or interpolated electromagnetic fields** behind interchangeable field backends;
- **modular numerical physics**, including swappable integrators, collision operators, source models, resamplers, boundary models, depositors, and plasma closures;
- **physics-driven convergence and validation**, using analytical results and published runaway-electron benchmarks. RAMc is used only as internal implementation provenance and a legacy comparison where explicitly identified.

Automatic differentiation is available where the JAX-native formulation permits it, but GPU throughput, numerical robustness, and physical correctness are the primary design requirements.

## Implemented reference model

The current reference implementation contains the major building blocks needed for 0-D studies and the circular 1-D RAMc geometry:

- fixed-size weighted particle state in `(gamma, xi, x, y, phi)`;
- uniform 0-D deterministic orbit model;
- analytic circular RAMc guiding-center equations;
- synchrotron radiation in the deterministic orbit;
- Euler, midpoint, and RK4 fixed-step integrators behind a common contract;
- Maxwellian-background test-particle small-angle collisions;
- conservative linearized Moller gain-loss large-angle collisions;
- fixed-capacity local marker population control with overflow-only stratified thinning;
- tritium and Compton kinetic sources;
- radial binning and RAMc-compatible parallel-current deposition;
- scalar slab/0-D moment deposition through the same macrostep contract;
- Spitzer resistivity reference closure;
- implicit BDF2 radial electric-field evolution;
- Picard particle/Ohm coupling for the circular 1-D model;
- geometry-neutral particle macrostep and serial/sharded moment reduction;
- geometry-neutral Picard driver with circular and slab backend seams;
- homogeneous slab algebraic Ohm coupling adapter;
- circular safety-factor evolution;
- electron/ion energy bookkeeping;
- implicit charge-state evolution using preprocessed OpenADAS/ADAS rate tables;
- JAX sharding helpers and a particle-throughput benchmark driver.

This is a **reference implementation**, not yet a fully validated production release. The validation plan in `docs/validation.md` distinguishes implemented coding tests from the analytical and published physics benchmarks that remain to be completed. Internal RAMc comparisons are identified as implementation cross-checks, not independent scientific validation.

## Computational model

The expensive kinetic work is regular and accelerator-oriented. Particle arrays remain fixed in size and are advanced with a fixed particle timestep. Different physical processes may have independent cadences:

```text
particle timestep dt_p
    deterministic orbit + synchrotron
    small-angle collisions

large-angle cadence dt_LA
    conservative Moller gain-loss update
    local capacity control and overflow thinning

plasma-coupling cadence dt_c
    deposit kinetic moments
    nonlinear particle/plasma coupling iteration
    implicit BDF2 electric-field / Ohm update
    optional q, energy, and charge-state updates
```

`dt_p`, `dt_LA`, and `dt_c` are intentionally distinct. Their values are selected by convergence and nonlinear-coupling robustness rather than by a single global definition of a macro timestep.

## Repository structure

```text
jonta/
├── README.md
├── AGENTS.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CITATION.cff
├── CHANGELOG.md
├── pyproject.toml
├── .github/
├── benchmark_results/
├── docs/
│   ├── architecture.md
│   ├── agent_workflow.md
│   ├── benchmarks.md
│   ├── code_map.md
│   ├── conventions.md
│   ├── numerics.md
│   ├── physics.md
│   ├── installation.md
│   └── validation.md
├── src/
│   ├── boundaries/
│   ├── collisions/
│   ├── core/
│   ├── coupling/
│   ├── deposition/
│   ├── diagnostics/
│   ├── fields/
│   ├── integrators/
│   ├── orbits/
│   ├── parallel/
│   ├── plasma/
│   ├── resampling/
│   ├── sources/
│   └── simulation.py
├── tests/
│   ├── unit/              coding/unit contracts
│   ├── integration/       compiled and serial/parallel execution tests
│   └── regression/        deterministic software reference cases
├── benchmarks/            paper/analytic benchmark drivers and reference data
├── examples/
└── scripts/
```

The implementation lives directly beneath `src/`; there is deliberately no `src/jonta/` package layer.

## Documentation

Read these in roughly this order:

- [`docs/physics.md`](docs/physics.md) — governing physical model and citations.
- [`docs/architecture.md`](docs/architecture.md) — module boundaries, state ownership, and JAX/multi-GPU data flow.
- [`docs/numerics.md`](docs/numerics.md) — Monte Carlo realization, timestepping, resampling, deposition, BDF2, and coupling algorithms.
- [`docs/conventions.md`](docs/conventions.md) — normalization, signs, units, FP64, coordinates, and array layout.
- [`docs/validation.md`](docs/validation.md) — required coding, convergence, published-physics, and performance tests.
- [`docs/installation.md`](docs/installation.md) — CPU/CUDA setup and environment verification.
- [`docs/benchmarks.md`](docs/benchmarks.md) — reproducible benchmark commands and result policy.
- [`docs/configuration.md`](docs/configuration.md) — validated YAML schema and benchmark templates.
- [`docs/code_map.md`](docs/code_map.md) — equation/model-to-source navigation map.
- [`docs/agent_workflow.md`](docs/agent_workflow.md) — LLM task routing, execution modes, validation ladder, and handoff contract.
- [`AGENTS.md`](AGENTS.md) — repository-wide rules for human and LLM-assisted development.

## Getting started

JONTA requires Python 3.11+ and JAX. For GPU use, install the JAX build appropriate to the CUDA environment using the current JAX installation instructions, then install JONTA in editable mode:

```bash
python -m pip install -e .
```

JONTA defaults to FP64. Select FP32 before creating arrays or JIT kernels with
`JONTA_PRECISION=32`, or pass `precision=32` to particle initialization and
particle-block construction. Precision is process-static; do not switch it
after arrays or compiled kernels exist.

For development:

```bash
python -m pip install -e '.[dev]'
PYTHONPATH=src pytest -q
```

`pytest -q` runs the complete coding-test suite only. Physical validation is
explicit and lives under `benchmarks/`; use the commands in
[`docs/benchmarks.md`](docs/benchmarks.md).

Paper and analytical comparisons live under `benchmarks/` and are run through
the commands documented in [`docs/benchmarks.md`](docs/benchmarks.md). They
are never hidden behind a test marker.

On Apple silicon, the optional Metal backend can be installed with:

```bash
python -m pip install -e '.[dev,metal]'
```

The Metal JAX plugin is experimental and is suitable only for limited
float32/backend-availability checks. JONTA's production kernels intentionally enable
FP64, which the Metal plugin does not support. Use `JAX_PLATFORMS=cpu` for all
JONTA tests and validation runs:

```bash
JAX_PLATFORMS=cpu python -c 'import jax; print(jax.devices())'
```

Without that override, JAX will select the Metal device when the local macOS
runtime exposes one, but JONTA's compiled FP64 kernels are expected to fail on
that backend. Published validation runs should therefore use CPU on Apple
silicon or a supported FP64 accelerator such as CUDA, and record the backend.

The explicit `PYTHONPATH` is useful when working directly from a checkout; an editable install also exposes the packages under `src/`.

Current developer/reference scripts (not yet the polished end-to-end public
quickstart):

```bash
PYTHONPATH=src python examples/zero_d_runaway.py
PYTHONPATH=src python examples/ramc_1d.py
```

Particle blocks expose one backend-independent execution policy:

```python
from core.config import ExecutionConfig
from simulation import build_particle_block

serial = build_particle_block(..., execution=ExecutionConfig("serial", "cpu"))
parallel = build_particle_block(
    ..., execution=ExecutionConfig("parallel", "cpu", n_devices=4)
)
```

Use `platform="gpu"` for a supported accelerator. Serial execution is the
reference path; parallel execution shards markers, performs population control
locally, and replicates compact field state. Global resampling is not required
for the particle/plasma coupling design.

A simple throughput driver is available as:

```bash
PYTHONPATH=src python scripts/benchmark_particle_push.py --help
```

## Development principles

- Keep physical equations separate from geometry, field storage, interpolation, and integration algorithms.
- Keep all hot particle kernels JAX-native and device resident.
- Preserve fixed particle-array shapes during production evolution.
- Use FP64 throughout the physics path.
- Replicate compact field/profile data and shard particles across GPUs by default.
- Treat stochastic reproducibility in terms of stable particle IDs, global step, and operator stream, not execution order.
- Require convergence/validation evidence for integrator, collision-cadence, plasma-coupling, and resampling changes.
- Avoid full-particle host output in normal production diagnostics.
- Keep files, interfaces, and documentation explicit enough that future LLM agents can safely inspect and refactor the codebase.

## Current status and next validation milestones

The software skeleton and reference 0-D/1-D physics kernels are in place. A self-contained Maxwellian-relaxation benchmark is currently reproducible from its checked-in driver and recorded result manifest. Other benchmark families have reference data, specifications, or legacy drivers at different stages of migration; their presence does not constitute completed paper-level validation. Before calling JONTA production-ready, the highest-priority remaining work is:

1. additional deterministic orbit/transport benchmarks against published analytical and numerical results;
2. Dreicer generation and the partially screened avalanche-threshold benchmark;
3. manufactured BDF2/current-deposition and nonlinear plasma-coupling tests;
4. OpenADAS-backed charge-state/radiation datasets and verification;
5. RTX PRO 6000 Blackwell and B200 single-/multi-GPU throughput and scaling studies.
