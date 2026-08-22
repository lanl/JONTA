# LLM development workflow

This file is the short operational guide for an agent entering JONTA. The
repository contract remains [`AGENTS.md`](../AGENTS.md); this page explains how
to navigate and validate a change without relying on conversation history.

## Start here

1. Read `AGENTS.md` completely.
2. Read the relevant section of `docs/architecture.md` and the matching row in
   `docs/code_map.md`.
3. Read `docs/conventions.md` before touching normalized variables, signs,
   coordinates, state layout, or precision.
4. Read `docs/physics.md` before changing an equation or model.
5. Read `docs/numerics.md` before changing an integrator, stochastic operator,
   cadence, resampler, deposition, or coupling algorithm.
6. Read `docs/validation.md` and the relevant `tests/validation/README.md`
   section before changing a benchmark.
7. Read `docs/benchmarks.md` before adding or interpreting checked-in results.

Use the source tree as the implementation map:

| Task | Primary location | Required evidence |
| --- | --- | --- |
| state, precision, RNG, configuration | `src/core/` | unit test plus convention update when applicable |
| field representation/interpolation | `src/fields/` | field/unit test and interpolation convergence when applicable |
| orbit equations | `src/orbits/` | equation citation, orbit regression/convergence test |
| integration algorithm | `src/integrators/` | order/conservation convergence test |
| collisions, sources, resampling | corresponding physics directory | conservation/statistical test and numerical documentation |
| serial/parallel device execution | `src/parallel/`, `src/simulation.py` | serial reference versus CPU sharding equivalence |
| plasma/coupling/deposition | `src/plasma/`, `src/coupling/`, `src/deposition/` | manufactured or conservation test plus coupling residual evidence |
| coding test | `tests/unit/`, `tests/integration/` | deterministic contract or execution-path evidence |
| numerical/physical validation | `tests/validation/` | convergence, conservation, or statistical evidence |
| benchmark driver or figure | `benchmarks/` | exact parameter grid, raw output, plot, and acceptance comparison |

## Execution modes

Particle blocks use one backend-independent policy:

```python
from core.config import ExecutionConfig

ExecutionConfig(mode="serial", platform="cpu")
ExecutionConfig(mode="parallel", platform="cpu", n_devices=2)
ExecutionConfig(mode="parallel", platform="gpu")
```

Serial execution is the trusted one-device reference. Parallel execution
shards the leading particle axis, performs population control locally, and
replicates compact field/background state. CPU parallel tests emulate multiple
devices with XLA; this tests device decomposition, not physical-core speedup.

Apple Metal is not a production validation backend for JONTA's FP64 path. Use
CPU for Apple-silicon development and a supported CUDA/ROCm backend for future
GPU validation.

## Validation ladder

Run the smallest relevant check first, then expand:

```bash
# syntax/style for the files touched by a focused change
python -m ruff check path/to/changed_source.py path/to/changed_test.py

# complete coding-test suite
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q

# serial/parallel CPU equivalence (logical CPU devices)
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  tests/integration/test_simulation.py tests/integration/test_parallel_orbit.py

# explicit numerical/physical validation
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q tests/validation
```

Coding tests always run at full configured fidelity. Numerical/physical
validation and paper benchmarks are explicit jobs, never hidden behind a
selective test filter. GPU-specific changes additionally require a CUDA test when
such a device is available.

The current reference tree has pre-existing whole-tree Ruff debt in older
benchmark/test files. Do not mass-reformat unrelated files as part of a physics
or architecture change; run Ruff on touched paths and track a dedicated lint
cleanup separately.

## Benchmark discipline

Benchmark defaults must remain physically meaningful. Do not add a tiny,
under-resolved case merely to make a test fast. If a bounded development
preview is needed:

- override every reduced axis explicitly (fields, timesteps, markers, final time);
- name the output directory `*_preview` or `*_development`;
- write the exact requested grid and runtime backend alongside raw results;
- state clearly that the preview is not an acceptance result.

For convergence scans, report the requested grid and the actual adjusted
timestep used after `final_time / n_steps` quantization. Preserve histories,
fits, uncertainty estimates, and plots required by the corresponding paper
comparison.

## Change and handoff contract

Before editing, inspect `git status` and preserve unrelated work. After editing:

1. add/update the smallest relevant tests;
2. update the governing documentation and `docs/code_map.md` when interfaces or
   module ownership change;
3. run the validation ladder appropriate to the risk;
4. report changed files, commands/results, unresolved assumptions, and whether
   the full acceptance benchmark was actually run.

Do not rewrite synced project reference material or generated data outside the
requested scope. Keep commits focused and use a short Conventional Commit
message when the user requests a snapshot.
