# Agent-assisted development workflow

This page is an optional, provider-neutral workflow for coding agents (Claude
Code, Codex, or any other tool) working on JONTA. It is layered on top of the
normal developer workflow in [`docs/installation.md`](installation.md) and
[`CONTRIBUTING.md`](../CONTRIBUTING.md); it adds no required tooling. The
repository contract remains [`AGENTS.md`](../AGENTS.md); this page explains how
to navigate and validate a change without relying on conversation history.

JONTA does not depend on any agent tool, container, or launcher. Agent-specific
configuration, credentials, and worktree orchestration belong outside this
repository.

## Where commands run

Do not assume the agent runs on the same machine as the simulation backend.
Typical arrangements include:

- the agent, checkout, and CPU reference environment on one workstation;
- the agent editing locally while GPU or HPC jobs run on a remote host,
  scheduler, or cluster that the agent cannot reach directly;
- the agent running in a sandbox or container without accelerators.

Run `scripts/validate.sh` in whatever environment holds the checkout and its
Python environment. For GPU, multi-node, or HPC validation, either run the
commands on the target backend or hand them to the user with the exact
command, backend, and expected evidence. Never report a GPU or HPC result that
was not actually observed, and state which environment produced each result.

## Start here

1. Read `AGENTS.md` completely.
2. Read the relevant section of `docs/architecture.md` and the matching row in
   `docs/code_map.md`.
3. Read `docs/conventions.md` before touching normalized variables, signs,
   coordinates, state layout, or precision.
4. Read `docs/physics.md` before changing an equation or model.
5. Read `docs/numerics.md` before changing an integrator, stochastic operator,
   cadence, resampler, deposition, or coupling algorithm.
6. Read `docs/validation.md` and the relevant benchmark README under
   `benchmarks/` before changing a physical benchmark.
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
| numerical/physical validation | `benchmarks/` | convergence, conservation, or statistical evidence |
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

Run the smallest relevant check first, then expand. The final local step for
any code change is the canonical full validation, `bash scripts/validate.sh`.

```bash
# syntax/style for the files touched by a focused change
python -m ruff check path/to/changed_source.py path/to/changed_test.py

# complete coding-test suite
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q

# serial/parallel CPU equivalence (logical CPU devices)
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q \
  tests/integration/test_simulation.py tests/integration/test_parallel_orbit.py

# canonical full local validation: ruff src tests, full pytest, python -m build
# (CPU, two emulated XLA devices)
bash scripts/validate.sh

# explicit physical benchmark (choose the benchmark documented in docs/benchmarks.md)
PYTHONPATH=src JAX_PLATFORMS=cpu python benchmarks/one_d/invariant_conservation/run.py \
  --output-dir benchmark_results/one_d/invariant_conservation_gpu
```

Coding tests always run at full configured fidelity. Numerical/physical
validation and paper benchmarks are explicit jobs, never hidden behind a
selective test filter. GPU-specific changes additionally require a CUDA test when
such a device is available; if none is available to the agent or user, say
explicitly in the handoff that GPU validation was not run.

CI lint covers the source and coding-test trees. Benchmark drivers are kept
outside that gate because they are executable validation programs with
benchmark-specific dependencies; run Ruff on any touched benchmark driver and
avoid mass-reformatting unrelated files.

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

## Dirty worktrees

Before editing, inspect `git status`. Treat any existing uncommitted change,
untracked file, or local branch state as unrelated work that belongs to the
user or another session:

- do not revert, overwrite, stash, reformat, or delete it;
- keep your edits separable from it, and mention overlaps in the handoff;
- if the requested change cannot be made without touching it, stop and ask.

After validation, inspect `git diff` and `git status` again. Confirm that only
intended files changed and that no build outputs (`dist/`, `build/`,
`*.egg-info`) or scratch benchmark results are tracked.

## Commit, push, and external actions

Agents do not commit, push, merge, tag, open or update pull requests, post
comments, or submit remote/HPC jobs unless the user explicitly requests that
action. A request covers the named action only; it does not authorize later
ones. When a commit is requested, keep it focused and use a short Conventional
Commit message.

## Change and handoff contract

Before editing, inspect `git status` and preserve unrelated work. After editing:

1. add/update the smallest relevant tests;
2. update the governing documentation and `docs/code_map.md` when interfaces or
   module ownership change;
3. run the validation ladder appropriate to the risk, ending with
   `bash scripts/validate.sh`;
4. report changed files, commands/results and the environment that ran them,
   `git status`, unresolved assumptions, whether GPU validation ran, and whether
   the full acceptance benchmark was actually run.

Do not rewrite synced project reference material or generated data outside the
requested scope.
