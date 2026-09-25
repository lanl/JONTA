# Contributing to JONTA

JONTA is an open scientific-software project. Contributions should preserve
physical correctness, numerical convergence, reproducibility, and the
serial/parallel execution contract.

## Development setup

Follow [`docs/installation.md`](docs/installation.md) to create an environment
and install the development dependencies. Read [`AGENTS.md`](AGENTS.md) before
changing physics, numerics, benchmarks, or execution code; its rules apply to
every contributor, human or agent-assisted.

Agent-assisted development is optional. No coding-agent tool, Node.js, or
Docker is needed to contribute. Contributors who use one can follow the
provider-neutral guidance in
[`docs/agent_workflow.md`](docs/agent_workflow.md).

## Before opening a pull request

- keep the change focused and explain the physical or numerical motivation;
- add or update the smallest relevant coding test; keep physical validation
  and paper comparisons in `benchmarks/`;
- update the governing document and `docs/code_map.md` when an interface,
  equation, or module boundary changes;
- run `bash scripts/validate.sh`;
- run explicit validation/benchmark jobs relevant to the change;
- report commands, backend, precision, and any benchmark that was not run.

Focused checks are useful while iterating:

```bash
python -m ruff check path/to/changed/files
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q path/to/test_module.py
```

The canonical full local pre-PR check is:

```bash
bash scripts/validate.sh
```

It runs `ruff check src tests`, the complete coding-test suite, and
`python -m build` on CPU with two emulated XLA devices, matching CI.

Coding tests and physics benchmarks are distinct. `scripts/validate.sh` runs
coding tests only. Physical validation and paper comparisons are explicit jobs
under `benchmarks/`, run with the commands in
[`docs/benchmarks.md`](docs/benchmarks.md); they are never hidden behind a
test marker. GPU-specific changes additionally need a run on a CUDA device
when one is available; say so explicitly when one is not.

CI lint covers `src/` and coding tests. Benchmark drivers are executable
validation programs with benchmark-specific dependencies; lint touched drivers
when changing them, but do not mass-reformat unrelated files in a focused
change.

## Physics and benchmark changes

Physics changes require a governing citation, an explicit statement of the
normalization/sign convention, and validation evidence. Benchmark changes must
record the exact command, requested parameter grid, actual adjusted timestep,
backend, and output interpretation. A reduced development preview must never
be presented as paper-level acceptance evidence.

## Pull requests

Use a clear title and describe:

1. what changed;
2. why it changed;
3. tests and benchmarks run;
4. known limitations or follow-up work.

By contributing, you agree that your contribution may be distributed under
the MIT License in this repository.
