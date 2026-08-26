# Contributing to JONTA

JONTA is an open scientific-software project. Contributions should preserve
physical correctness, numerical convergence, reproducibility, and the
serial/parallel execution contract.

## Development setup

Follow [`docs/installation.md`](docs/installation.md) to create an environment
and install the development dependencies. Read [`AGENTS.md`](AGENTS.md) and
[`docs/agent_workflow.md`](docs/agent_workflow.md) before changing physics,
numerics, benchmarks, or execution code.

## Before opening a pull request

- keep the change focused and explain the physical or numerical motivation;
- add or update the smallest relevant coding test; keep physical validation
  and paper comparisons in `benchmarks/`;
- update the governing document and `docs/code_map.md` when an interface,
  equation, or module boundary changes;
- run the complete coding-test suite;
- run explicit validation/benchmark jobs relevant to the change;
- report commands, backend, precision, and any benchmark that was not run.

The standard local checks are:

```bash
python -m ruff check path/to/changed/files
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q
```

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
