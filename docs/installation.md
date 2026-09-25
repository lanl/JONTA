# Installation and development setup

JONTA currently targets Python 3.11 or newer. The supported development path
on Apple silicon is CPU execution. CUDA/ROCm installation is supported when a
matching JAX accelerator build is installed; Apple Metal remains experimental
and is not a production FP64 backend for JONTA.

## CPU development

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Verify the backend and precision path:

```bash
JAX_PLATFORMS=cpu python -c \
  'import jax; print(jax.default_backend()); print(jax.devices())'
JAX_PLATFORMS=cpu PYTHONPATH=src python -m pytest -q
```

JONTA enables FP64 by default. FP32 is available for experiments with
`JONTA_PRECISION=32`, but it is not a replacement for FP64 validation.

Run the canonical full local validation (Ruff, coding tests, and package
build on CPU with two emulated XLA devices) from the activated environment:

```bash
bash scripts/validate.sh
```

### Required tooling

Only Python 3.11+, `pip`, and the dependencies declared in `pyproject.toml`
are required. Node.js, Docker, Claude Code, Codex, and other coding-agent
tools are not required to install, run, test, benchmark, or develop JONTA.

### Optional setup convenience

`scripts/setup_environment.sh` performs the virtual-environment steps above in
one command. It creates `.venv` at the repository root when needed and
installs the package with its development dependencies:

```bash
bash scripts/setup_environment.sh
source .venv/bin/activate
```

It is a convenience only; the manual steps above remain the reference path.

### Multiple checkouts and git worktrees

Parallel lines of work can use separate clones or `git worktree` checkouts.
Give each checkout its own `.venv` and editable install, so that `src/`
imports resolve to that checkout rather than another one. Either repeat the
manual setup or run `bash scripts/setup_environment.sh` inside the checkout.
For CPU reference runs, set `JAX_PLATFORMS=cpu` in the environment or prefix
individual commands with it; `scripts/validate.sh` sets it automatically.

## CUDA or ROCm development

Install the JAX build appropriate to the target driver and accelerator using
the current JAX installation instructions, then install JONTA:

```bash
python -m pip install -e '.[dev]'
python -c 'import jax; print(jax.default_backend()); print(jax.devices())'
```

Record the JAX version, accelerator model, precision, and execution mode with
benchmark results. GPU-specific tests should be run in addition to the CPU
reference suite.

## Current scope

This page intentionally documents installation and validation setup before a
polished end-to-end example. The public quickstart will be expanded once the
coupled simulation workflow is complete. Current reproducible benchmark entry
points are listed in [`docs/benchmarks.md`](benchmarks.md).
