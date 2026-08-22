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

### Codex local worktrees

The repository includes `scripts/setup_environment.sh` for Codex local
environments. Set the environment's setup script to:

```bash
bash scripts/setup_environment.sh
```

It creates `.venv` when needed and installs the package with its development
dependencies from `pyproject.toml`. No Node.js or Docker setup is required.

For CPU reference runs, add `JAX_PLATFORMS=cpu` as an environment variable or
prefix individual test commands with it. Leave the cleanup script empty unless
the worktree later gains project-specific temporary services.

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
