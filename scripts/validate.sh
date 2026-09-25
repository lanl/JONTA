#!/usr/bin/env bash
# Canonical full local validation for JONTA: lint, coding tests, and packaging.
#
# Usage: bash scripts/validate.sh
#
# Runs on CPU with two emulated XLA host devices so serial/parallel execution
# paths are exercised without an accelerator. Set PYTHON to select a specific
# interpreter (default: python on PATH, e.g. an activated .venv).
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

export JAX_PLATFORMS=cpu
export XLA_FLAGS=--xla_force_host_platform_device_count=2
export PYTHONPATH=src

PYTHON="${PYTHON:-python}"

stage() {
    printf '\n==> %s\n' "$1"
}

stage "ruff check src tests"
"${PYTHON}" -m ruff check src tests

stage "pytest"
"${PYTHON}" -m pytest -q

stage "build"
"${PYTHON}" -m build

stage "validation passed"
