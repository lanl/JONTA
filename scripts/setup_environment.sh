#!/usr/bin/env bash
# Install the JONTA development environment in a Codex local worktree.
set -euo pipefail

cd "${CODEX_WORKTREE_PATH:-.}"

if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'

echo "JONTA environment ready: $PWD/.venv"
