#!/usr/bin/env bash
# Optional convenience: create .venv at the repository root and install JONTA
# with its development dependencies. Equivalent to the manual steps in
# docs/installation.md; no agent tooling, Node.js, or Docker is required.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if [[ ! -x .venv/bin/python ]]; then
    python3 -m venv .venv
fi

.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'

echo "JONTA environment ready: $PWD/.venv"
