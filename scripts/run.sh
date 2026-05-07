#!/usr/bin/env bash
# One-shot wrapper around `python -m src.main`.
#
# Usage:
#   ./scripts/run.sh <repo_url> [extra flags...]
#
# Examples:
#   ./scripts/run.sh https://github.com/psf/requests
#   ./scripts/run.sh https://github.com/psf/requests --no-xpu --phase1-timeout 1200
set -euo pipefail

if [ ! -f .env ]; then
    echo "ERROR: .env not found — run 'cp .env.example .env' and fill in OPENAI_API_KEY first." >&2
    exit 1
fi

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <repo_url> [extra flags...]" >&2
    exit 1
fi

# Prefer .venv if present, else fall back to system python.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
else
    PY="python"
fi

exec "$PY" -m src.main "$@"
