#!/usr/bin/env bash
set -euo pipefail

PY=python3.11
ROOT_DIR=$(cd "$(dirname "$0")"/.. && pwd)
cd "$ROOT_DIR"

if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Python 3.11 not found (expected: $PY)" >&2
  exit 1
fi

VENV_DIR=.venv
if [ ! -d "$VENV_DIR" ]; then
  "$PY" -m venv "$VENV_DIR"
fi
. "$VENV_DIR"/bin/activate

python -m pip install --upgrade pip

# Install CPU-only torch from PyTorch CPU index
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
python -m pip install -e ".[cpu,dev]"

echo "Bootstrap complete. Activate with: source .venv/bin/activate"


