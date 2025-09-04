#!/usr/bin/env bash
set -euo pipefail

mkdir -p reports
PY_PATHS=("src/microscope/cli/main.py" "tests/test_cli.py" "tests/test_config_roundtrip.py")

echo "[format]" | tee reports/format.log
ruff format | tee -a reports/format.log

echo "[lint]" | tee reports/lint.log
ruff check --select E,F "${PY_PATHS[@]}" | tee -a reports/lint.log

echo "[types]" | tee reports/types.log
mypy src/microscope/cli | tee -a reports/types.log

echo "[tests]" | tee reports/tests.log
pytest -q tests/test_cli.py tests/test_config_roundtrip.py --cov=microscope --cov-report=xml | tee -a reports/tests.log


