$ErrorActionPreference = "Stop"
New-Item -ItemType Directory -Force -Path reports | Out-Null
$py = @("src/microscope/cli/main.py","tests/test_cli.py","tests/test_config_roundtrip.py")

"[$(Get-Date)] format" | Tee-Object reports/format.log
ruff format | Tee-Object -Append reports/format.log

"[$(Get-Date)] lint" | Tee-Object reports/lint.log
ruff check --select E,F $py | Tee-Object -Append reports/lint.log

"[$(Get-Date)] types" | Tee-Object reports/types.log
mypy src/microscope/cli | Tee-Object -Append reports/types.log

"[$(Get-Date)] tests" | Tee-Object reports/tests.log
pytest -q tests/test_cli.py tests/test_config_roundtrip.py --cov=microscope --cov-report=xml | Tee-Object -Append reports/tests.log


