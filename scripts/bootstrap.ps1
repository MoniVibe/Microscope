$ErrorActionPreference = 'Stop'

# Prefer Python 3.11 via py launcher
$py = 'python'
if (Get-Command py -ErrorAction SilentlyContinue) {
  $py = 'py -3.11'
}
Write-Host "Using Python: $py"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

$venv = ".venv"
if (-not (Test-Path $venv)) {
  Invoke-Expression "$py -m venv $venv"
}

# Activate venv for the current process
. .\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip

# Install CPU-only torch from PyTorch CPU index
python -m pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.6.0"
python -m pip install -e ".[cpu,dev]"

Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1"


