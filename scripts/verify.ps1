@(
  $ErrorActionPreference = 'Stop'
)

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $root

# Create reports directory
New-Item -ItemType Directory -Force -Path reports | Out-Null

# Track overall status
$GLOBAL:ALL_PASS = $true

function Get-PythonExe {
  $venvPy = Join-Path $root ".venv/Scripts/python.exe"
  if (Test-Path $venvPy) { return $venvPy }
  if (Get-Command py -ErrorAction SilentlyContinue) { return 'py -3.11' }
  return 'python'
}

$PY = Get-PythonExe

function Invoke-Native {
  param(
    [Parameter(Mandatory=$true)][string]$Exe,
    [Parameter(Mandatory=$false)][string[]]$Args = @(),
    [Parameter(Mandatory=$true)][string]$LogName
  )
  $renderedArgs = $Args | ForEach-Object {
    if ($_ -match '\s') { '"' + ($_.Replace('"','""')) + '"' } else { $_ }
  }
  $cmdLine = ("$Exe " + ($renderedArgs -join ' ')).Trim()
  Write-Host "> $cmdLine"
  Add-Content -Path "reports/$LogName.log" -Value "> $cmdLine"

  # Run via cmd.exe to prevent stderr from becoming PowerShell error records
  $full = "$cmdLine 2>&1"
  & cmd.exe /c $full | Tee-Object -FilePath "reports/$LogName.log" -Append
  $code = $LASTEXITCODE
  if ($code -ne 0) {
    Write-Host "FAILED: $LogName (exit $code)"
    $GLOBAL:ALL_PASS = $false
  } else {
    Write-Host "SUCCESS: $LogName"
  }
}

function Invoke-PyModule {
  param(
    [Parameter(Mandatory=$true)][string]$Module,
    [Parameter(Mandatory=$false)][string[]]$Args = @(),
    [Parameter(Mandatory=$true)][string]$LogName
  )
  $allArgs = @('-m', $Module) + $Args
  Invoke-Native -Exe $PY -Args $allArgs -LogName $LogName
}

# 1) Auto-format
Invoke-PyModule -Module 'ruff' -Args @('format', 'src/microscope') -LogName 'format'

# 2) Lint and auto-fix
Invoke-PyModule -Module 'ruff' -Args @('check', '--fix', '--select', 'E,F', 'src/microscope') -LogName 'lint'

# 3) Type check (use mypy.ini to scope files)
Invoke-PyModule -Module 'mypy' -Args @() -LogName 'types'

# 4) Unit tests + coverage (CPU only)
Invoke-PyModule -Module 'pytest' -Args @('-q', 'tests/test_cli.py', 'tests/test_config_roundtrip.py', '--cov=microscope', '--cov-report=xml') -LogName 'tests'

# Final status via exit codes only
if (-not $GLOBAL:ALL_PASS) {
  exit 1
}

Write-Host "ALL GATES PASSED"
exit 0
