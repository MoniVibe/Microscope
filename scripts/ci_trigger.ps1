param(
  [string]$Workflow = 'CI'
)

$ErrorActionPreference = 'Stop'

if (Get-Command gh -ErrorAction SilentlyContinue) {
  $p = Start-Process -FilePath gh -ArgumentList @('workflow','run', $Workflow) -NoNewWindow -Wait -PassThru -ErrorAction SilentlyContinue
  if ($p -and $p.ExitCode -eq 0) {
    Write-Host "Triggered GitHub workflow '$Workflow' via gh."
    exit 0
  } else {
    Write-Warning "gh workflow run failed or returned non-zero; falling back to empty commit trigger."
  }
}

Write-Host "gh not available; creating empty commit to trigger CI..."
git commit --allow-empty -m "ci: manual trigger ($Workflow)" | Out-Null
git push | Out-Null
Write-Host "Pushed empty commit to trigger CI."


