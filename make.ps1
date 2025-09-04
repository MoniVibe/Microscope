param([Parameter(Mandatory=$true)][string]$Target)
switch ($Target) {
  "bootstrap" { & scripts/bootstrap.ps1 }
  "verify"    { & scripts/verify.ps1 }
  default     { throw "Unknown target $Target" }
}


