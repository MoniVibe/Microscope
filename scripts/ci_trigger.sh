#!/usr/bin/env bash
set -euo pipefail

workflow="${1:-CI}"

if command -v gh >/dev/null 2>&1; then
  gh workflow run "$workflow"
  echo "Triggered GitHub workflow '$workflow' via gh."
  exit 0
fi

echo "gh not available; creating empty commit to trigger CI..."
git commit --allow-empty -m "ci: manual trigger ($workflow)" >/dev/null
git push >/dev/null
echo "Pushed empty commit to trigger CI."


