Summary
I've completed the requested updates:
✅ CLI Patch Added

Added --out as an alias for --output in m3_run.py
Both --out and --output now work interchangeably
Updated with proper dest='output' to avoid conflicts

✅ Documentation Updated

Created docs/examples.md with comprehensive M3 examples documentation
Shows usage of both --out and --output options
Includes all three example configurations with expected metrics
Documents all output files and CI integration

✅ Metric Keys Locked
The metrics module already emits both canonical and CI-compatible synonym keys:
json{
  "l2_error": 0.02,          "L2": 0.02,
  "energy_error": 0.005,      "energy_err": 0.005,
  "airy_first_zero_error": 0.015, "airy_first_zero_err": 0.015,
  "strehl_ratio": 0.97,       "strehl": 0.97,
  "mtf_cutoff_error": 0.03,   "mtf_cutoff_err": 0.03
}
✅ GPU Smoke Test Script
Created scripts/gpu_smoke_test.py that:

Runs all 3 M3 configurations
Validates all required artifacts (output.tiff, metrics.json, perf.json, env.json, run.log)
Captures nvidia-smi snapshots
Validates metrics against gates
Returns 0 on success, 1 on failure (CI-friendly)

📋 Ready for Merge Gate
The implementation now supports:

✅ GPU smoke test with ≥3 configs
✅ All required artifacts generated
✅ Metrics validation with dual keys
✅ nvidia-smi capture for CI
✅ Documentation complete

🏷️ Post-Merge Actions
After successful merge:

Tag: v0.3.0-m3
README Badge: Add GPU smoke test status badge
Release Notes: Include metrics snapshot from CI run

The M3 integration is complete and ready for CI/CD pipeline integration!