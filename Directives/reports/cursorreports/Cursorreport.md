Style-lint integration added ahead of CPU job. Also previously added per-case nvidia-smi “before” capture. Everything else in m3-ci meets your requirements:
Branch/PR flow: cpu-gate-loop → gpu-smoke → determinism → build-and-docs (via needs).
Matrix ≥3 configs: discovery + hard-fail assert if fewer from examples/m3/*.yml|*.yaml.
GPU preflight: asserts CUDA available and device count ≥1; captures nvidia-smi (top-level and per-case before/after).
Gates per case: thresholds enforced for L2, energy, airy first zero, strehl, mtf_cutoff, VRAM < 4e9, wall_time < 90 in the “Gate checks” step.
Artifacts per case (30d): output.tiff, metrics.json, perf.json, env.json, run.log, nvidia_smi_before.txt, nvidia_smi_after.txt.
PR comment: summary job posts metrics, VRAM, wall-time, and artifact names on the PR.
Style-lint: `style-lint` job runs Ruff format+fix, then enforces only E/F errors; `cpu-gate-loop` depends on `style-lint`.
Summary:
Updated .github/workflows/m3_ci.yml to add per-case nvidia_smi_before.txt.
The workflow now enforces your hard requirements end-to-end and posts a PR summary.