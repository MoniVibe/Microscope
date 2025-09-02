### CI pipeline trigger — M3

- Branch/PR: `m3/ci-wire-and-release` — “Wire M3 runner in CI, enforce gates, build docs and wheels.”
- Workflow: `m3-ci` pushed and enabled. Order: cpu-gate-loop → gpu-smoke → determinism → build-and-docs.
- Matrix discovery asserts ≥3 configs from `examples/m3/*.yml|*.yaml`.
- Determinism env: `PYTORCH_DETERMINISTIC=1`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `TORCH_ALLOW_TF32=0`, Python prelude sets deterministic algorithms and `torch.set_float32_matmul_precision("highest")`.
- Per-case gates: L2|l2_error ≤0.03, energy_err|energy_error ≤0.01, airy_first_zero_* ≤0.02, strehl|strehl_ratio ≥0.95, mtf_cutoff_* ≤0.05, peak_vram_bytes <4e9, wall_time_sec <90. Anti-dummy checks included.
- Artifacts per case (30d): `output.tiff`, `metrics.json`, `perf.json`, `env.json`, `run.log`, `nvidia_smi_before.txt`, `nvidia_smi_after.txt`.
- Build job: wheel+sdist, import-smoke, docs build strict; uploads `dist/*` and `site/**` (30d).
- PR summary: sticky comment posts per-case metrics and budgets with artifact names.

Repo: [MoniVibe/Microscope](https://github.com/MoniVibe/Microscope)

Update 2025-09-02 — Bootstrap alignment
- Broadened triggers to push/PR on all branches; set `permissions: contents: read`.
- Added timeouts: CPU 20m, GPU 60m, Determinism 20m, Build+Docs 20m.
- GPU smoke: added concurrency group `gpu-smoke`, CUDA preflight (torch CUDA availability + device list), `nvidia-smi -L` and `nvidia-smi` capture.
- Artifacts now include top-level `nvidia_smi.txt` in addition to per-case before/after logs.
- Build-and-docs now depends on `determinism` only, per bootstrap sequence.
- CPU gate job uses pip cache for faster installs.
