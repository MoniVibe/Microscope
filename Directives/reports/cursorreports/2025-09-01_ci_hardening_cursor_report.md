### CI hardening for gpu-smoke, artifacts, and docs (Cursor report)

- Implemented runtime-discovered GPU smoke matrix from `examples/m3/*.yml, *.yaml`.
- Added GPU job concurrency guard:
  - group: `gpu-smoke`
  - cancel-in-progress: `false`
- Enforced determinism:
  - Job env: `PYTORCH_DETERMINISTIC=1`, `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=":4096:8"`, `TORCH_ALLOW_TF32=0` (as applicable)
  - Runtime: `torch.use_deterministic_algorithms(True)`, TF32 disabled, highest matmul precision.
- Per-case artifact layout under `artifacts/${{ matrix.name }}/`:
  - `output.tiff`, `metrics.json`, `perf.json`, `env.json`, `run.log`, `nvidia_smi_before.txt`, `nvidia_smi_after.txt`.
- Replaced boolean gate with explicit numeric physics checks in CI:
  - `L2 <= 0.03`, `energy_err <= 0.01`, `airy_first_zero_err <= 0.02`, `strehl >= 0.95`.
  - Budgets: `vram_bytes < 4e9`, `wall_time_s < 90`.
- Docs job now real: MkDocs `mkdocs build --strict` and uploads built `site/`.
- Wheels build added: sdist/wheel built and uploaded from `dist/`.

Notes
- Discovery handles empty `examples/m3/` gracefully; matrix is empty if no files. Add cases to enable full gpu-smoke coverage.
- Placeholder run currently writes deterministic dummy `output.tiff`, `metrics.json`, and `perf.json`. Replace with the real CLI/runner when available.

