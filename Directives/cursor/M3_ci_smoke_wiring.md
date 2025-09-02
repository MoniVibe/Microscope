### M3 CI smoke wiring — Cursor AI (GPT-5)

Goal
- Wire CI gpu-smoke to Claude’s real runner and forbid empty/dummy runs.

Implementation
- Matrix discovery: runtime scan of `examples/m3/*.yml|*.yaml`. CI fails if < 3 configs.
- GPU job:
  - Concurrency guard: `group: gpu-smoke`, `cancel-in-progress: false`.
  - Determinism: `PYTORCH_DETERMINISTIC=1`, TF32 disabled, highest matmul precision.
  - Runner call:
    - `python -m optics_sim.cli.run --config ${{ matrix.cfg }} --out artifacts/${{ matrix.name }}`
  - Per-case artifacts (30-day retention):
    - `output.tiff`, `metrics.json`, `perf.json`, `env.json`, `run.log`, `nvidia_smi_before.txt`, `nvidia_smi_after.txt`.
  - Gates (numeric, no booleans):
    - Physics: `L2 <= 0.03`, `energy_err <= 0.01`, `airy_first_zero_err <= 0.02`, `strehl >= 0.95`.
    - Budgets: `peak_vram_bytes < 4e9`, `wall_time_sec < 90`.
    - Anti-dummy: `peak_vram_bytes >= 10_000_000`, `wall_time_sec >= 1`, `env.gpu_name` present, `nvidia_smi_before.txt` non-empty.
- Docs: MkDocs strict build and upload of `site/`.
- Wheels: Build sdist/wheels and upload from `dist/`.

Acceptance
- gpu-smoke green on ≥3 real configs with required artifacts.
- Empty matrix or dummy outputs fail the job.
- Docs and wheels artifacts uploaded and readable.

