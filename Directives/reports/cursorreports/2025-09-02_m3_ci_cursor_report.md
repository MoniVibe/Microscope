### Cursor CI Wiring Report: M3 runner, gates, wheels, docs

- Branch/PR: `m3/ci-wire-and-release` — Wire M3 runner in CI, enforce gates, build docs and wheels.
- gpu-smoke:
  - Matrix discovery reads `examples/m3/*.yml,*.yaml`; hard-fails if fewer than 3.
  - Determinism: TF32 disabled; `torch.use_deterministic_algorithms(True)`; `torch.set_float32_matmul_precision("highest")` and assertion; env flags set.
  - Runner: `python -m optics_sim.cli.m3_run --config ${{matrix.cfg}} --output artifacts/${{matrix.name}} --device cuda`.
  - Artifacts per case (30d): `output.tiff`, `metrics.json`, `perf.json`, `env.json`, `run.log`, `nvidia_smi_before.txt`, `nvidia_smi_after.txt`.
  - Gates enforced per case:
    - `L2|l2_error <= 0.03`, `energy_err|energy_error <= 0.01`, `airy_first_zero_err|airy_first_zero_error <= 0.02`, `strehl|strehl_ratio >= 0.95`, `mtf_cutoff_err|mtf_cutoff_error <= 0.05`.
    - Budgets: `peak_vram_bytes < 4e9`, `wall_time_sec < 90`.
    - Anti-dummy: `env.gpu_name` present; `nvidia_smi_before.txt` non-empty; `peak_vram_bytes >= 1e7`; `wall_time_sec >= 1`.
- Build-and-docs:
  - Build wheel + sdist; import-smoke from built wheel; docs via MkDocs `--strict`.
  - Uploads: `dist/*` and `site` (30d retention).
- CI report:
  - Job aggregates per-case metrics (L2, energy, Airy0, Strehl, MTF) plus VRAM/time and posts a PR comment; also uploads `ci_report.md` as artifact.

Acceptance readiness: gpu-smoke runs on ≥3 configs; gates/budgets enforced; wheels/docs artifacts uploaded; PR comment summarizes results with links.
