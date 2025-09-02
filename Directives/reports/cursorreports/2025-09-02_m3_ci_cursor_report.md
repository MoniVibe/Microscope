## M3 CI run report — 2025-09-02

Summary
- CPU baseline: tests passed, validation suite failed. Halting before GPU and build+docs per gate order.

CPU Baseline
- pytest tests/test_precision_policy.py: PASS (JUnit: pytest_precision.xml)
- run_validation_suite.py: FAIL

Validation failures (key excerpts)
```
1. GAUSSIAN FREE SPACE PROPAGATION
✗ BPM Vector Wide: L2 error 115.453% exceeds 3%
✓ Split-step Fourier: L2 ≤3%, Energy ≤1%
✓ Angular Spectrum: L2 ≤3%, Energy ≤1%

2. AIRY PATTERN (APERTURE DIFFRACTION)
✗ Airy pattern: First zero position error 10.6% exceeds 2%

3. THIN LENS FOCUSING (PARAXIAL)
✗ Thin lens: Strehl ratio 0.045 < 0.95 for paraxial lens

4. PHASE GRATING DIFFRACTION ORDERS
✗ Grating orders: Order -2 efficiency error 4.9% exceeds 3%

6. TIFF I/O WITH METADATA
✗ TIFF I/O: Complex should have 2 planes

Summary: OVERALL: 4/9 tests passed (bpm_gaussian, airy, lens, grating, tiff failed)
```

Artifacts (CPU gate)
- Generated: pytest_precision.xml, validation_suite.log
- Not applicable: GPU artifacts (not executed), dist/* (not executed)

Stop reason
- Gate order requires CPU baseline to pass first. Validation suite failed several numeric thresholds.

Next steps (proposed)
- Investigate BPM Vector Wide parameters/units leading to high L2.
- Review Airy first-zero calculation and lens Strehl computation; ensure precision policy and dtype consistency across solvers.
- Fix TIFF complex-plane layout expectations in tests or writer.
- Re-run CPU gate; proceed to GPU smoke and build once green.

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
