Microscope Progress Assessment — 2025-09-01

Audience: ChatGPT (planning/blueprints)

Scope since last report
- CI wired for lint, CPU, and self-hosted GPU jobs; artifacts captured for triage.
- GPU smoke example configs added and referenced in README; install docs aligned with extras.
- Solvers implemented at a baseline level across three paths: angular spectrum, split-step Fourier BPM, and wide-angle vector BPM.

Repo health snapshot
- Build/install: `pyproject.toml` present; editable install works; CPU/CUDA extras defined; strict mypy enabled; Ruff minimal rules enabled.
- CI: `.github/workflows/ci.yml` runs Ruff lint, CPU tests with JUnit XML, and GPU-marked tests on self-hosted GPU runner. Latest runs: [CI runs](https://github.com/MoniVibe/Microscope/actions?query=workflow%3Aci+branch%3Amain+sha%3Aea6c661f).
- Lint: 290 remaining issues (naming style N8xx, complexity PLR09xx, magic-values PLR2004 dominate). Many are low-risk hygiene; a subset require refactors.
- Tests (CPU): Local `pytest_cpu.xml` shows collection leaking into `Lib/site-packages` (e.g., colorama tests). Root cause: project contains a `Lib/` folder under repo root; pytest recurses into it on some invocations. CI limits to `tests/`, but to harden locally, add `norecursedirs = Lib` or `--ignore=Lib`.
- Tests (GPU): Self-hosted job executes `-m gpu` subset and uploads logs. Local machine lacks CUDA PyTorch; rely on runner for GPU coverage.

Implementation status (high level)
- Core: `core.frames` and `core.config` exist; frames support right-handed Z-Y-X Euler; config normalizes units and validates ranges.
- Planning: `prop.plan` computes grid, Δz, sampling presets; includes VRAM budgeting and mixed-precision toggles.
- Solvers: `prop/solvers/as_multi_slice.py` implements nonparaxial H(fx,fy,λ) with evanescent handling and soft NA band-limit; BPM split-step Fourier and wide-angle vector BPM stubs include PML and optional mixed precision.
- IO: `io.tiff` writes 32-bit stacks; embeds metadata; basic reader present for tests.
- Validation: Analytic cases and metrics provided (Gaussian propagation, Airy, thin lens, phase grating) to gate physics.

Known issues and risks
- CPU local test discovery pulls vendor tests due to `Lib/` folder under repo root; causes immediate collection error. Action: add `norecursedirs = Lib` in pytest config and/or pass `--ignore=Lib` in local scripts.
- Lint noise (naming/complexity) obscures signal. Action: expand Ruff rule selection gradually and/or add targeted ignores for test files and generated helpers until physics is green.
- Physics tolerances in `tests/test_aperture_airy.py` and `tests/test_grating_orders.py` may fail with current solver defaults (aliasing, normalization, spectral windowing). These need Claude’s focused tuning in solvers and validation methods.

Recommendations for ChatGPT directives
- Direct Claude (Opus) to:
  1) Close CPU physics gaps: Airy peak/first-zero within 2%; grating order efficiencies within 3% by fixing normalization, band-limits, and energy accounting in AS/BPM paths.
  2) Finalize `io.tiff` metadata coverage and add round-trip read tests gated on `tifffile` availability.
  3) Tighten `core.frames` API and tests for batch transforms; ensure dtype preservation and naming alignment.
  4) Reduce aliasing in far-field validations (padding/windowing) and ensure energy conservation checks are robust.
- Direct Cursor (this agent) to:
  - Harden pytest configuration against vendor directories (`Lib/`) and ensure local runs mirror CI.
  - Maintain filtered error logs on each run (lint/test) and publish to `Directives/reports/cursorreports/` per user preference [[memory:7705018]].
  - Keep CI green by scoping lint rules and incrementally enforcing style after physics gates are met.

Next milestones
- M1: CPU tests green for non-GPU suite; vendor test collection blocked; core physics gates met at current tolerances.
- M2: GPU smoke tests green on self-hosted runner across all three solvers; example configs produce TIFF stacks with correct metadata.

Key artifacts (latest)
- CI workflow: `.github/workflows/ci.yml`
- Lint summary: `Directives/reports/cursorreports/latest_lint.txt` (290 remaining after autofixes)
- CPU JUnit: `pytest_cpu.xml` (shows vendor test collection issue on local runs)
- GPU logs/env: `Directives/reports/cursorreports/pytest_gpu.log`, `env.txt` (from runner)
- Example configs: `examples/gpu_smoke_*.yml`

Notes
- Local environment files indicate no CUDA; rely on the self-hosted GPU runner for `-m gpu` coverage until a local CUDA setup is available.



