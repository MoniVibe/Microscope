Microscope v0.1 — Cursor AI Progress Report (2025-08-30)

Summary
- Initialized repo scaffold at project root with packaging, linting, typing, and CI.
- Implemented `src/optics_sim` package with minimal stubs matching blueprint paths.
- Added tests skeleton; local CPU tests pass (4 passed).
- Added docs skeleton, README, examples, and utility script.
- Pre-commit installed and hooks run clean.

Key changes
- pyproject.toml with setuptools build, Python 3.11, deps (numpy, tifffile).
- .pre-commit-config.yaml with Ruff and basic hooks; .ruff.toml; mypy.ini.
- GitHub Actions CI: lint, CPU tests, placeholder GPU job.
- Source stubs under `src/optics_sim`: core/config & frames; sources (base, gaussian); components (thin/thick lens, aperture, grating, stubs); prop plan/samplers/solvers stubs; recorders; io/tiff; validation metrics; runtime stubs; logging.
- Tests: imports, frames, source, solver identity; `conftest.py` seeds RNG and adds gpu marker.
- Docs/Examples: README, docs/CONFIG.md, examples for low/mid/high NA.
- Script: scripts/print_env.py.

Local status
- Editable install: OK.
- pytest -q: 4 passed in ~0.17s.
- pre-commit hooks: Passed.

Next steps (for Claude)
- Implement real config parsing/validation and unit normalization.
- Fill propagation planner logic (grid, Δz, sampling, PML sizing, budgets).
- Implement solvers: vector BPM, split-step Fourier, angular-spectrum multi-slice.
- Implement component physics and proper coordinate transforms.
- Add TIFF writer metadata per blueprint and recorder integration.
- Expand tests to analytic validations and metrics.

Update (GPU CI and stubs verification)
- Added GPU smoke test `tests/test_gpu_smoke.py` with `@pytest.mark.gpu` on tiny tensor; CI runs it on self-hosted GPU.
- Added `dev` extra in `pyproject.toml`; CI now installs `.[cpu,dev]` and `.[cuda,dev]`.
- Made torch imports optional in `core.frames`, `core.config`, and sources to keep CPU CI green.
- Simplified `sources/gaussian_finite_band.py` to a numpy-based stub to satisfy current tests; real torch-backed model to follow.
- Verified stubs exist for all modules listed in the blueprint and `cursortask0.md`; paths and names match exactly.

GPU CI status
- Commit: b94cb53
- Status: self-hosted GPU job pending in CI; provide URL/screenshot of green run when available. M0 closes upon first passing GPU run referencing commit b94cb53 (or newer) with unchanged GPU smoke.


