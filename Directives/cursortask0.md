Use this prompt in Cursor or Claude:

Role: Senior refactor engineer. Apply surgical, reversible changes. No feature work.

Goal: Standardize and organize the optics simulation repo so it passes local gates and CI and is easy to extend.

Context: Target Python 3.11. Missing torch blocks tests. ~1k ruff issues dominated by naming and complexity. Adopt baselines, enforce “no new issues,” then pay down. 

Standards to enforce

Repo docs flow: VISION.md → HILEVEL.md → GOALS.md → OBJECTIVES/*.

Each objective file: owner, inputs, outputs, test, gate.

Invariants: pinned deps, deterministic tooling, idempotent scripts, single verify entrypoint, no red to GitHub.

Gates order: format → lint → type → unit → integration. Block commit/merge on fail.

CI must mirror local. Upload coverage and gate logs. Small PRs only.

Deliverables

Repo hygiene

README.md with quickstart and one-command verify for Python 3.11 CPU.

VISION.md placeholder with TODO section, and stubs for docs/HILEVEL.md, docs/GOALS.md, OBJECTIVES/README.md + sample OBJECTIVES/000_bootstrap.md.

CONTRIBUTING.md describing gates, PR checklist, rollback recipe, reproduce script pattern.

.editorconfig, .gitignore, CODEOWNERS (default owner Opus).

Move code to src/<package>/...; ensure tests/ mirrors layout.

Environment and pins

Pin Python to 3.11 via .python-version and CI matrix.

Use pyproject.toml with locked constraints. Add extras: cpu, cuda, dev.

Prefer pip-tools or uv. Generate frozen lock artifacts (requirements.{cpu,dev}.txt or uv.lock). No unpinned ranges.

Tooling config

Ruff: enable format. Create ruff.toml with rule set, extend-exclude, and a ruff-baseline.toml generated from current tree. Policy: zero new lints.

Mypy: mypy.ini strict flags, plus mypy-baseline.txt created from current errors. Policy: zero new type errors.

Pytest: pytest.ini with -q, addopts = -ra --cov=<package> --cov-report=xml, markers gpu and slow. Ensure CPU tests skip GPU paths when CUDA not present.

Scripts

scripts/verify.sh running: ruff format check → ruff check → mypy → pytest. Write logs to reports/{format,lint,types,tests}.log. Exit non-zero on first failure.

scripts/bootstrap.sh that creates .venv for 3.11 and installs .[cpu,dev] from the PyTorch CPU index.

Makefile targets: bootstrap, verify, format, lint, types, test, clean.

CI (GitHub Actions)

Single workflow .github/workflows/ci.yml: Python 3.11 on ubuntu. Cache pip. Install .[cpu,dev] from CPU index. Run make verify. Always upload coverage.xml and reports/*. Set required checks to block merge.

Naming and imports

Run ruff --fix for safe rules. Convert magic numbers to named consts where trivial. Remove unused imports/vars. Keep function signatures stable.

Do not churn public API names without a deprecation shim. Add __all__ in public modules.

Types and errors

Add return and param types to public functions. Prefer dataclasses or pydantic models for config. Centralize error types under <package>/errors.py.

Tests

Ensure CPU test path does not import CUDA-only modules at import time. Skip GPU tests when CUDA absent. Add minimal smoke tests for core APIs.

Set coverage threshold to 70% now. Raise later.

One entrypoint

Add python -m <package>.cli with subcommands for common tasks. Document CLI flags and logging. Default noisy logs behind --verbose.

PR policy scaffolding

Add .github/pull_request_template.md: link brief/spec, test evidence, changelog, rollback, reproduce script.

Acceptance criteria

make bootstrap && make verify succeeds on a clean clone with Python 3.11 CPU.

CI green with artifacts uploaded. Coverage reported.

ruff check --statistics only reports baseline items. New code is clean.

mypy only reports baseline items. New code typed.

Repository contains the docs scaffold and objective template.

Tests run CPU-only without torch.cuda present. GPU tests are marked and skipped when unavailable.

Execution order

Create docs scaffold and repo structure.

Configure pyproject, lock deps, add extras.

Add ruff, mypy, pytest configs. Generate baselines.

Implement scripts and Makefile. Wire verify.

Apply safe autofixes. Minimal renames only where mechanical.

Add types to public surfaces. Centralize errors.

Fix import-time GPU coupling. Ensure CPU tests run.

Add CI workflow. Require checks.

Non-goals

No feature additions. No algorithm rewrites. No version upgrades beyond pins needed for PyTorch CPU on 3.11.

Notes for this repo

Use Python 3.11. Install torch from CPU index for tests. Keep GPU tests marked and skipped by default. Generate baselines first, then ratchet.