# Routing Note
Send **only the relevant section** to each agent. Do not share Claude’s section with Cursor or vice versa.

- **Project root directory**: `C:\Users\Moni\Documents\claudeprojects\Microscope`
- **Repo name**: `microscope` (package `optics_sim`, project name `optics-sim`)
- **Python**: 3.11
- **Backend**: PyTorch CUDA 12.1

---

# TASKS_CURSOR.md — Integration, Scaffold, CI, Hygiene

## Goal
Create a clean, reproducible repository at the project root, wire CI, and provide stubs, docs, and examples so Claude can implement.

## Deliverables
1) **Repo scaffold** at `C:\Users\Moni\Documents\claudeprojects\Microscope` with:
```
microscope/
  pyproject.toml
  src/optics_sim/...
  tests/...
  examples/
  docs/
  scripts/
  .github/workflows/ci.yml
  .pre-commit-config.yaml
  .ruff.toml
  mypy.ini
  README.md
  LICENSE
```

2) **`pyproject.toml`** (editable install; Ruff as formatter; strict typing):
```
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "optics-sim"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
  "numpy>=1.26",
  "tifffile>=2024.5.22",
  "typing-extensions>=4.10",
]

[project.optional-dependencies]
# Install one of these manually depending on environment
cpu = ["torch==2.3.*+cpu; platform_system != 'Windows'",]
cuda = ["torch==2.3.*"]

[tool.setuptools]
package-dir = {"" = "src"}
packages = ["optics_sim"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E","F","I","UP","B","N","W","PL","PYI"]

[tool.mypy]
python_version = "3.11"
strict = true
warn_unused_ignores = true
```

3) **Pre-commit** `.pre-commit-config.yaml`:
```
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: v0.6.9
  hooks:
  - id: ruff
    args: [--fix]
  - id: ruff-format
- repo: https://github.com/pre-commit/pre-commit-hooks
  rev: v4.6.0
  hooks:
  - id: check-yaml
  - id: end-of-file-fixer
  - id: trailing-whitespace
```

4) **GitHub Actions** CI `.github/workflows/ci.yml`:
```
name: ci
on: [push, pull_request]
jobs:
  lint:
    runs-on: ubuntu-22.04
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: {python-version: '3.11'}
    - run: pipx install pre-commit
    - run: pre-commit run --all-files

  test-cpu:
    runs-on: ubuntu-22.04
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: {python-version: '3.11'}
    - run: pip install -e .[cpu] -f https://download.pytorch.org/whl/cpu/torch_stable.html
    - run: pip install pytest
    - run: pytest -q -m "not gpu" --maxfail=1

  test-gpu:
    runs-on: [self-hosted, gpu]
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v5
      with: {python-version: '3.11'}
    - run: pip install -e .[cuda]
    - run: pip install pytest
    - run: python -c "import torch; assert torch.cuda.is_available()"
    - run: pytest -q -m gpu --maxfail=1
```

5) **Source tree stubs** (signatures only; minimal logic):
```
src/optics_sim/__init__.py
src/optics_sim/core/__init__.py
src/optics_sim/core/config.py
src/optics_sim/core/frames.py
src/optics_sim/sources/__init__.py
src/optics_sim/sources/base.py
src/optics_sim/sources/gaussian_finite_band.py
src/optics_sim/components/__init__.py
src/optics_sim/components/{lens_thin.py,lens_thick.py,aperture.py,phase_grating.py,mirror_stub.py,bs_stub.py}
src/optics_sim/prop/__init__.py
src/optics_sim/prop/plan.py
src/optics_sim/prop/samplers.py
src/optics_sim/prop/solvers/__init__.py
src/optics_sim/prop/solvers/{bpm_vector_wide.py,bpm_split_step_fourier.py,as_multi_slice.py}
src/optics_sim/recorders/planes.py
src/optics_sim/io/tiff.py
src/optics_sim/validation/{cases.py,metrics.py}
src/optics_sim/runtime/{budget.py,checkpoints.py,resume.py}
src/optics_sim/logging/logs.py
```

6) **Tests skeleton** (import and placeholder assertions):
```
tests/test_imports.py
tests/test_frames.py
tests/test_sources_gauss.py
tests/test_solvers_gaussian_free_space.py
tests/test_lens_paraxial.py
tests/test_aperture_airy.py
tests/test_grating_orders.py
tests/test_io_shapes_meta.py

# pytest markers
tests/conftest.py  # seeds RNG; provides device() fixture choosing CUDA when present
```

7) **Docs skeleton**
- `README.md`: install guides for CPU and CUDA on Windows and Ubuntu; quickstart.
- `docs/CONFIG.md`: list all config keys with units and enums.

8) **Examples**
- `examples/low_na.yml`, `examples/mid_na.yml`, `examples/high_na.yml` with plausible placeholders.

9) **Scripts**
- `scripts/print_env.py`: print torch version, CUDA availability, device count, commit hash.

## Constraints
- Respect Python 3.11. Package root `src/optics_sim`.
- Default units µm; accept nm in config but normalize.

## Acceptance criteria
- `pip install -e .` works on the project root.
- Pre-commit passes locally and in CI.
- `pytest -q` on CPU CI is green for non-GPU tests.
- GPU job green on self-hosted runner, or locally via `-m gpu`.
- All stubs and tests exist, import, and match names and paths above.
- Examples run end-to-end once Claude lands solvers.

---



