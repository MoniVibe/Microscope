optics-sim (Microscope v0.1)

![m3-ci](https://github.com/MoniVibe/Microscope/actions/workflows/m3_ci.yml/badge.svg?branch=main)

Scaffolded Python 3.11 project for optical field simulation. Backend target: PyTorch (CUDA 12.1).

Quickstart (CPU, Python 3.11):
- make bootstrap
- make verify

Manual install:
- pip install -e .
- pip install -e .[cpu,dev] -f https://download.pytorch.org/whl/cpu/torch_stable.html

CUDA setup (Windows/Ubuntu):
- pip install -e .[cuda,dev]

Tests:
- python -m pytest -q                           # CPU tests (non-GPU)
- python -m pytest -q -m gpu                    # GPU smoke (requires CUDA)

Single entrypoint:
- make verify  # runs format → lint → type → unit; uploads logs to reports/

Layout:
- src/optics_sim
- tests/
- examples/
- scripts/

Development
- Precision policy: see docs/precision_policy.md

License: MIT


