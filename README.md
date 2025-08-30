optics-sim (Microscope v0.1)

Scaffolded Python 3.11 project for optical field simulation. Backend target: PyTorch (CUDA 12.1).

Install (editable):
- pip install -e .

CPU setup (Linux/macOS):
- pip install -e .[cpu,dev] -f https://download.pytorch.org/whl/cpu/torch_stable.html

CUDA setup (Windows/Ubuntu):
- pip install -e .[cuda,dev]

Pre-commit:
- python -m pip install pre-commit
- python -m pre_commit install
- python -m pre_commit run --all-files

Tests:
- python -m pytest -q              # CPU tests (non-GPU)
- python -m pytest -q -m gpu       # GPU smoke (requires CUDA)

Layout:
- src/optics_sim
- tests/
- examples/
- scripts/

License: MIT


