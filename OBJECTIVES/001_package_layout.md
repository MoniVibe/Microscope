---
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/*, tests/conftest.py]
tests: [T-006, T-011, T-012]
gate: [format, lint, type, unit]
est: "≤2h"
status: COMPLETE
---

# OBJECTIVES/001_package_layout.md

**Task**: Create package layout per HILEVEL. Add `__all__` in public modules. Add `python -m microscope.cli` entry.

**Done**: 
- Created `src/microscope/` package structure with all required modules
- Added `__all__` exports in all `__init__.py` files
- Created CLI entry point allowing `python -m microscope.cli`
- Package layout matches HILEVEL specification

**Outputs created**:
- `src/microscope/__init__.py`
- `src/microscope/cli/__init__.py` and `__main__.py`
- `src/microscope/core/__init__.py` and submodules
- `src/microscope/physics/__init__.py`
- `src/microscope/gpu/__init__.py`
- `src/microscope/io/__init__.py`
- `src/microscope/validate/__init__.py`
- `src/microscope/presets/__init__.py`

**Tests**: `pytest -q` runs discovery; `python -m microscope.cli --help` works.
