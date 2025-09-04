# M1 Vertical Slice Implementation

## Summary
Implements M1 milestone: Package layout, CLI scaffold, and config models for the Microscope optical simulation package.

## Changes
- Created `src/microscope/` package structure per HILEVEL specification
- Implemented CLI with run/validate/inspect subcommands
- Added Pydantic configuration models with YAML/JSON I/O
- Created test suite for CLI and config round-trip
- Added example configuration file

## Brief/Spec Links
- Vision: `VISION.md`
- High-level design: `docs/HILEVEL.md`
- Goals: `docs/GOALS.md`
- Objectives: `OBJECTIVES/001_package_layout.md`, `OBJECTIVES/002_cli_scaffold.md`, `OBJECTIVES/003_config_models.md`

## Test Evidence
```bash
# All tests passing:
pytest tests/test_cli.py tests/test_config_roundtrip.py -v
# Result: 13 passed

# CLI works:
python -m microscope.cli --help
# Shows: run, validate, inspect subcommands

# Config loading works:
python -c "from microscope.core.config import load_config; s = load_config('examples/minimal.yaml'); print(f'Loaded: {len(s.sources)} sources')"
# Output: Loaded: 1 sources
```

## Reproduce Script
```bash
# Clone and setup
git clone https://github.com/MoniVibe/Microscope.git
cd Microscope

# Bootstrap (Windows)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .[dev]
python -m pip install pydantic pyyaml

# Verify
python -m microscope.cli --help
pytest tests/test_cli.py tests/test_config_roundtrip.py -v
python test_m1.py
```

## Checklist
- [x] Package layout matches HILEVEL specification
- [x] CLI entry point works: `python -m microscope.cli`
- [x] All three subcommands implemented (run, validate, inspect)
- [x] Pydantic models for all required components
- [x] YAML/JSON round-trip serialization works
- [x] Test coverage for new code
- [x] Example configuration loads successfully
- [x] No import-time CUDA coupling
- [x] Documentation updated in OBJECTIVES/

## Notes
- This is a vertical slice focusing on structure and interfaces
- Physics implementations are stubs for M1 (to be completed in M2-M4)
- Zero-new policy: baseline files prepared for ruff and mypy
- Compatible with existing `optics_sim` package structure
