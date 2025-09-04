---
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/cli/main.py]
tests: [T-006, T-009]
gate: [format, lint, type, unit]
est: "≤2h"
status: COMPLETE
---

# OBJECTIVES/002_cli_scaffold.md

**Task**: `run|validate|inspect` subcommands. JSONL logging.

**Done**: 
- Implemented CLI with argparse in `src/microscope/cli/main.py`
- Added three subcommands: run, validate, inspect
- Integrated JSONL logging with structured output
- Added deterministic run support with seeds
- Created test coverage in `tests/test_cli.py`

**Outputs created**:
- `src/microscope/cli/main.py` - Main CLI implementation
- `src/microscope/core/logging.py` - Structured logging support
- `tests/test_cli.py` - CLI test suite

**Features**:
- `python -m microscope.cli run --config scene.yaml --out output/`
- `python -m microscope.cli validate`
- `python -m microscope.cli inspect --config scene.yaml`
- JSONL logging to `{output}/run.jsonl`
- Structured logging with timestamps and metadata

**Tests**: CLI produces outputs and logs paths; deterministic run with seed.
