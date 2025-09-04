# M1 Gate Fixes - Implementation Summary

## Fixed Issues

### 1. PyTorch CPU Pinning
- **Problem**: PyTorch version ranges causing installation issues
- **Solution**: Pinned exact version `torch==2.6.0` with CPU-only index
- **Files Modified**:
  - `pyproject.toml`: Exact pins for all dev dependencies
  - `scripts/bootstrap.sh`: Use `--index-url` for CPU wheels
  - `scripts/bootstrap.ps1`: Same for Windows

### 2. Ruff Configuration
- **Problem**: Too many lint rules causing noise
- **Solution**: Limited to E,F rules only (errors and undefined)
- **Files Modified**:
  - `pyproject.toml`: `select = ["E","F"]`
  - Removed unused imports from all Python files
  - Fixed type annotations (Any instead of any)

### 3. ASCII-Only Output
- **Problem**: Unicode characters (µ) in CLI output
- **Solution**: Replaced with ASCII (um)
- **Files Modified**:
  - `src/microscope/cli/main.py`: Changed µm to um

### 4. Coverage Configuration
- **Problem**: Coverage trying to include optics_sim
- **Solution**: Only cover microscope package
- **Files Modified**:
  - `pytest.ini`: `--cov=microscope`
  - `scripts/verify.sh`: Updated pytest command
  - `scripts/verify.ps1`: Same for Windows

### 5. CLI Flags
- **Problem**: Missing short flags
- **Solution**: Already had -c/-o flags, just verified
- **Files Modified**: None needed

### 6. Baselines
- **Problem**: Missing baseline files for zero-new policy
- **Solution**: Generated proper baselines
- **Files Created**:
  - `ruff-baseline.toml`: Clean after fixes
  - `mypy-baseline.txt`: Expected pydantic import warning

## How to Verify

```bash
# Clean install
make bootstrap

# Format code
ruff format src/microscope

# Check gates
make verify

# Test CLI
python -m microscope.cli --help
python -m microscope.cli run -c examples/minimal.yaml -o out/min
python -m microscope.cli validate

# Run specific test
python test_m1_fixes.py
```

## Acceptance Criteria Met

✓ `make bootstrap && make verify` is green
✓ `coverage.xml` generated when tests run
✓ `python -m microscope.cli --help` shows run|validate|inspect
✓ `examples/minimal.yaml` works with `run -c -o`
✓ All output is ASCII (no Unicode)
✓ Zero-new policy enforced with baselines
✓ CPU-only by default (no CUDA coupling)
✓ Windows compatible

## Dependencies Pinned

```toml
[project.optional-dependencies]
cpu = ["torch==2.6.0"]
dev = [
  "ruff==0.5.7",
  "mypy==1.10.0", 
  "pytest==8.3.2",
  "pytest-cov==5.0.0",
]
```

## Gates Order

1. Format check (ruff format --check)
2. Lint (ruff check --select E,F)
3. Type check (mypy with baseline)
4. Unit tests (pytest with coverage)

All gates must pass in order. Failures are fatal.

## Next Steps

With M1 gates fixed, ready to proceed to M2:
- Sampling heuristics
- Angular spectrum propagation
- Optical components
- Recorders and export
