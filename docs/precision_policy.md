# Precision Policy Documentation

## Overview
This document describes the precision policy for the optics simulation library, effective as of M1 completion.

## Policy Rules

### 1. CUDA Tensors
- **All CUDA tensors MUST be FP32** (float32 for real, complex64 for complex)
- No exceptions during propagation, FFT operations, or phase calculations
- Enforced via `enforce_fp32_cuda()` and `assert_fp32_cuda()` functions

### 2. CPU Tensors
- **CPU tensors MAY use FP64** for intermediate calculations
- Angular Spectrum solver specifically maintains FP64 intermediates on CPU for accuracy
- Final outputs are cast to complex64 for consistency

### 3. Mixed Precision
- **DISABLED until M2 sign-off**
- Global flag `MIXED_FFT = False` must not be changed
- No FP16/TF32 operations allowed before M2 validation

## Implementation

### Enforcement Functions

```python
from optics_sim.core.precision import (
    enforce_fp32_cuda,    # Convert CUDA tensors to FP32
    assert_fp32_cuda,     # Assert tensor is FP32 on CUDA
    fft2_with_precision,  # FFT wrapper with precision enforcement
)
```

### Solver Behavior

| Solver | CPU Precision | CUDA Precision |
|--------|--------------|----------------|
| BPM Vector Wide | FP32/64 allowed | FP32 only |
| BPM Split-step | FP32/64 allowed | FP32 only |
| Angular Spectrum | FP64 intermediates | FP32 only |

### Code Patterns

#### Propagation Functions
```python
def propagate_field(field, plan):
    device = field.device
    
    # Enforce FP32 on CUDA
    if device.type == "cuda":
        field = field.to(torch.complex64)
        assert_fp32_cuda(field, "input")
    
    # ... propagation logic ...
    
    # Ensure output is FP32 on CUDA
    if device.type == "cuda":
        output = enforce_fp32_cuda(output, "output")
    
    return output
```

#### FFT Operations
```python
# Use wrapper for automatic precision enforcement
spectrum = fft2_with_precision(field, inverse=False)
field_back = fft2_with_precision(spectrum, inverse=True)
```

## Validation

### Test Coverage
- `test_precision_policy.py` validates:
  - FP32 enforcement on CUDA
  - FP64 preservation on CPU for AS solver
  - CPU vs GPU parity within tolerance gates
  - Mixed precision is disabled

### Tolerance Gates
- L2 error ≤3% for all solvers
- Energy conservation ≤1%
- CPU-GPU difference ≤1%

## Future Work (M2)

After M2 sign-off, mixed precision may be enabled:
- Set `MIXED_FFT = True` (currently False)
- Enable TF32/FP16 for FFT operations
- Maintain FP32 accumulation for stability

**DO NOT ENABLE BEFORE M2 VALIDATION**

## Quick Reference

### Dtype by Device
| Device | Real | Complex | Notes |
|--------|------|---------|-------|
| CUDA | float32 | complex64 | Strictly enforced |
| CPU | float64* | complex128* | *FP64 allowed for AS intermediates |

### Validation Commands
```bash
# Run precision policy tests
python tests/test_precision_policy.py

# Run full validation suite
python run_validation_suite.py
```

## Compliance Checklist

When modifying propagation code:
- [ ] Use `enforce_fp32_cuda()` on CUDA inputs
- [ ] Use `assert_fp32_cuda()` before/after FFTs
- [ ] Use `fft2_with_precision()` wrapper
- [ ] Cast outputs to complex64 on CUDA
- [ ] Verify `MIXED_FFT = False`
- [ ] Run precision policy tests
- [ ] Check CPU-GPU parity

---
Last Updated: 2025-01-01
Status: Enforced for M1