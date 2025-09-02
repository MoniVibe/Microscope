# Metrics Documentation

## Overview
This document describes the optical performance metrics computed by the M3 metrics module.

## Metric Definitions

### Energy Conservation
- **Canonical key**: `energy_error`
- **CI synonym**: `energy_err`
- **Description**: Relative energy difference between input and output fields
- **Threshold**: ≤ 1% (0.01)
- **Formula**: `|1 - E_out/E_in|`

### L2 Error
- **Canonical key**: `l2_error`
- **CI synonym**: `L2`
- **Description**: Normalized L2 difference between fields or vs. theory
- **Threshold**: ≤ 3% (0.03)
- **Formula**: `sqrt(mean((I_out - I_ref)^2))`

### Strehl Ratio
- **Canonical key**: `strehl_ratio`
- **CI synonym**: `strehl`
- **Description**: Ratio of peak intensity to ideal (diffraction-limited) peak
- **Threshold**: ≥ 0.95
- **Range**: 0-1 (1 = perfect)

### Airy First Zero Error
- **Canonical key**: `airy_first_zero_error`
- **CI synonym**: `airy_first_zero_err`
- **Description**: Relative error in Airy disk first minimum position
- **Threshold**: ≤ 2% (0.02)
- **Theory**: First zero at 0.61λ/NA

### MTF Cutoff Error
- **Canonical key**: `mtf_cutoff_error`
- **CI synonym**: `mtf_cutoff_err`
- **Description**: Relative error in MTF cutoff frequency
- **Threshold**: ≤ 2% (0.02)
- **Theory**: Cutoff at 2NA/λ

## Usage

### Python API
```python
from optics_sim.metrics import calculate_m3_metrics

metrics = calculate_m3_metrics(
    field=output_field,
    wavelength=550e-9,
    NA=0.65,
    dx=0.1e-6,
    reference_field=input_field
)

# Access with canonical keys
energy_err = metrics['energy_error']
strehl = metrics['strehl_ratio']

# Or use CI synonyms
energy = metrics['energy_err']
strehl_alt = metrics['strehl']
```

### Validation Gates
All metrics must pass their thresholds for acceptance:
- Energy conservation within 1%
- L2 error within 3%
- Strehl ratio above 0.95
- Airy first zero within 2% of theory
- MTF cutoff within 2% of theory

## Implementation Notes

### Precision Policy
- All metrics use FP32 (complex64) on CUDA
- CPU calculations may use FP64 intermediates
- FFT operations use `norm='ortho'` for consistency

### Dual Key Support
Every metric provides two keys for compatibility:
1. **Canonical key**: Full descriptive name (e.g., `energy_error`)
2. **CI synonym**: Short form for CI/testing (e.g., `energy_err`)

Both keys point to the same value in the returned dictionary.

---
Last Updated: 2025-01-29
Status: M3 Compliant