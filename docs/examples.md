# M3 Examples

## Overview

The M3 pipeline provides three example configurations demonstrating different optical propagation scenarios with strict FP32 precision and deterministic execution.

## Quick Start

```bash
# Navigate to project root
cd /path/to/Microscope

# Run with default output directory
python -m optics_sim.cli.m3_run --config examples/m3/as_airy.yaml

# Specify custom output directory (--out or --output)
python -m optics_sim.cli.m3_run --config examples/m3/as_airy.yaml --out results/test1

# Run on specific device
python -m optics_sim.cli.m3_run --config examples/m3/as_airy.yaml --device cuda --out output/gpu_run

# Enable verbose logging
python -m optics_sim.cli.m3_run --config examples/m3/as_airy.yaml --verbose --out debug/run1
```

## Available Examples

### 1. Angular Spectrum - Airy Pattern (`as_airy.yaml`)

Demonstrates free-space propagation and Airy disk formation.

**Configuration:**
- Solver: Angular Spectrum
- NA: 0.5
- Grid: 1024×1024
- Wavelength: 550 nm
- Distance: 10 mm

**Run:**
```bash
python -m optics_sim.cli.m3_run \
    --config examples/m3/as_airy.yaml \
    --out output/as_airy \
    --device cuda
```

**Expected Metrics:**
- L2 error: ≤ 2%
- Energy error: ≤ 0.5%
- Strehl ratio: ≥ 0.98
- First zero error: ≤ 2%

### 2. Split-Step BPM - Phase Grating (`bpm_ss_grating.yaml`)

Simulates diffraction through a binary phase grating.

**Configuration:**
- Solver: Split-Step BPM
- Wavelength: 632.8 nm (HeNe)
- Grid: 512×512
- Grating period: 20 μm
- Distance: 50 mm

**Run:**
```bash
python -m optics_sim.cli.m3_run \
    --config examples/m3/bpm_ss_grating.yaml \
    --out output/grating \
    --device cuda
```

**Expected Metrics:**
- Energy error: ≤ 0.8%
- Zero-order efficiency: ~40%
- First-order efficiency: ~25%

### 3. Wide-Angle BPM - High-NA Focus (`bpm_wa_highNA.yaml`)

Models high-NA focusing in water immersion.

**Configuration:**
- Solver: Wide-Angle BPM
- NA: 0.85
- Medium: Water (n=1.33)
- Grid: 768×768
- Wavelength: 488 nm
- Focal length: 5 mm

**Run:**
```bash
python -m optics_sim.cli.m3_run \
    --config examples/m3/bpm_wa_highNA.yaml \
    --out output/highNA \
    --device cuda
```

**Expected Metrics:**
- L2 error: ≤ 2.5%
- Strehl ratio: ≥ 0.96
- FWHM: ~350 nm
- Encircled energy (50%): ≥ 80%

## Output Files

Each run produces the following files in the output directory:

### `output.tiff`
- Intensity distribution
- Metadata includes grid parameters, wavelength, NA, solver type

### `metrics.json`
Performance metrics with both canonical and CI-compatible keys:
```json
{
  "l2_error": 0.02,
  "L2": 0.02,
  "energy_error": 0.005,
  "energy_err": 0.005,
  "airy_first_zero_error": 0.015,
  "airy_first_zero_err": 0.015,
  "strehl_ratio": 0.97,
  "strehl": 0.97,
  "mtf_cutoff_error": 0.03,
  "mtf_cutoff_err": 0.03
}
```

### `perf.json`
Performance statistics:
```json
{
  "wall_time_sec": 45.2,
  "solver_time_sec": 38.1,
  "peak_vram_bytes": 3758096384,
  "peak_vram_gb": 3.5
}
```

### `env.json`
Environment information:
```json
{
  "torch_version": "2.0.1+cu118",
  "cuda_available": true,
  "cuda_version": "11.8",
  "cudnn_version": 8700,
  "gpu_name": "NVIDIA GeForce RTX 3080",
  "deterministic": true,
  "seed": 1337
}
```

## Command-Line Options

| Option | Aliases | Description | Default |
|--------|---------|-------------|---------|
| `--config` | `-c` | Path to YAML configuration file | Required |
| `--output` | `--out`, `-o` | Output directory for results | `output` |
| `--device` | `-d` | Compute device (`cuda` or `cpu`) | `cuda` |
| `--verbose` | `-v` | Enable detailed logging | `false` |

## Performance Guidelines

### GPU Requirements
- Minimum: 4 GB VRAM
- Recommended: 8 GB VRAM
- All examples designed to run on consumer GPUs

### Expected Performance
- Wall time: < 90 seconds per run
- Peak VRAM: < 4 GB
- Solver time: 30-60 seconds (depends on grid size)

## Validation Gates

All examples must pass these metric gates:

| Metric | Threshold | Type |
|--------|-----------|------|
| L2 error | ≤ 3% | Maximum |
| Energy error | ≤ 1% | Maximum |
| First zero error | ≤ 2% | Maximum |
| Strehl ratio | ≥ 0.95 | Minimum |
| MTF cutoff error | ≤ 5% | Maximum |

## Troubleshooting

### CUDA Not Available
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Run on CPU instead
python -m optics_sim.cli.m3_run --config examples/m3/as_airy.yaml --device cpu
```

### Out of Memory
- Reduce grid size in configuration
- Ensure no other GPU processes are running
- Use `nvidia-smi` to check GPU memory usage

### Determinism Issues
- Verify seed is set to 1337 in config
- Check that `torch.backends.cudnn.deterministic = True`
- Ensure no random operations outside controlled seeds

## Advanced Usage

### Batch Processing
```bash
#!/bin/bash
for config in examples/m3/*.yaml; do
    name=$(basename $config .yaml)
    python -m optics_sim.cli.m3_run \
        --config $config \
        --out results/$name \
        --device cuda
done
```

### Custom Configurations
Create your own YAML configuration following the schema:
```yaml
solver:
  type: angular_spectrum  # or split_step_bpm, wide_angle_bpm
  params:
    wavelength: 550.0e-9
    NA: 0.5
    
grid:
  N: 1024
  dx: 1.0e-6
  
propagation:
  distance: 10.0e-3
  
device: cuda
seed: 1337
```

## CI Integration

For continuous integration, ensure all artifacts are generated:
- `output.tiff` - Image output
- `metrics.json` - Performance metrics
- `perf.json` - Timing and memory stats
- `env.json` - Environment snapshot
- `run.log` - Execution log (if logging to file)
- `nvidia_smi_*.txt` - GPU state (optional)

Example CI command:
```bash
python -m optics_sim.cli.m3_run \
    --config examples/m3/as_airy.yaml \
    --out ci_artifacts/run_${BUILD_ID} \
    --device cuda \
    --verbose 2>&1 | tee ci_artifacts/run_${BUILD_ID}/run.log
```
