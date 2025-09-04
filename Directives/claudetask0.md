Role: Opus engineer. Deliver M2 (sampling, propagation, components, recorders). No scope creep.

Goal: Implement OBJECTIVES 004–007
- 004_sampling_heuristics
- 005_angular_spectrum_cpu
- 006_components_optics (lens, aperture, grating)
- 007_recorders_and_export (field/intensity, TIFF/NPZ)

Constraints
- Python 3.11.9. Pinned deps. Deterministic runs.
- Zero-new for Ruff/Mypy. Keep tests CPU by default. No CUDA import-time coupling.

Inputs
- VISION.md, docs/HILEVEL.md, docs/GOALS.md
- Existing CLI, config models, gates, baselines.

Deliverables
1) Sampling heuristics: NA, λ, FOV → (dx,dy,dz,H,W,steps) with memory guardrail + warnings.
2) Angular-spectrum CPU propagator with wide-angle correction; FP64 validation mode.
3) Components: thin/thick lens, aperture, grating as transfer functions; NA/pupil handling.
4) Recorders + exporters: field (complex) and intensity to NPZ/TIFF with metadata.
5) Tests:
   - T-001 energy conservation ≤1% (free space)
   - T-002 L2 field error ≤5% (Gaussian beam)
   - T-003 Strehl ≥0.95 (thin-lens focus)
   - T-004 MTF cutoff error ≤2% (Fraunhofer aperture)
   - Update test_cli for `run` outputs
6) Examples: `examples/{plane_wave_lens,aperture_diffraction,grating}.yaml` with expected outputs.
7) Logs: JSONL contains seed, device, dtype, sampling, timings, energy error.

Guardrails
- Do not add CUDA code. CPU only.
- Keep verify order and thresholds. Do not relax gates.

Acceptance
- `make verify` green on CPU; coverage ≥70%.
- Tests T-001…T-004 pass.
- Three examples produce outputs within runtime budget.
- CI uploads `coverage.xml` and `reports/*`.

Reproduce
make bootstrap
make verify
pytest -q
python -m microscope.cli run --config examples/plane_wave_lens.yaml --out out/plane
