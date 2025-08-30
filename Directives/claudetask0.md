# Routing Note
Send **only the relevant section** to each agent. Do not share Claude’s section with Cursor or vice versa.

- **Project root directory**: `C:\Users\Moni\Documents\claudeprojects\Microscope`
- **Repo name**: `microscope` (package `optics_sim`, project name `optics-sim`)
- **Python**: 3.11
- **Backend**: PyTorch CUDA 12.1


# TASKS_CLAUDE.md — Core Implementation and Tests

## Goal
Implement minimal, correct, GPU-first modules to meet physics and engineering gates for v0.1.

## Environment assumptions
- Python 3.11, PyTorch 2.3 CUDA 12.1, single NVIDIA GPU.
- Deterministic seeds in config. Eager mode. Mixed precision allowed for FFTs with FP32 accumulation.

## Work order
1) `core.config` → 2) `core.frames` → 3) `sources.gaussian_finite_band` → 4) `prop.plan` + `prop.samplers` → 5) solvers (`bpm_vector_wide`, `bpm_split_step_fourier`, `as_multi_slice`) → 6) `recorders` + `io.tiff` → 7) `validation` metrics + analytic cases.

## Mandatory APIs
### `core.config`
- `load(path: str) -> dict`
- `validate(cfg: dict) -> dict` → normalized `Cfg` dict with units in µm, ranges checked, RNG seeds captured.
- Enforce presence: `lambda_nm|min,max`, `NA_max`, `grid.target_px`, `recorders`, `components`, `sources`, `runtime.budget.vram_gb`, `runtime.budget.time_s`.
- Determinism: store `seed_tensor`, `seed_sampler` and environment snapshot.

### `core.frames`
- Right-handed world frame. Euler Z-Y-X. Translation in µm.
- `compose(euler_zyx: Tensor, t_um: Tensor) -> dict`
- `to_world(p_local: Tensor, T: dict) -> Tensor`
- `transform_grid(nx:int, ny:int, pitch_um:float, T:dict) -> Tensor`
- Round-trip local→world→local error < `1e-6` µm in tests.

### `sources`
- `base.Source`: `prepare(cfg, device)`, `emit(sample_idx) -> complex Tensor[Ny,Nx]` on source plane.
- `gaussian_finite_band`: inputs center λ, FWHM, angular σ; planner sets S samples. Converge to <2% L2 vs 2× denser spectrum on Standard preset.

### `prop.plan`
- `make_plan(cfg) -> Plan` computing Δx, Δy, Δz list; spectral and angle counts; guard bands; PML thickness; memory budget fit for 10 GB baseline. Enforce Δx rules: Δx ≤ λ_min/(2·NA_max); high-NA preset uses ≤ λ_min/(3·NA_max).

### `prop.samplers`
- `resample(field, from_pitch_um, to_pitch_um)` phase-preserving interpolation with anti-alias and Nyquist checks.

### `prop.solvers`
- Common: run on CUDA device; honor band-limit and PML mask; allow mixed precision for FFTs; accumulate in FP32.
- `bpm_vector_wide`: split-step with wide-angle vector correction; adaptive Δz capped by curvature proxy; stability guard.
- `bpm_split_step_fourier`: scalar wide-angle BPM using frequency-domain propagation.
- `as_multi_slice`: angular spectrum multi-slice with nonparaxial transfer `H(fx,fy,λ) = exp(i k z sqrt(1-(λfx)^2-(λfy)^2))`, evanescent clamp.
- Public: `run(field: Tensor, plan: Plan, sampler) -> Tensor`.

### `recorders`
- `add_plane(z_um: float, what: set[str])` with allowed: `{"intensity","complex","phase"}`
- `capture(state) -> dict[str, Tensor]`; shapes `[Ny,Nx]` or `[S,Ny,Nx]` when spectral samples present.

### `io.tiff`
- Write 32-bit TIFF stacks. Complex as two planes (real, imag). Embed metadata: units=µm, Δx/Δy/Δz, λ list, NA, seeds, config snapshot, commit hash, coordinate frame.

### `validation`
- `metrics.py`: L2 field error, energy conservation, Strehl, MTF cutoff.
- `cases.py`: analytic setups for Gaussian propagation, aperture diffraction, thin lens focus, phase grating orders; and a high-NA AS reference generator.

## Tests and gates (must pass)
- `test_frames.py`: round-trip < 1e-6 µm.
- `test_sources_gauss.py`: spectral convergence <2% L2.
- `test_solvers_gaussian_free_space.py`: L2 ≤3%, energy ≤1%.
- `test_lens_paraxial.py`: Strehl ≥0.95; MTF cutoff within 2% at low NA.
- `test_aperture_airy.py`: peak and first-zero within 2%.
- `test_grating_orders.py`: order power ratios within 3%.
- `test_io_shapes_meta.py`: exact array shapes; required metadata keys present.

## Performance and budgeting
- Respect presets: Standard, High-NA, Aggressive. Fit within 10 GB VRAM. Use streaming of z-planes and in-place ops when needed. Prefer TF32/FP16 for FFTs with FP32 accumulation; fall back to FP32 if error >2%.

## Done criteria
- All tests green locally on a CUDA GPU.
- CPU CI green for non-GPU tests.
- Example configs execute end-to-end producing TIFF stacks with correct metadata.

## Hints
- Build PML as a separable radial mask; audit absorbed energy ≤1%.
- Enforce band-limits in frequency domain based on NA and λ.
- Δz adaptation: cap by curvature and a CFL-like condition; record chosen steps in plan for reproducibility.

---

# Hand-off order
1) Cursor completes scaffold and CI (**M0**). Provide repo URL or commit hash.
2) Claude implements modules and tests (**M1–M3**) and shares metrics.
3) Cursor wires examples, docs, packaging, and release tag (**M4**).

# Acceptance summary
- Repo builds and installs from project root.
- Pre-commit and CI pass.
- GPU smoke test passes core solvers.
- TIFF outputs deterministic with matching metadata.