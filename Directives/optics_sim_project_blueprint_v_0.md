# Optics‑sim — Project Blueprint v0.1

> Canonical, high‑level reference for the project’s intent, scope, architecture, standards, and decision log. Stable across versions; implementation plans live in per‑release HLDs.

## 0. Identity
- **Repo**: `microscope`
- **Root path**: `C:\Users\Moni\Documents\claudeprojects\microscope`
- **Product name (public)**: Optics‑sim
- **Version policy**: SemVer. This blueprint targets **v0.1**.

## 1. Vision
Simulate coherent and partially coherent optical fields for microscope‑like systems on a single NVIDIA GPU with deterministic, reproducible outputs and CI‑verified accuracy. Emphasis on correctness, reproducibility, and developer ergonomics over maximal speed.

## 2. Outcomes and success metrics
- **Physical fidelity**: pass analytic gates (Gaussian propagation, aperture diffraction, lens focus, grating orders) within stated tolerances.
- **Determinism**: identical outputs given the same config, seeds, and version.
- **Usability**: CLI completes three provided example configs without manual tuning.
- **Performance**: complete target presets within VRAM/runtime budgets on RTX 3080 10 GB.
- **Sustainability**: clean plugin points for sources/components/solvers; >90% unit test pass on CPU CI; GPU smoke test green.

## 3. Non‑goals for v0.1
- Nonlinear optics, time‑domain propagation, fluorescence, scattering media modeling, thermal noise.
- Live GUI. Multi‑GPU or distributed execution. CPU performance tuning beyond basic correctness.
- Full polarization APIs exposed to users (kept as internal hooks only).

## 4. Primary users
- Research engineers prototyping optical stacks.
- Simulation engineers generating ground‑truth fields for downstream analysis.

## 5. Operating constraints
- **OS**: Windows 10/11 and Ubuntu 22.04+.
- **Language**: Python 3.11.
- **Backend**: PyTorch with CUDA 12.1 wheels; eager mode first; optional mixed precision for FFTs.
- **Hardware**: single NVIDIA GPU, 10 GB VRAM baseline.
- **Units**: micrometers for spatial dimensions; nanometers accepted at input and normalized.

## 6. Architecture (static)
```
optics_sim/
  core/          # config, frames (coords, transforms)
  sources/       # field emitters (spectral + angular sampling)
  components/    # lenses, apertures, phase elements, placement
  prop/          # planners, samplers, solvers (BPM/AS)
  recorders/     # capture planes and quantities
  io/            # TIFF writer/reader and metadata
  validation/    # analytic cases and metrics
  runtime/       # budgets, checkpoints, resume
  logging/       # structured logs
```
**Plugin points**: `sources`, `components`, `solvers` register via a lightweight registry. Lifecycle: `configure → prepare(device) → run(batch) → finalize`.

## 7. Architecture (dynamic dataflow)
**Config → Source Synth → Propagation Planner → Solver → Recorders → Output Writer → Validation**
- Config defines wavelengths, NA, components, recorders, budgets.
- Planner fixes grid, Δz list, spectral/angle samples, PML and guard bands under VRAM caps.
- Solvers execute per policy (see §8). Recorders capture intensity/phase/complex.
- Writer persists 32‑bit TIFF stacks with metadata and config snapshot.

## 8. Physics policy
- **Low NA (≤0.30)**: vector BPM with wide‑angle correction.
- **Mid NA (0.30–0.70)**: wide‑angle BPM + split‑step Fourier with spectral band‑limits.
- **High NA (>0.70)**: angular‑spectrum multi‑slice, short Δz, nonparaxial transfer.
- **Boundaries**: PML enabled by default; guard‑band padding sized from NA and λ.
- **Sampling rules**: Δx ≤ λ_min/(2·NA_max) target; Δx ≤ λ_min/(3·NA_max) for high‑NA presets.

## 9. Performance and budgets
Preset targets on RTX 3080 10 GB:
- **Standard (NA ≤0.3)**: 1024–1536² grid, 128–192 planes, 1–3 spectral × 1–5 angle samples, <6 GB, ≤3–5 min.
- **High‑NA (0.3–0.7)**: 1536–2048², 192–288 planes, 3–5 × 5–9, <9 GB, ≤5–7 min.
- **Aggressive (>0.7)**: 2048–2560², 256–384 planes, 5–7 × 7–13, ~10 GB, ≤7–9 min.
Adaptive reductions use error estimators targeting ΔL2 <2% for spectra and stable Δz.

## 10. I/O and metadata
- **Format**: 32‑bit TIFF stacks. Complex fields stored as two planes (real, imag). Optional intensity and phase planes.
- **Metadata**: units, Δx/Δy/Δz, λ list, NA, seeds, config snapshot, commit hash, coordinate frame.
- **Compression**: none by default for speed and reproducibility.

## 11. Validation and quality gates
**Analytic gates**
- Gaussian free‑space: L2 field error ≤3%, energy error ≤1%.
- Circular aperture: Airy peak and first‑zero within 2%.
- Thin lens (paraxial): Strehl ≥0.95; MTF cutoff within 2% at low NA.
- Phase grating: order power ratios within 3%.
- High‑NA cross‑check: AS vs denser reference ΔL2 ≤5%.

**Engineering gates**
- Unit tests ≥90% pass on CPU CI; GPU smoke test must pass core solvers on small grids.
- Deterministic seeds enforced through config and RNG capture.

## 12. Interfaces (contracts)
### 12.1 `core.config`
- `load(path:str)->dict` and `validate(cfg:dict)->Cfg` with schema enforcement.
- Units normalized to µm; deterministic flags and seeds recorded.

### 12.2 `core.frames`
- Right‑hand world frame; Z‑Y‑X Euler order.
- `compose(euler_zyx, t_um)->T`, `to_world(p_local, T)->p_world`.

### 12.3 `sources`
- Inputs: spectrum, angular spread, coherence model.
- Output per sample: complex field `[Ny,Nx]` on source plane.

### 12.4 `components`
- Thin/thick lens, aperture, phase grating; 6‑DOF placement; returns transmission/phase map on local grid.

### 12.5 `prop.plan`
- `make_plan(cfg)->Plan` yields grid, Δz list, samples, PML, guard bands, memory budget.

### 12.6 `prop.solvers`
- `bpm.vector_wide_angle`, `bpm.split_step_fourier`, `as.multi_slice_nonparaxial` expose `run(field, plan, sampler)->field`.
- Must honor planner band‑limits, PML, and mixed precision policy.

### 12.7 `recorders`
- `add_plane(z_um, what={'intensity','complex','phase'})`, `capture(state)->Record`.

### 12.8 `io.tiff`
- Write TIFF stacks with metadata; read path optional post‑v0.1.

## 13. Tooling and CI
- **Linters**: Ruff; **typing**: mypy strict.
- **CI**: GitHub Actions CPU jobs + optional self‑hosted GPU job.
- **Pre‑commit**: YAML checks, trailing whitespace, Ruff format.

## 14. Governance and roles
- **ChatGPT**: maintains Blueprint and release HLDs; arbitrates scope; requests decisions.
- **Claude (Opus)**: writes core implementation and tests per HLD.
- **Cursor AI (GPT‑5)**: scaffolding, CI, hygiene, integration, examples, docs.
- **Routing rule**: each agent receives only its task file section.

## 15. Roadmap (high‑level)
- **v0.1**: CLI, sources, BPM/AS solvers, recorders, TIFF I/O, validation suite, budgets, CI.
- **v0.2**: richer components (mirrors, beamsplitters), polarization surface exposure, performance passes.
- **v0.3**: 3D objects and phase plates, basic GUI, extended readers.

## 16. Risks and mitigations
- High‑NA accuracy → short Δz, stricter band‑limits, AS reference checks.
- Aliasing → enforce Δx rules, adaptive resampling, guard‑band padding.
- VRAM ceiling → streaming planes, in‑place ops, mixed‑precision FFTs with FP32 accumulation.
- PML tuning → auto‑size by NA and λ; energy audit ≤1% leak.

## 17. Decision log (locked for v0.1)
- Backend: **PyTorch (CUDA 12.1)**.
- Python: **3.11**.
- Rotation order: **Z‑Y‑X**.
- TIFF compression: **none**.
- Default recorder outputs: **intensity + complex field**.
- Repo and path: **`microscope`**, **`C:\\Users\\Moni\\Documents\\claudeprojects\\microscope`**.

## 18. Change control
- Any change to §8, §9, or §11 requires a minor or patch bump and an HLD update.
- Open a PR with rationale and measured impact; ChatGPT approves blueprint deltas.

## 19. Glossary
- **AS**: Angular Spectrum method.
- **BPM**: Beam Propagation Method.
- **MTF/OTF**: Modulation/Optical Transfer Function.
- **PML**: Perfectly Matched Layer.
- **NA**: Numerical Aperture.

## 20. Appendix: File conventions
- Package root: `src/optics_sim`.
- Tests: `tests/`, pytest markers `gpu` for GPU‑only suites.
- Examples: `examples/low_na.yml`, `examples/mid_na.yml`, `examples/high_na.yml`.

