# Microscope v0.1 — Lock-ins + ChatGPT Prompt

**Project path:** `C:\Users\Moni\Documents\claudeprojects\microscope`

---

## Lock-ins (authoritative)

### Scope and end goal
- Design tool. Arbitrary microscopes and optical systems. No replication.
- No hard deadline.

### Physics and materials
- Static, steady-state, frequency-domain vector field; no time-dependent solvers.
- Linear effects only in v0.1. Prepare framework for nonlinear effects.
- Materials: air and glass in v0.1. Framework for additional dispersive materials.

### Sources
- Unlimited sources; may be mutually coherent or not.
- Extended sources only.
- Models: laser δ-λ, finite-band Gaussian, top-hat, blackbody, LED.
- Wavelength range: 300–1500 nm. Bandwidth: 5–1200 nm.
- Angle of illumination included per source.
- Coherence specified per source (spatial and spectral). Polarization prepared but excluded in v0.1.

### Propagation and boundaries
- NA-aware method selection:
  - Small NA ≤ 0.30 → vector BPM with wide-angle correction.
  - Moderate NA 0.30–0.70 → wide-angle BPM + split-step Fourier, spectral band-limit, adaptive Δz.
  - High NA > 0.70 → angular-spectrum multi-slice propagation with short Δz and nonparaxial treatment.
  - Auto-select per step. Accuracy prioritized over speed.
- Free-space FFT steps allowed anywhere; arbitrary recorder planes permitted.
- PML enabled by default. Ghost identification groundwork stored via secondary path graph; multi-bounce tracing disabled in v0.1.

### Components and targets
- Lenses: thin or thick by user choice.
- Apertures and phase gratings included in v0.1. Beam splitters and mirrors groundwork only (excluded in v0.1).
- Arbitrary 3D placement and orientation (6-DOF).
- Aberrations from optics, curvatures, and orientations; allow thin-lens + chosen aberrations; no measured phase maps.
- Targets: amplitude-only image-as-transmission in v0.1. Framework for phase plates, surface-relief objects, and full 3D refractive-index volumes.

### Grids and sampling
- Adaptive grid based on NA and beam size.
- Pixel pitch: Δx ≤ λ_min/(2·NA_max); phase-fidelity target Δx ≤ λ_min/(3·NA_max).
- 8–12 cells per wavelength at NA_max in tightest region.
- Padding: 8–16 px guards + PML 0.5–1.0 λ on open boundaries.
- Arbitrary user-selected z-planes recorded on demand.

### Domain presets (RTX 3080 10 GB, FP32)
- Standard: 1024×1024×64 planes, pitch 0.2–0.5 µm, ~1–3 min for 8–16 wavelengths.
- High-NA: 1536×1536×96 planes, pitch 0.1–0.25 µm, ~3–6 min.
- Aggressive: 2048×2048×128 planes, pitch 0.08–0.2 µm, ~6–12 min.
- Runtime budget: several minutes per image; heuristics may reduce spectral samples or z-planes.

### Precision and hardware
- Single GPU; target RTX 3080+; cross-vendor planned (home workstation).
- No real-time preview.
- Precision policy: FP32 default; allow mixed precision (TF32/FP16 for FFTs and elementwise) with FP32 accumulations; FP64 validation mode.

### Inputs and formats
- Early releases: text config; later GUI and node editor for PCs.
- Global right-handed world frame (x,y,z) in µm; per-component local frames with explicit origin and rotation order; transforms local→world.
- Target image format: TIFF 8-bit. Complex-field encoding uses two matrices (real, imaginary) when applicable.
- Component library: parameter files; groundwork for database-backed libraries.
- Reproducibility required.

### Outputs
- Complex E(x,y,z) vector fields, intensities, phase.
- Intermediate planes only when explicitly requested.
- Export format: TIFF stacks (32-bit float recommended).

### Validation and metrics
- Validate against analytical cases: Gaussian beam, Fresnel/Fraunhofer diffraction, thin-lens focusing, plane-wave through lens.
- Metrics: energy conservation ≤ 1%; field L2 error ≤ 2–5% vs analytic; Strehl ≥ 0.95 for standard presets; MTF cutoff within 2% for low-NA cases.

### Engineering
- Python primary; C++ kernels if profiling requires.
- Plugin architecture for Sources, Components, Solvers.
- Logging, checkpoints, and resume support required.
- Versioning: SemVer.
- Tests: PyTest with golden analytical cases and deterministic seeds.
- CI: GitHub Actions or equivalent; lint + tests; small GPU smoke test.
- OS targets: Windows 10/11 and Ubuntu 22.04+.
- Offline only.

### Repository logistics
- Path: `C:\\Users\\Moni\\Documents\\claudeprojects\\microscope`.
- Repo location and branches to be decided later.

---

## Prompt for ChatGPT — Generate High-Level Directive

Copy-paste the prompt below into ChatGPT when you want the directive generated.

```text
You are ChatGPT. Role: produce a **High-Level Directive** for project "microscope v0.1" based on the lock-ins provided in this same message. Follow this agent flow:
- ChatGPT: high-level directive, clarify only if a lock-in is missing, then produce code blueprint and task breakdown.
- Claude (Opus): code writing per ChatGPT blueprint.
- Cursor AI (GPT-5): integration, glue, consolidation, hygiene, and implementation directives.

Constraints and context:
- Units: µm throughout. Offline-only. Single-GPU home workstation targeting RTX 3080+. FP32 default; allow TF32/FP16 for FFTs and elementwise with FP32 accumulations; FP64 validation mode. No real-time preview. Runtime: several minutes per image max. NA-aware propagation policy with PML on by default. Extended sources only. Arbitrary 3D placements. Outputs as TIFF stacks.

Deliverables (one markdown file):
1) Objective and scope summary consistent with lock-ins.
2) Architecture and propagation policy (NA-aware BPM/AS switching) and component/target set for v0.1.
3) Module tree and interfaces, including plugin points (Sources, Components, Solvers) and coordinate frames.
4) Performance targets and runtime budget strategy.
5) Validation plan and error metrics.
6) I/O formats and metadata, reproducibility plan.
7) Task split:
   - **Claude (Opus)**: concrete modules to implement first, in order, with acceptance criteria.
   - **Cursor AI (GPT-5)**: repo scaffold, CI, logging/checkpoints/resume, examples, runtime budget enforcer, plugin stubs, docs skeleton.
8) Milestones with exit criteria for v0.1 CLI release.
9) Risks and mitigations tied to accuracy at high NA and memory limits.

Input you will receive alongside this prompt:
- A pasted **Lock-ins** section. Treat it as authoritative. Do not add explanations. Do not change requirements.

Output rules:
- Write a single cohesive High-Level Directive in markdown, no preamble, no extra commentary. No code beyond minimal pseudo-interfaces if needed.
- Keep tone concise and technical. No explanations of basic concepts.

Project path for references: `C:\\Users\\Moni\\Documents\\claudeprojects\\microscope`.
```

