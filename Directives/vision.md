# VISION — Microscope (Z‑max alternative)

**One‑liner**: Build the most accurate, GPU‑accelerated 3D vector‑field microscope simulator available to individuals. Zero *uncontrolled* approximations. Accuracy over speed. Deterministic and reproducible by default.

**Why**

* Existing optical design tools optimize for convenience and paraxial speed. Our use cases require full‑vector, high‑NA accuracy that can be audited and reproduced.
* We target research and advanced prototyping on a single workstation GPU.

**Who**

* Primary: optical engineers and researchers modeling microscopes and custom optical columns.
* Secondary: algorithm developers validating propagation and sampling schemes.

**Product vision**

* Alternative to Z‑max/Zemax focused on *exactness-first* simulation, not lens‑catalog ergonomics.
* Compose arbitrary microscopes from sources, lenses, apertures, and gratings. Arbitrary 3D placement and orientation (6‑DOF).
* Compute vector electric fields **E(x,y,z)** and intensity images at any recorder plane along the column.
* Users control number and types of light sources, coherence, spectra, placement, and bandwidth. Components expose physical parameters: focal length, NA, aperture size, grating pitch/orientation, etc.

**Physics vision**

* Frequency‑domain, steady‑state vector fields.
* Linear effects in v0.1. Framework hooks for nonlinearity later.
* Materials start with air and glass. Framework for dispersive materials.
* NA‑aware propagation: BPM and angular‑spectrum with wide‑angle and non‑paraxial treatment. PML boundaries by default.

**Principles**

* *Accuracy first*: choose the most accurate known method for the NA and sampling.
* *Zero uncontrolled approximations*: all numerical choices are explicit, documented, and testable.
* *Determinism*: fixed seeds, pinned dependencies, idempotent scripts, single verify entrypoint.
* *Reproducibility*: inputs, transforms, and outputs are logged and exportable.
* *No surprise state*: explicit coordinate frames and units.

**Scope (v0.1)**

* Sources: unlimited count. Extended sources only. Models: laser (δ‑λ), Gaussian, top‑hat, blackbody, LED. Control coherence (spatial/spectral), wavelength range 300–1500 nm, bandwidth 5–1200 nm, polarization prepared but excluded in v0.1. Angle of illumination per source.
* Components: thin/thick lenses, apertures, phase gratings. Beam splitters and mirrors groundwork only.
* Targets: amplitude‑only image‑as‑transmission. Hooks for phase plates and volumetric RI later.
* Propagation policy: auto‑select method by NA and step; free‑space FFT steps allowed; arbitrary recorder planes.
* Grids and sampling: adaptive pitch with phase‑fidelity target; padding and PML on open boundaries.

**Non‑goals (v0.1)**

* No real‑time preview. No time‑domain solvers. No nonlinear media. No measured phase maps. No vendor lens catalogs.

**Quality bar**

* Energy conservation ≤ 1%.
* Field L2 error ≤ 2–5% vs analytic cases.
* Strehl ≥ 0.95 for standard presets.
* MTF cutoff within 2% for low‑NA cases.

**Validation set**

* Gaussian beam propagation, Fresnel/Fraunhofer diffraction, thin‑lens focusing, plane‑wave through lens.
* CPU/GPU parity checks on small grids. FP64 validation mode for reference.

**Inputs**

* Text configuration first. Later: GUI and node editor.
* Global right‑handed world frame in µm. Per‑component local frames with explicit origins and rotation order; transforms local→world are recorded.
* Component library via parameter files.

**Outputs**

* Complex vector fields **E(x,y,z)**, intensities, and phase maps.
* TIFF stacks for images (32‑bit float recommended). Complex fields encoded as real/imag matrices when exported.
* Intermediate planes only on request.

**Runtime and hardware**

* Single‑GPU workstation (RTX 3080+ recommended). FP32 default with optional mixed precision for FFTs and element‑wise ops; FP32 accumulations. Several minutes per image maximum for standard presets.

**Product surface**

* CLI first: `python -m microscope.cli` with subcommands for run, export, validate, and inspect.
* Examples: minimal microscope, high‑NA objective, grating + aperture, multisource illumination.

**Risks and mitigations**

* High‑NA accuracy vs memory: enforce adaptive sampling and z‑step heuristics; provide presets that fit 10 GB.
* Import‑time GPU coupling: design CPU‑only code paths and mark GPU tests; skip when CUDA absent.
* User mis‑sampling: add runtime budget and sampling auditors with actionable warnings.

**Milestones**

* **v0.1 CLI (accuracy‑first)**: linear media, sources/components/targets above, NA‑aware propagation, TIFF outputs, validation harness, presets, single‑GPU.
* **v0.2 (breadth)**: dispersive materials, more components, polarization handling, GUI stub.
* **v1.0 (usability)**: node editor, library manager, advanced targets, nonlinearity framework behind flags.

**Done criteria for v0.1**

* `make bootstrap && make verify` green on Windows and Ubuntu CPU. GPU smoke test pass on RTX 3080.
* Validation metrics satisfied on the analytic set. Reproducible runs with locked deps and logs.
* CLI produces fields and images at arbitrary recorder planes with documented sampling.

**Repository and governance**

* Path: `C:\\Users\\Moni\\Documents\\claudeprojects\\Microscope`.
* GitHub: [https://github.com/MoniVibe/Microscope](https://github.com/MoniVibe/Microscope)
* Doc flow: `VISION.md` → `docs/HILEVEL.md` → `docs/GOALS.md` → `OBJECTIVES/*`.
* Default owner: Opus. GPT‑5 integrates, configures CI, and writes tests per plan.

**Interfaces to specify in HILEVEL**

* Plugin contracts: `Source`, `Component`, `Solver` with explicit units and frames.
* Error types and logging schema. I/O schema for configs and exports.
* Acceptance tests and baselines, including FP64 reference cases.
