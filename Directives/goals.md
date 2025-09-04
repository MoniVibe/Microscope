# GOALS — v0.1 (Microscope)

## Outcomes
- **Accuracy**: energy conservation ≤1%; L2 field error ≤5% vs analytic; Strehl ≥0.95 for presets; MTF cutoff error ≤2% at low NA.
- **Determinism**: fixed RNG seeds; pinned deps; idempotent scripts; one `verify` entrypoint.
- **Usability**: CLI `python -m microscope.cli`; three worked examples; TIFF/NPZ exports; logs with config and sampling.
- **Performance**: single‑GPU workstation target; standard presets complete ≤10 min; memory guardrail prevents OOM.
- **Quality gates**: ruff+format clean on new code; mypy zero‑new; coverage ≥70%; CI mirrors local and blocks on red.

## Milestones and dates
- **M1 Vertical slice** (2025‑09‑12): package layout, CLI stub, config I/O, basic sampling, CPU angular‑spectrum on plane‑wave. Deliver: CLI runs, 1 example, tests pass on CPU.
- **M2 Components** (2025‑09‑19): thin/thick lens, aperture, grating; recorders for field/intensity; exporters. Deliver: 3 examples produce TIFF/NPZ.
- **M3 Validation** (2025‑09‑24): analytic cases (Gaussian, thin‑lens focus, Fraunhofer aperture); metrics enforced; GPU tests marked and skipped if CUDA absent. Deliver: `validate` green with thresholds.
- **M4 Harden** (2025‑09‑30): sampling heuristics with memory guardrail; logs and error taxonomy; docs polish. Deliver: coverage ≥70%, CI artifacts and required checks.

## KPIs and thresholds
- Coverage ≥70% lines; no drop in PRs.
- `ruff check` and `mypy` only report baseline items; new code is clean.
- CLI end‑to‑end run produces logs with seed, device, dtype, memory budget, sampling, timings.
- Three examples complete within runtime budget on CPU.

## Test plan (IDs)
- **T‑001** Energy conservation on free‑space propagation ≤1%.
- **T‑002** L2 field error ≤5% for Gaussian beam.
- **T‑003** Thin‑lens focus Strehl ≥0.95.
- **T‑004** Fraunhofer aperture MTF cutoff error ≤2% at low NA.
- **T‑005** CPU/GPU parity on 128×128 grid within tolerance.
- **T‑006** CLI `run` produces TIFF/NPZ and JSONL logs.
- **T‑007** Sampling auditor warns on under‑sampling and caps memory.
- **T‑008** Import‑time CPU‑only path; GPU tests marked and skipped when CUDA missing.
- **T‑009** Reproducibility: same seed → identical outputs.
- **T‑010** Config schema round‑trip YAML↔JSON.
- **T‑011** Coverage ≥70%.
- **T‑012** CI uploads `coverage.xml` and `reports/*`.

## Risks and mitigations
- High‑NA sampling pressure → memory guardrail and presets.
- Import‑time CUDA coupling → backend isolation and marks.
- Numeric drift → FP64 validation path for analytic cases.

## Acceptance for v0.1
- `make bootstrap && make verify` green on Windows and Ubuntu CPU.
- All KPIs met. All tests T‑001…T‑012 pass or are skipped only where specified.
- Three examples match documented outputs.

---

# OBJECTIVES seeds (create as separate files under `OBJECTIVES/`)

> Template front‑matter:
>
> ```yaml
> owner: Opus
> inputs: [VISION.md, docs/HILEVEL.md, docs/GOALS.md]
> outputs: []
> tests: []
> gate: [format, lint, type, unit, integration]
> est: "≤2h"
> ```

## OBJECTIVES/001_package_layout.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/*, tests/conftest.py]
tests: [T‑006, T‑011, T‑012]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: Create package layout per HILEVEL. Add `__all__` in public modules. Add `python -m microscope.cli` entry.
**Done**: `pytest -q` runs discovery; `python -m microscope.cli --help` works.

## OBJECTIVES/002_cli_scaffold.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/cli/main.py]
tests: [T‑006, T‑009]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: `run|validate|inspect` subcommands. JSONL logging.
**Done**: CLI produces outputs and logs paths; deterministic run with seed.

## OBJECTIVES/003_config_models.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/core/config.py]
tests: [T‑010]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: Pydantic models for Scene, Sources, Components, Recorders. YAML/JSON I/O.
**Done**: Round‑trip and schema validation tests pass.

## OBJECTIVES/004_sampling_heuristics.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/core/sampling.py]
tests: [T‑007]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: From NA, λ, FOV → (dx,dy,dz,H,W,steps). Memory guardrail. Warnings on under‑sampling.
**Done**: Heuristics exercised by `inspect`; warns appropriately.

## OBJECTIVES/005_angular_spectrum_cpu.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/physics/propagation/angular_spectrum.py, tests/test_propagation_as.py]
tests: [T‑001, T‑002]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: CPU angular‑spectrum propagation with wide‑angle correction. FP32 default, FP64 validation path.
**Done**: Energy and L2 error thresholds met.

## OBJECTIVES/006_components_optics.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/physics/components/{lens.py,aperture.py,grating.py}, tests/test_lens.py, tests/test_aperture.py, tests/test_grating.py]
tests: [T‑003, T‑004]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: Thin/thick lens, apertures, gratings as transfer functions. Include NA and pupil handling.
**Done**: Strehl and MTF tests meet thresholds.

## OBJECTIVES/007_recorders_and_export.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/io/export.py, tests/test_cli.py]
tests: [T‑006]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: Field and intensity recorders. TIFF/NPZ export with metadata.
**Done**: CLI run writes expected files and logs.

## OBJECTIVES/008_validation_harness.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/validate/cases.py, tests/test_validation.py]
tests: [T‑001, T‑002, T‑003, T‑004]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: Analytic cases and thresholds. Report table in logs.
**Done**: `cli validate` passes with metrics logged.

## OBJECTIVES/009_gpu_optional.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/gpu/torch_backend.py, tests/marks]
tests: [T‑005, T‑008]
gate: [format, lint, type, unit]
est: "≤2h"
```
**Task**: Optional CUDA backend. Import‑time isolation. GPU tests marked and skipped if CUDA absent.
**Done**: CPU/GPU parity test within tolerance on small grid.

## OBJECTIVES/010_examples.md
```yaml
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [examples/{plane_wave_lens,aperture_diffraction,grating}.yaml]
tests: [T‑006]
gate: [format, lint, unit]
est: "≤2h"
```
**Task**: Three runnable scenes with expected outputs documented.
**Done**: Each runs within budget and produces files.

