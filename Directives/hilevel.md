# HILEVEL — Architecture and Interfaces

## System overview
Simulate 3D vector electromagnetic fields in a microscope column and produce intensity images at arbitrary recorder planes. Prioritize accuracy and determinism. Single‑GPU workstation target.

```
Config → Scene(Graph of Sources, Components, Recorders)
        ↓
Builder → Solver(Pipeline)
        ↓                          ┌──────────┐
   Field planes (E)  ───────────▶  │Exporters │ → TIFF/NPZ + logs
        │                          └──────────┘
        └── Validation harness (analytic + regression baselines)
```

## Package layout
```
src/microscope/
  __init__.py
  cli/__init__.py
  cli/main.py                 # `python -m microscope.cli`
  core/types.py               # Tensor aliases, dataclasses
  core/units.py               # µm, nm, rad, deg; helpers
  core/frames.py              # RH coordinate frames, transforms
  core/errors.py              # Typed exceptions
  core/logging.py             # Structured logs (JSON lines)
  core/config.py              # Pydantic models and schema IO
  core/sampling.py            # Grid heuristics, padding, PML params
  gpu/torch_backend.py        # Device, dtype, FFT wrappers, RNG
  physics/sources.py          # Source primitives
  physics/components/
    lens.py                   # Thin/thick lens, NA and pupil
    aperture.py               # Circular/rectangular stops
    grating.py                # Phase gratings
  physics/propagation/
    angular_spectrum.py       # Wide‑angle, vector aware
    bpm.py                    # Beam propagation method (linear)
    pml.py                    # Simple PML/absorbing boundaries
  physics/solvers.py          # Pipeline orchestration
  io/export.py                # TIFF/NPZ writers, metadata
  validate/cases.py           # Analytic cases + tolerances
  presets/                    # Sampling presets and scene examples

tests/
  conftest.py
  test_cli.py
  test_sampling.py
  test_sources.py
  test_lens.py
  test_aperture.py
  test_grating.py
  test_propagation_as.py
  test_validation.py
```

## Coordinate frames and units
- Global right‑handed world frame **W** in **µm**. z increases along the optical axis, x to the right, y upward.
- Each element has a local frame **L** with pose `Pose = {t: (x,y,z)[µm], r: (rx,ry,rz)[deg], order: 'ZYX'}`. Transforms are explicit `T_LW`.
- Angles in degrees for config, radians internally. Wavelengths in nm, converted to µm for computation.

## Data model (config)
Pydantic models. JSON/YAML accepted. Minimal sketch:
```python
class LightSource(BaseModel):
    id: str
    kind: Literal['laser','gaussian','top_hat','blackbody','led']
    center_wavelength_nm: float
    bandwidth_nm: float | None = None
    spatial_profile: Literal['point','gaussian','top_hat']
    coherence: Literal['coherent','partial','incoherent']
    pose: Pose

class Lens(BaseModel):
    id: str
    kind: Literal['thin','thick']
    focal_length_mm: float | None
    NA: float | None
    aperture_mm: float | None
    pose: Pose

class Aperture(BaseModel):
    id: str
    shape: Literal['circular','rect']
    size_mm: float | tuple[float,float]
    pose: Pose

class Grating(BaseModel):
    id: str
    pitch_um: float
    orientation_deg: float
    phase_radians: float = 0.0
    pose: Pose

class Recorder(BaseModel):
    id: str
    plane_z_um: float | None
    pose: Pose | None
    kind: Literal['field','intensity']
    size_um: tuple[float,float]
    samples: tuple[int,int]

class Scene(BaseModel):
    sources: list[LightSource]
    components: list[Component]  # Union of Lens/Aperture/Grating
    recorders: list[Recorder]
    sampling: SamplingPreset
```

## Core interfaces (contracts)
```python
class Field2D(Protocol):
    tensor: Tensor  # [H, W, 2 or 3 complex components]
    spacing_um: tuple[float,float]
    wavelength_um: float

class Source(Protocol):
    def emit(self, grid: GridSpec, device: Device) -> Field2D: ...

class Component(Protocol):
    def apply(self, field: Field2D, ctx: Context) -> Field2D: ...

class Propagator(Protocol):
    def step(self, field: Field2D, dz_um: float, ctx: Context) -> Field2D: ...

class Solver(Protocol):
    def run(self, scene: Scene, ctx: Context) -> dict[str, Any]: ...
```
- Vector fields: store Ex,Ey,(Ez) as complex FP32. FP64 validation mode.
- Intensity images computed as `|E|^2` with energy conservation checks.

## Physics choices
- Start with linear, frequency‑domain propagation.
- Angular spectrum method for free‑space, with wide‑angle correction.
- BPM for sections where stepwise propagation is preferred.
- Pupil and element effects as multiplicative transfer functions in Fourier or spatial domain as appropriate.
- Non‑paraxial, high‑NA handling via vector transfer functions.

## Sampling policy
- Auto‑sampling from NA, wavelength, and FOV to meet phase error target.
- Padding and optional PML at edges.
- Heuristics return `(dx, dy, dz, H, W, steps)` with memory guardrail based on available VRAM.

## CLI
- `python -m microscope.cli run --config scene.yaml --out out_dir` → writes TIFF/NPZ and logs.
- `validate` subcommand runs analytic cases and checks tolerances.
- `inspect` prints sampling and memory estimates.

## Errors and logging
- `MicroscopeError` base with subclasses: `ConfigError`, `SamplingError`, `BackendError`, `PhysicsError`, `IOError`.
- JSONL logs: start time, git SHA, seed, device, dtype, memory budget, sampling, per‑stage timings, energy error.

## Acceptance tests
- Analytic: Gaussian beam propagation, thin‑lens focus, Fraunhofer aperture patterns, plane‑wave through lens.
- CPU/GPU parity within tolerance on 128×128 grids.
- Energy conservation ≤ 1% by default. L2 field error ≤ 5% for analytic cases.

## Backends and determinism
- Default device: CPU. Optional GPU via PyTorch CUDA when available.
- Fixed RNG seeds. Torch determinstic flags enabled when possible.
- All dependencies pinned. One `verify` entrypoint gates format→lint→type→unit→integration.

## Roadmap mapping
- v0.1: Sources + components above, angular spectrum + BPM, recorders, CLI, validation harness, TIFF/NPZ exporters.
- v0.2: Dispersive materials, polarization handling, more components, GUI stub.
- v1.0: Node editor, richer targets, nonlinearity hooks.

