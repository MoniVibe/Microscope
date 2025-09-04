---
owner: Opus
inputs: [docs/HILEVEL.md]
outputs: [src/microscope/core/config.py]
tests: [T-010]
gate: [format, lint, type, unit]
est: "≤2h"
status: COMPLETE
---

# OBJECTIVES/003_config_models.md

**Task**: Pydantic models for Scene, Sources, Components, Recorders. YAML/JSON I/O.

**Done**: 
- Implemented full Pydantic models in `src/microscope/core/config.py`
- Created models for all optical elements per HILEVEL specification
- Added YAML/JSON loading and saving functions
- Implemented validation with appropriate ranges
- Created comprehensive test suite with round-trip tests

**Models created**:
- `LightSource` - laser, gaussian, top_hat, blackbody, LED sources
- `Lens` - thin/thick lens with NA and focal length
- `Aperture` - circular/rectangular apertures
- `Grating` - phase gratings with pitch and orientation
- `Recorder` - field/intensity recorders at arbitrary planes
- `Scene` - complete simulation configuration
- `Pose` - 6-DOF positioning system
- Supporting enums for type safety

**Features**:
- Automatic unit conversion (nm to µm for wavelengths)
- Validation of physical parameters (NA, wavelength ranges, etc.)
- YAML/JSON round-trip serialization
- Default values following physics conventions
- Type-safe enumerations for all options

**Outputs created**:
- `src/microscope/core/config.py` - Complete Pydantic models
- `tests/test_config_roundtrip.py` - Round-trip test suite
- `examples/minimal.yaml` - Example configuration

**Tests**: Round-trip and schema validation tests pass.
