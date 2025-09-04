"""Configuration models and I/O for microscope simulation.

Pydantic models for Scene, Sources, Components, Recorders with YAML/JSON I/O.
Units are normalized to micrometers internally, with nanometers accepted at input.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Literal, Union

import yaml

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
except ImportError:
    # Fallback for environments without pydantic
    BaseModel = object  # type: ignore
    Field = lambda *args, **kwargs: None  # type: ignore
    field_validator = lambda *args, **kwargs: lambda x: x  # type: ignore
    model_validator = lambda *args, **kwargs: lambda x: x  # type: ignore


class CoherenceModel(str, Enum):
    """Coherence model for light sources."""

    COHERENT = "coherent"
    PARTIAL = "partial"
    INCOHERENT = "incoherent"


class SourceKind(str, Enum):
    """Type of light source."""

    LASER = "laser"
    GAUSSIAN = "gaussian"
    TOP_HAT = "top_hat"
    BLACKBODY = "blackbody"
    LED = "led"


class SpatialProfile(str, Enum):
    """Spatial intensity profile."""

    POINT = "point"
    GAUSSIAN = "gaussian"
    TOP_HAT = "top_hat"


class ComponentKind(str, Enum):
    """Type of optical component."""

    THIN_LENS = "thin"
    THICK_LENS = "thick"
    APERTURE = "aperture"
    GRATING = "grating"


class ApertureShape(str, Enum):
    """Shape of aperture."""

    CIRCULAR = "circular"
    RECT = "rect"


class RecorderKind(str, Enum):
    """Type of recorder output."""

    FIELD = "field"
    INTENSITY = "intensity"


class SamplingPreset(str, Enum):
    """Predefined sampling configurations."""

    LOW_NA = "low_na"
    MID_NA = "mid_na"
    HIGH_NA = "high_na"
    CUSTOM = "custom"


class Pose(BaseModel):
    """6-DOF pose in 3D space."""

    t: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Translation (x, y, z) in micrometers"
    )
    r: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Rotation (rx, ry, rz) in degrees"
    )
    order: Literal["ZYX"] = Field(default="ZYX", description="Rotation order (Euler angles)")


class LightSource(BaseModel):
    """Light source configuration."""

    id: str = Field(description="Unique identifier")
    kind: SourceKind = Field(description="Type of light source")
    center_wavelength_nm: float = Field(description="Center wavelength in nanometers")
    bandwidth_nm: float | None = Field(default=None, description="Spectral bandwidth in nanometers")
    spatial_profile: SpatialProfile = Field(
        default=SpatialProfile.GAUSSIAN, description="Spatial intensity profile"
    )
    coherence: CoherenceModel = Field(
        default=CoherenceModel.COHERENT, description="Coherence model"
    )
    pose: Pose = Field(default_factory=Pose, description="Position and orientation")

    @field_validator("center_wavelength_nm")
    @classmethod
    def validate_wavelength(cls, v: float) -> float:
        """Validate wavelength is in reasonable range."""
        if not 100 <= v <= 2000:
            raise ValueError(f"Wavelength must be between 100 and 2000 nm, got {v}")
        return v


class Lens(BaseModel):
    """Lens component configuration."""

    id: str = Field(description="Unique identifier")
    kind: Literal["thin", "thick"] = Field(description="Lens type")
    focal_length_mm: float | None = Field(default=None, description="Focal length in millimeters")
    NA: float | None = Field(default=None, description="Numerical aperture")
    aperture_mm: float | None = Field(
        default=None, description="Clear aperture diameter in millimeters"
    )
    pose: Pose = Field(default_factory=Pose, description="Position and orientation")

    @model_validator(mode="after")
    def validate_lens_params(self) -> Lens:
        """Ensure at least one optical parameter is specified."""
        if not any([self.focal_length_mm, self.NA]):
            raise ValueError("Lens must specify either focal_length_mm or NA")
        if self.NA is not None and not 0.01 <= self.NA <= 1.4:
            raise ValueError(f"NA must be between 0.01 and 1.4, got {self.NA}")
        return self


class Aperture(BaseModel):
    """Aperture component configuration."""

    id: str = Field(description="Unique identifier")
    shape: ApertureShape = Field(description="Aperture shape")
    size_mm: float | tuple[float, float] = Field(
        description="Size in mm (diameter for circular, (width, height) for rect)"
    )
    pose: Pose = Field(default_factory=Pose, description="Position and orientation")

    @field_validator("size_mm")
    @classmethod
    def validate_size(cls, v: float | tuple[float, float]) -> float | tuple[float, float]:
        """Validate aperture size is positive."""
        if isinstance(v, (int, float)):
            if v <= 0:
                raise ValueError(f"Aperture size must be positive, got {v}")
        elif any(s <= 0 for s in v):
            raise ValueError(f"All aperture dimensions must be positive, got {v}")
        return v


class Grating(BaseModel):
    """Phase grating component configuration."""

    id: str = Field(description="Unique identifier")
    pitch_um: float = Field(description="Grating pitch in micrometers")
    orientation_deg: float = Field(default=0.0, description="Orientation angle in degrees")
    phase_radians: float = Field(default=0.0, description="Phase shift in radians")
    pose: Pose = Field(default_factory=Pose, description="Position and orientation")

    @field_validator("pitch_um")
    @classmethod
    def validate_pitch(cls, v: float) -> float:
        """Validate grating pitch is reasonable."""
        if not 0.1 <= v <= 1000:
            raise ValueError(f"Grating pitch must be between 0.1 and 1000 µm, got {v}")
        return v


# Union type for all components
Component = Union[Lens, Aperture, Grating]


class Recorder(BaseModel):
    """Recorder configuration for capturing fields or intensities."""

    id: str = Field(description="Unique identifier")
    plane_z_um: float | None = Field(
        default=None, description="Z position of recording plane in micrometers"
    )
    pose: Pose | None = Field(
        default=None, description="Custom pose (overrides plane_z_um if specified)"
    )
    kind: RecorderKind = Field(
        default=RecorderKind.INTENSITY, description="Type of recording (field or intensity)"
    )
    size_um: tuple[float, float] = Field(
        default=(100.0, 100.0), description="Physical size (width, height) in micrometers"
    )
    samples: tuple[int, int] = Field(default=(512, 512), description="Number of samples (nx, ny)")

    @model_validator(mode="after")
    def validate_position(self) -> Recorder:
        """Ensure recorder has a position defined."""
        if self.plane_z_um is None and self.pose is None:
            raise ValueError("Recorder must specify either plane_z_um or pose")
        return self


class SamplingConfig(BaseModel):
    """Sampling configuration for grid and propagation."""

    preset: SamplingPreset = Field(default=SamplingPreset.MID_NA, description="Sampling preset")
    target_px: int | None = Field(default=None, description="Target grid size in pixels")
    pitch_um: float | None = Field(default=None, description="Grid pitch in micrometers")
    z_steps: int | None = Field(default=None, description="Number of z-propagation steps")

    @field_validator("target_px")
    @classmethod
    def validate_grid_size(cls, v: int | None) -> int | None:
        """Validate grid size is reasonable."""
        if v is not None and not 64 <= v <= 8192:
            raise ValueError(f"Grid size must be between 64 and 8192, got {v}")
        return v


class RuntimeBudget(BaseModel):
    """Runtime resource budgets."""

    vram_gb: float = Field(default=10.0, description="Maximum VRAM usage in GB")
    time_s: float = Field(default=600.0, description="Maximum runtime in seconds")


class Scene(BaseModel):
    """Complete scene configuration."""

    sources: list[LightSource] = Field(default_factory=list, description="Light sources")
    components: list[Component] = Field(default_factory=list, description="Optical components")
    recorders: list[Recorder] = Field(default_factory=list, description="Recording planes")
    sampling: SamplingPreset | SamplingConfig = Field(
        default=SamplingPreset.MID_NA, description="Sampling configuration"
    )
    runtime_budget: RuntimeBudget = Field(
        default_factory=RuntimeBudget, description="Resource budgets"
    )

    @model_validator(mode="after")
    def validate_scene(self) -> Scene:
        """Validate scene has at least minimal components."""
        if not self.sources:
            raise ValueError("Scene must have at least one light source")
        if not self.recorders:
            raise ValueError("Scene must have at least one recorder")
        return self


def load_config(path: str | Path) -> Scene:
    """Load configuration from YAML or JSON file.

    Args:
        path: Path to configuration file

    Returns:
        Validated Scene object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config is invalid
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        if path.suffix.lower() in [".yaml", ".yml"]:
            data = yaml.safe_load(f)
        elif path.suffix.lower() == ".json":
            data = json.load(f)
        else:
            # Try YAML first, then JSON
            content = f.read()
            f.seek(0)
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError:
                data = json.loads(content)

    # Convert nm to um for wavelengths during loading
    if "sources" in data:
        for source in data["sources"]:
            if "center_wavelength_nm" not in source and "center_um" in source:
                source["center_wavelength_nm"] = source["center_um"] * 1000
            if "bandwidth_nm" not in source and "bandwidth_um" in source:
                source["bandwidth_nm"] = source["bandwidth_um"] * 1000

    return Scene(**data)


def save_config(scene: Scene, path: str | Path) -> None:
    """Save configuration to YAML or JSON file.

    Args:
        scene: Scene configuration to save
        path: Output file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = scene.model_dump(mode="json", exclude_unset=True)

    with open(path, "w") as f:
        if path.suffix.lower() in [".yaml", ".yml"]:
            yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            json.dump(data, f, indent=2)


def round_trip_config(scene: Scene) -> Scene:
    """Test round-trip serialization of a scene.

    Args:
        scene: Input scene

    Returns:
        Scene after serialization and deserialization
    """
    data = scene.model_dump(mode="json", exclude_unset=True)
    yaml_str = yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
    loaded_data = yaml.safe_load(yaml_str)
    return Scene(**loaded_data)
