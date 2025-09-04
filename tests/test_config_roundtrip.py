"""Test configuration round-trip serialization."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from microscope.core.config import (
    Aperture,
    Grating,
    Lens,
    LightSource,
    Recorder,
    Scene,
    load_config,
    round_trip_config,
    save_config,
)
from optics_sim.core import config as cfg


def test_empty_scene_creation():
    """Test creating an empty scene fails validation."""
    with pytest.raises(ValueError, match="at least one light source"):
        Scene()


def test_minimal_scene():
    """Test creating a minimal valid scene."""
    scene = Scene(
        sources=[
            LightSource(
                id="source1",
                kind="laser",
                center_wavelength_nm=532.0,
            )
        ],
        recorders=[
            Recorder(
                id="recorder1",
                plane_z_um=100.0,
            )
        ],
    )
    assert len(scene.sources) == 1
    assert len(scene.recorders) == 1
    assert scene.sampling == "mid_na"  # Default


def test_yaml_round_trip():
    """Test YAML serialization round-trip."""
    # Create a scene with various components
    scene = Scene(
        sources=[
            LightSource(
                id="laser",
                kind="laser",
                center_wavelength_nm=632.8,
                coherence="coherent",
            ),
            LightSource(
                id="led",
                kind="led",
                center_wavelength_nm=470.0,
                bandwidth_nm=20.0,
                coherence="partial",
            ),
        ],
        components=[
            Lens(
                id="objective",
                kind="thin",
                NA=0.65,
                focal_length_mm=4.0,
            ),
            Aperture(
                id="iris",
                shape="circular",
                size_mm=2.0,
            ),
            Grating(
                id="grating",
                pitch_um=10.0,
                orientation_deg=45.0,
            ),
        ],
        recorders=[
            Recorder(
                id="camera",
                plane_z_um=200.0,
                kind="intensity",
                size_um=(50.0, 50.0),
                samples=(512, 512),
            ),
        ],
    )

    # Round-trip through YAML
    scene_reloaded = round_trip_config(scene)

    # Check all fields preserved
    assert len(scene_reloaded.sources) == 2
    assert len(scene_reloaded.components) == 3
    assert len(scene_reloaded.recorders) == 1

    # Check specific values
    assert scene_reloaded.sources[0].center_wavelength_nm == 632.8
    assert scene_reloaded.sources[1].bandwidth_nm == 20.0
    assert scene_reloaded.components[0].NA == 0.65  # type: ignore
    assert scene_reloaded.components[1].size_mm == 2.0  # type: ignore
    assert scene_reloaded.components[2].pitch_um == 10.0  # type: ignore
    assert scene_reloaded.recorders[0].samples == (512, 512)


def test_yaml_file_io():
    """Test saving and loading from YAML file."""
    scene = Scene(
        sources=[
            LightSource(
                id="source",
                kind="gaussian",
                center_wavelength_nm=488.0,
            )
        ],
        recorders=[
            Recorder(
                id="detector",
                plane_z_um=50.0,
            )
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save to YAML
        yaml_path = Path(tmpdir) / "test_config.yaml"
        save_config(scene, yaml_path)

        # Check file exists
        assert yaml_path.exists()

        # Load back
        loaded_scene = load_config(yaml_path)

        # Verify
        assert loaded_scene.sources[0].id == "source"
        assert loaded_scene.sources[0].center_wavelength_nm == 488.0


def test_json_file_io():
    """Test saving and loading from JSON file."""
    scene = Scene(
        sources=[
            LightSource(
                id="source",
                kind="led",
                center_wavelength_nm=590.0,
                bandwidth_nm=30.0,
            )
        ],
        recorders=[
            Recorder(
                id="detector",
                plane_z_um=75.0,
                kind="field",
            )
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save to JSON
        json_path = Path(tmpdir) / "test_config.json"
        save_config(scene, json_path)

        # Check file exists
        assert json_path.exists()

        # Load back
        loaded_scene = load_config(json_path)

        # Verify
        assert loaded_scene.sources[0].id == "source"
        assert loaded_scene.sources[0].bandwidth_nm == 30.0
        assert loaded_scene.recorders[0].kind == "field"


def test_yaml_json_compatibility():
    """Test that YAML and JSON produce same scene."""
    scene = Scene(
        sources=[
            LightSource(
                id="laser",
                kind="laser",
                center_wavelength_nm=1064.0,
            )
        ],
        components=[
            Lens(
                id="lens",
                kind="thick",
                focal_length_mm=10.0,
                aperture_mm=25.4,
            )
        ],
        recorders=[
            Recorder(
                id="cam",
                plane_z_um=100.0,
            )
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save as both formats
        yaml_path = Path(tmpdir) / "config.yaml"
        json_path = Path(tmpdir) / "config.json"

        save_config(scene, yaml_path)
        save_config(scene, json_path)

        # Load both
        yaml_scene = load_config(yaml_path)
        json_scene = load_config(json_path)

        # Compare key fields
        assert (
            yaml_scene.sources[0].center_wavelength_nm == json_scene.sources[0].center_wavelength_nm
        )
        assert (
            yaml_scene.components[0].focal_length_mm  # type: ignore
            == json_scene.components[0].focal_length_mm  # type: ignore
        )
        assert yaml_scene.recorders[0].plane_z_um == json_scene.recorders[0].plane_z_um


def test_validation_errors():
    """Test that validation catches invalid configs."""
    # Invalid NA
    with pytest.raises(ValueError, match="NA must be between"):
        Lens(
            id="bad_lens",
            kind="thin",
            NA=2.0,  # Too high
        )

    # Invalid wavelength
    with pytest.raises(ValueError, match="Wavelength must be between"):
        LightSource(
            id="bad_source",
            kind="laser",
            center_wavelength_nm=50.0,  # Too low
        )

    # Invalid aperture size
    with pytest.raises(ValueError, match="Aperture size must be positive"):
        Aperture(
            id="bad_aperture",
            shape="circular",
            size_mm=-1.0,  # Negative
        )

    # Invalid grating pitch
    with pytest.raises(ValueError, match="Grating pitch must be between"):
        Grating(
            id="bad_grating",
            pitch_um=0.01,  # Too small
        )


def test_config_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "a.yaml"
    p.write_text(
        """
NA_max: 0.6
lambda_nm: 532
grid:
  target_px: 64
runtime:
  budget:
    vram_gb: 10.0
    time_s: 60
"""
    )
    loaded = cfg.load(p)
    norm = cfg.validate(loaded)
    out = tmp_path / "norm.json"
    out.write_text(json.dumps(norm, indent=2))
    assert norm["NA_max"] == 0.6 and "lambda_um" in norm
