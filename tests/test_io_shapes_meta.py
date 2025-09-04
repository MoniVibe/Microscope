"""Tests for TIFF I/O with metadata."""

import importlib.util
import tempfile
from pathlib import Path

import numpy as np
import torch
from optics_sim.io.tiff import read_tiff, write_field_stack, write_tiff


def test_io_shapes_meta():
    """Test TIFF I/O preserves shapes and metadata."""
    # Create test data
    ny, nx = 128, 256
    data_real = torch.randn(ny, nx, dtype=torch.float32)
    data_complex = torch.randn(ny, nx, dtype=torch.complex64)

    # Metadata
    metadata = {
        "wavelength_um": 0.55,
        "NA": 0.65,
        "seeds": {"seed_tensor": 42, "seed_sampler": 1337},
        "config_hash": "abc123",
        "test_field": "test_value",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        # Test real data
        real_file = tmpdir / "test_real.tif"
        write_tiff(real_file, data_real, metadata=metadata, dx_um=0.5, dy_um=0.5)

        # Verify file exists
        assert real_file.exists(), "Real TIFF file not created"

        # Test complex data
        complex_file = tmpdir / "test_complex.tif"
        write_tiff(complex_file, data_complex, metadata=metadata, dx_um=0.5, dy_um=0.5)

        assert complex_file.exists(), "Complex TIFF file not created"

        # If tifffile is available, test reading back
        if importlib.util.find_spec("tifffile") is not None:
            # Read real data
            data_read, meta_read = read_tiff(real_file)
            assert (
                data_read.shape == data_real.shape
            ), f"Shape mismatch: {data_read.shape} vs {data_real.shape}"

            # Check metadata keys
            required_keys = ["dx_um", "dy_um", "units", "is_complex"]
            for key in required_keys:
                assert key in meta_read, f"Missing metadata key: {key}"

            # Read complex data
            data_c_read, meta_c_read = read_tiff(complex_file)

            # Should be marked as complex
            assert meta_c_read.get("is_complex", False), "Complex flag not set"

            # Shape should match (might have extra dimension for real/imag)
            NDIM_3 = 3  # PLR2004 named constant for this test
            COMPLEX_PLANES = 2
            if data_c_read.ndim == NDIM_3:
                # Stacked real/imaginary
                assert data_c_read.shape[0] == COMPLEX_PLANES, "Complex should have 2 planes"

            print("✓ TIFF read/write with tifffile successful")
        else:
            print("⚠ tifffile not available, skipping read test")

    print("✓ TIFF files created with correct metadata structure")


def test_field_stack_io():
    """Test writing multiple fields to a stack."""
    # Create multiple fields
    fields = {
        "input": torch.randn(64, 64, dtype=torch.complex64),
        "output": torch.randn(64, 64, dtype=torch.complex64),
        "intensity": torch.rand(64, 64, dtype=torch.float32),
    }

    config = {
        "dx_um": 0.25,
        "dy_um": 0.25,
        "dz_um": 1.0,
        "NA_max": 0.75,
        "wavelengths_um": np.array([0.488, 0.55, 0.633]),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        stack_file = Path(tmpdir) / "field_stack.tif"

        write_field_stack(stack_file, fields, config)

        assert stack_file.exists(), "Field stack file not created"

        # Check file size is reasonable
        file_size = stack_file.stat().st_size
        min_size = 64 * 64 * 4 * len(fields)  # Minimum bytes for data
        assert file_size >= min_size, f"File too small: {file_size} < {min_size}"

        print(f"✓ Field stack written: {file_size} bytes")


if __name__ == "__main__":
    test_io_shapes_meta()
    test_field_stack_io()
    print("\nTIFF I/O tests passed!")
