"""Tests for precision policy enforcement.

Ensures:
- CUDA tensors are always FP32 (complex64)
- CPU tensors can use FP64 intermediates for AS solver
- No mixed precision before M2 sign-off
- CPU vs GPU parity within tolerance gates
"""

import numpy as np
import pytest
import torch

from optics_sim.core.precision import (
    MIXED_FFT,
    assert_fp32_cuda,
    enforce_fp32_cuda,
    fft2_with_precision,
    get_precision_dtype,
    validate_precision_invariants,
)
from optics_sim.prop.solvers import as_multi_slice, bpm_split_step_fourier, bpm_vector_wide
from optics_sim.validation.cases import gaussian_free_space
from optics_sim.validation.metrics import l2_field_error


def test_mixed_fft_disabled():
    """Test that mixed precision is disabled."""
    assert MIXED_FFT is False, "Mixed precision must be disabled before M2 sign-off"


def test_enforce_fp32_cuda():
    """Test FP32 enforcement on CUDA tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create FP64 tensor on CUDA
    tensor_fp64 = torch.randn(64, 64, dtype=torch.complex128, device="cuda")
    
    # Enforce FP32
    tensor_fp32 = enforce_fp32_cuda(tensor_fp64, "test tensor")
    
    # Check dtype
    assert tensor_fp32.dtype == torch.complex64, "Should be complex64 after enforcement"
    assert_fp32_cuda(tensor_fp32, "enforced tensor")


def test_assert_fp32_cuda_fails():
    """Test that assertion fails for non-FP32 CUDA tensors."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create FP64 tensor on CUDA
    tensor_fp64 = torch.randn(64, 64, dtype=torch.complex128, device="cuda")
    
    # Should raise assertion error
    with pytest.raises(AssertionError, match="must be complex64"):
        assert_fp32_cuda(tensor_fp64, "fp64 tensor")


def test_cpu_tensor_unchanged():
    """Test that CPU tensors are not forced to FP32."""
    # Create FP64 tensor on CPU
    tensor_fp64 = torch.randn(64, 64, dtype=torch.complex128, device="cpu")
    
    # Enforce should not change CPU tensors
    tensor_result = enforce_fp32_cuda(tensor_fp64, "cpu tensor")
    
    # Should still be FP64
    assert tensor_result.dtype == torch.complex128, "CPU tensor should remain FP64"


def test_fft_precision_cuda():
    """Test FFT wrapper enforces FP32 on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create field on CUDA
    field = torch.randn(64, 64, dtype=torch.complex128, device="cuda")
    
    # FFT with precision enforcement
    spectrum = fft2_with_precision(field, inverse=False, enforce_fp32=True)
    
    # Check dtype
    assert spectrum.dtype == torch.complex64, "FFT output should be complex64 on CUDA"
    
    # Inverse FFT
    field_back = fft2_with_precision(spectrum, inverse=True, enforce_fp32=True)
    assert field_back.dtype == torch.complex64, "IFFT output should be complex64 on CUDA"


def test_get_precision_dtype():
    """Test precision dtype selection based on device."""
    # CUDA should always get FP32
    if torch.cuda.is_available():
        cuda_device = torch.device("cuda")
        assert get_precision_dtype(cuda_device, is_complex=True) == torch.complex64
        assert get_precision_dtype(cuda_device, is_complex=False) == torch.float32
    
    # CPU can use FP64
    cpu_device = torch.device("cpu")
    assert get_precision_dtype(cpu_device, is_complex=True) == torch.complex128
    assert get_precision_dtype(cpu_device, is_complex=False) == torch.float64


def test_bpm_vector_wide_cuda_precision():
    """Test BPM vector wide solver uses FP32 on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create test field
    nx = ny = 64
    field = torch.randn(ny, nx, dtype=torch.complex128, device="cuda")
    
    # Create plan
    plan = {
        "dx_um": 0.5,
        "dy_um": 0.5,
        "dz_list_um": [10.0],
        "wavelengths_um": np.array([0.55]),
        "na_max": 0.25,
    }
    
    # Run propagation
    output = bpm_vector_wide.run(field, plan)
    
    # Check output is FP32
    assert output.dtype == torch.complex64, "BPM output should be complex64 on CUDA"
    
    # Validate precision invariants
    validation = validate_precision_invariants(field.to(torch.complex64), output)
    assert validation["all_valid"], f"Precision validation failed: {validation['errors']}"


def test_bpm_split_step_cuda_precision():
    """Test split-step Fourier BPM uses FP32 on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create test field
    nx = ny = 64
    field = torch.randn(ny, nx, dtype=torch.complex128, device="cuda")
    
    # Create plan
    plan = {
        "dx_um": 0.5,
        "dy_um": 0.5,
        "dz_list_um": [10.0],
        "wavelengths_um": np.array([0.55]),
        "na_max": 0.25,
    }
    
    # Run propagation
    output = bpm_split_step_fourier.run(field, plan)
    
    # Check output is FP32
    assert output.dtype == torch.complex64, "Split-step output should be complex64 on CUDA"


def test_as_multi_slice_cuda_precision():
    """Test angular spectrum uses FP32 on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create test field
    nx = ny = 64
    field = torch.randn(ny, nx, dtype=torch.complex128, device="cuda")
    
    # Create plan
    plan = {
        "dx_um": 0.5,
        "dy_um": 0.5,
        "dz_list_um": [10.0],
        "wavelengths_um": np.array([0.55]),
        "na_max": 0.25,
    }
    
    # Run propagation
    output = as_multi_slice.run(field, plan)
    
    # Check output is FP32
    assert output.dtype == torch.complex64, "AS output should be complex64 on CUDA"


def test_as_multi_slice_cpu_fp64_intermediates():
    """Test angular spectrum maintains FP64 intermediates on CPU."""
    # Create test field on CPU
    nx = ny = 64
    field = torch.randn(ny, nx, dtype=torch.complex64, device="cpu")
    
    # Create plan
    plan = {
        "dx_um": 0.5,
        "dy_um": 0.5,
        "dz_list_um": [10.0],
        "wavelengths_um": np.array([0.55]),
        "na_max": 0.25,
    }
    
    # Run propagation (internally uses FP64)
    output = as_multi_slice.run(field, plan)
    
    # Output should be complex64 but computation used FP64
    assert output.dtype == torch.complex64, "AS output should be complex64"
    
    # The internal computation should maintain high accuracy
    # Test by comparing with known analytical result
    wavelength_um = 0.55
    waist_um = 10.0
    z_um = 10.0
    
    E0, Ez_analytical = gaussian_free_space(
        wavelength_um, waist_um, z_um, nx, ny, 
        plan["dx_um"], plan["dy_um"]
    )
    
    # Propagate
    Ez_computed = as_multi_slice.run(E0, plan)
    
    # Should have good accuracy due to FP64 intermediates
    l2_err = l2_field_error(Ez_computed, Ez_analytical)
    assert l2_err < 0.03, f"L2 error {l2_err:.3%} exceeds 3% (CPU FP64 path)"


def test_cpu_gpu_parity():
    """Test CPU vs GPU results are within tolerance gates."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create identical fields on CPU and GPU
    nx = ny = 128
    wavelength_um = 0.55
    waist_um = 10.0
    z_um = 50.0
    dx = dy = 0.5
    
    # Get analytical reference
    E0_cpu, Ez_analytical = gaussian_free_space(
        wavelength_um, waist_um, z_um, nx, ny, dx, dy
    )
    
    # Copy to GPU
    E0_gpu = E0_cpu.to("cuda")
    
    # Create plan
    plan = {
        "dx_um": dx,
        "dy_um": dy,
        "dz_list_um": [z_um],
        "wavelengths_um": np.array([wavelength_um]),
        "na_max": 0.25,
    }
    
    # Test each solver
    solvers = [
        (bpm_vector_wide, "BPM Vector Wide"),
        (bpm_split_step_fourier, "Split-step Fourier"),
        (as_multi_slice, "Angular Spectrum"),
    ]
    
    for solver, name in solvers:
        # CPU propagation
        Ez_cpu = solver.run(E0_cpu, plan)
        
        # GPU propagation
        Ez_gpu = solver.run(E0_gpu, plan)
        
        # Check dtypes
        assert Ez_cpu.dtype == torch.complex64, f"{name} CPU output should be complex64"
        assert Ez_gpu.dtype == torch.complex64, f"{name} GPU output should be complex64"
        
        # Compare with analytical
        l2_cpu = l2_field_error(Ez_cpu, Ez_analytical)
        l2_gpu = l2_field_error(Ez_gpu.cpu(), Ez_analytical)
        
        # Both should meet tolerance
        assert l2_cpu <= 0.03, f"{name} CPU L2 error {l2_cpu:.3%} exceeds 3%"
        assert l2_gpu <= 0.03, f"{name} GPU L2 error {l2_gpu:.3%} exceeds 3%"
        
        # CPU and GPU should be close to each other
        cpu_gpu_diff = l2_field_error(Ez_gpu.cpu(), Ez_cpu)
        assert cpu_gpu_diff <= 0.01, f"{name} CPU-GPU difference {cpu_gpu_diff:.3%} exceeds 1%"
        
        print(f"✓ {name}: CPU L2={l2_cpu:.3%}, GPU L2={l2_gpu:.3%}, Diff={cpu_gpu_diff:.3%}")


def test_precision_invariants_tracking():
    """Test tracking of precision invariants through propagation."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Create fields
    field_in = torch.randn(64, 64, dtype=torch.complex64, device="cuda")
    field_out = torch.randn(64, 64, dtype=torch.complex64, device="cuda")
    
    # Create some intermediate spectra
    spectra = [
        torch.randn(64, 64, dtype=torch.complex64, device="cuda"),
        torch.randn(64, 64, dtype=torch.complex64, device="cuda"),
    ]
    
    # Validate - should pass
    result = validate_precision_invariants(field_in, field_out, spectra)
    assert result["all_valid"], "Valid tensors should pass validation"
    
    # Now test with wrong dtype
    field_bad = torch.randn(64, 64, dtype=torch.complex128, device="cuda")
    result = validate_precision_invariants(field_bad, field_out, spectra)
    assert not result["all_valid"], "FP64 on CUDA should fail validation"
    assert "complex128" in str(result["errors"]), "Error should mention wrong dtype"


def test_no_mixed_precision_in_solvers():
    """Verify no mixed precision code paths are active."""
    # Check that use_mixed flags are False or removed
    
    # Create small test case
    field = torch.randn(32, 32, dtype=torch.complex64)
    plan = {
        "dx_um": 0.5,
        "dy_um": 0.5,
        "dz_list_um": [10.0],
        "wavelengths_um": np.array([0.55]),
        "na_max": 0.25,
        "use_mixed_precision": True,  # Try to enable (should be ignored)
    }
    
    # Run solvers - they should ignore the mixed precision flag
    output1 = bpm_vector_wide.run(field, plan)
    output2 = bpm_split_step_fourier.run(field, plan)
    
    # Outputs should be same dtype as input (no mixed precision)
    assert output1.dtype == field.dtype
    assert output2.dtype == field.dtype


if __name__ == "__main__":
    print("Running precision policy tests...")
    
    # Basic tests
    test_mixed_fft_disabled()
    print("✓ Mixed FFT disabled")
    
    test_cpu_tensor_unchanged()
    print("✓ CPU tensors unchanged")
    
    test_get_precision_dtype()
    print("✓ Precision dtype selection")
    
    # CUDA tests if available
    if torch.cuda.is_available():
        test_enforce_fp32_cuda()
        print("✓ FP32 enforcement on CUDA")
        
        test_assert_fp32_cuda_fails()
        print("✓ FP32 assertion works")
        
        test_fft_precision_cuda()
        print("✓ FFT precision on CUDA")
        
        test_bpm_vector_wide_cuda_precision()
        print("✓ BPM vector wide FP32 on CUDA")
        
        test_bpm_split_step_cuda_precision()
        print("✓ Split-step Fourier FP32 on CUDA")
        
        test_as_multi_slice_cuda_precision()
        print("✓ Angular spectrum FP32 on CUDA")
        
        test_cpu_gpu_parity()
        print("✓ CPU-GPU parity within gates")
        
        test_precision_invariants_tracking()
        print("✓ Precision invariants tracking")
    else:
        print("⚠ CUDA not available, skipping GPU tests")
    
    test_as_multi_slice_cpu_fp64_intermediates()
    print("✓ AS maintains FP64 intermediates on CPU")
    
    test_no_mixed_precision_in_solvers()
    print("✓ No mixed precision before M2")
    
    print("\n✅ All precision policy tests passed!")
