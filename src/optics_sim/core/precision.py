"""Precision Policy for Optics Simulation

This module enforces the precision policy:
- CUDA tensors: Always FP32 (complex64) for all operations
- CPU tensors: FP64 (complex128) intermediates allowed for AS solver
- Mixed precision: Disabled until M2 sign-off
"""

import torch


# Global flag for mixed precision (DISABLED until M2)
MIXED_FFT = False  # DO NOT ENABLE before M2 sign-off


def enforce_fp32_cuda(tensor: torch.Tensor, name: str = "tensor") -> torch.Tensor:
    """Enforce FP32 precision for CUDA tensors.
    
    Args:
        tensor: Input tensor
        name: Tensor name for error messages
        
    Returns:
        FP32 tensor on CUDA, unchanged on CPU
    """
    if tensor.is_cuda:
        if tensor.is_complex():
            if tensor.dtype != torch.complex64:
                return tensor.to(torch.complex64)
        else:
            if tensor.dtype != torch.float32:
                return tensor.to(torch.float32)
    return tensor


def assert_fp32_cuda(tensor: torch.Tensor, name: str = "tensor") -> None:
    """Assert that CUDA tensors are FP32.
    
    Args:
        tensor: Tensor to check
        name: Tensor name for error messages
        
    Raises:
        AssertionError: If CUDA tensor is not FP32
    """
    if tensor.is_cuda:
        if tensor.is_complex():
            assert tensor.dtype == torch.complex64, (
                f"{name} on CUDA must be complex64, got {tensor.dtype}"
            )
        else:
            assert tensor.dtype == torch.float32, (
                f"{name} on CUDA must be float32, got {tensor.dtype}"
            )


def fft2_with_precision(
    field: torch.Tensor, 
    inverse: bool = False,
    enforce_fp32: bool = True
) -> torch.Tensor:
    """FFT2 wrapper with precision enforcement.
    
    Args:
        field: Input field
        inverse: If True, perform inverse FFT
        enforce_fp32: If True, enforce FP32 on CUDA
        
    Returns:
        FFT result with appropriate precision
    """
    if enforce_fp32 and field.is_cuda:
        field = enforce_fp32_cuda(field, "FFT input")
    
    if inverse:
        result = torch.fft.ifft2(field)
    else:
        result = torch.fft.fft2(field)
    
    if enforce_fp32 and result.is_cuda:
        result = enforce_fp32_cuda(result, "FFT output")
    
    return result


def get_precision_dtype(device: torch.device, is_complex: bool = True) -> torch.dtype:
    """Get appropriate dtype based on device and precision policy.
    
    Args:
        device: Computation device
        is_complex: If True, return complex dtype
        
    Returns:
        Appropriate dtype for the device
    """
    if device.type == "cuda":
        return torch.complex64 if is_complex else torch.float32
    else:
        # CPU can use higher precision for intermediates
        return torch.complex128 if is_complex else torch.float64


def validate_precision_invariants(
    field_in: torch.Tensor,
    field_out: torch.Tensor,
    intermediate_spectra: list[torch.Tensor] = None
) -> dict:
    """Validate precision invariants throughout propagation.
    
    Args:
        field_in: Input field
        field_out: Output field
        intermediate_spectra: Optional list of FFT spectra
        
    Returns:
        Dict with validation results
    """
    results = {
        "input_valid": True,
        "output_valid": True,
        "spectra_valid": True,
        "errors": []
    }
    
    # Check input
    if field_in.is_cuda:
        if field_in.dtype != torch.complex64:
            results["input_valid"] = False
            results["errors"].append(f"Input on CUDA is {field_in.dtype}, expected complex64")
    
    # Check output
    if field_out.is_cuda:
        if field_out.dtype != torch.complex64:
            results["output_valid"] = False
            results["errors"].append(f"Output on CUDA is {field_out.dtype}, expected complex64")
    
    # Check intermediate spectra
    if intermediate_spectra:
        for i, spectrum in enumerate(intermediate_spectra):
            if spectrum.is_cuda and spectrum.dtype != torch.complex64:
                results["spectra_valid"] = False
                results["errors"].append(
                    f"Spectrum {i} on CUDA is {spectrum.dtype}, expected complex64"
                )
    
    results["all_valid"] = (
        results["input_valid"] and 
        results["output_valid"] and 
        results["spectra_valid"]
    )
    
    return results
