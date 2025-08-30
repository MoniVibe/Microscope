"""Field resampling and interpolation utilities.

Provides phase-preserving interpolation with anti-aliasing and Nyquist checks.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np


def resample(field: torch.Tensor, 
            from_pitch_um: float, 
            to_pitch_um: float,
            method: str = 'fourier',
            anti_alias: bool = True) -> torch.Tensor:
    """Resample complex field to different grid spacing.
    
    Preserves phase information and applies anti-aliasing when downsampling.
    
    Args:
        field: Complex field tensor of shape (..., ny, nx)
        from_pitch_um: Original grid spacing in micrometers
        to_pitch_um: Target grid spacing in micrometers
        method: Interpolation method ('fourier', 'bilinear', 'bicubic')
        anti_alias: Apply anti-aliasing filter when downsampling
        
    Returns:
        Resampled field with same physical extent but different resolution
        
    Raises:
        ValueError: If resampling violates Nyquist criterion
    """
    # Get dimensions
    original_shape = field.shape
    if field.dim() < 2:
        raise ValueError(f"Field must be at least 2D, got shape {field.shape}")
    
    ny_old, nx_old = field.shape[-2:]
    
    # Calculate new grid size to maintain physical extent
    # Physical extent = N * pitch
    extent_y = ny_old * from_pitch_um
    extent_x = nx_old * from_pitch_um
    
    ny_new = int(round(extent_y / to_pitch_um))
    nx_new = int(round(extent_x / to_pitch_um))
    
    # Check Nyquist criterion
    ratio = to_pitch_um / from_pitch_um
    if ratio > 2.0:
        print(f"Warning: Resampling ratio {ratio:.2f} may violate Nyquist criterion")
    
    # Reshape for processing
    if field.dim() > 2:
        batch_shape = field.shape[:-2]
        field = field.reshape(-1, ny_old, nx_old)
        batched = True
    else:
        field = field.unsqueeze(0)
        batched = False
    
    # Apply resampling based on method
    if method == 'fourier':
        resampled = _resample_fourier(field, ny_new, nx_new, anti_alias)
    elif method == 'bilinear':
        resampled = _resample_spatial(field, ny_new, nx_new, 'bilinear', anti_alias)
    elif method == 'bicubic':
        resampled = _resample_spatial(field, ny_new, nx_new, 'bicubic', anti_alias)
    else:
        raise ValueError(f"Unknown resampling method: {method}")
    
    # Reshape back
    if batched:
        resampled = resampled.reshape(*batch_shape, ny_new, nx_new)
    else:
        resampled = resampled.squeeze(0)
    
    return resampled


def _resample_fourier(field: torch.Tensor, 
                     ny_new: int, 
                     nx_new: int,
                     anti_alias: bool = True) -> torch.Tensor:
    """Resample using Fourier interpolation.
    
    Best for preserving phase and avoiding artifacts.
    
    Args:
        field: Complex field (batch, ny_old, nx_old)
        ny_new: New height
        nx_new: New width
        anti_alias: Apply anti-aliasing
        
    Returns:
        Resampled field (batch, ny_new, nx_new)
    """
    batch, ny_old, nx_old = field.shape
    
    # FFT to frequency domain
    field_fft = torch.fft.fft2(field)
    
    # Shift zero frequency to center
    field_fft = torch.fft.fftshift(field_fft, dim=(-2, -1))
    
    # Anti-aliasing filter if downsampling
    if anti_alias and (ny_new < ny_old or nx_new < nx_old):
        # Create low-pass filter
        fy = torch.fft.fftfreq(ny_old, d=1.0, device=field.device)
        fx = torch.fft.fftfreq(nx_old, d=1.0, device=field.device)
        fy = torch.fft.fftshift(fy)
        fx = torch.fft.fftshift(fx)
        
        # Cutoff at new Nyquist frequency
        fy_cut = 0.5 * min(1.0, ny_new / ny_old)
        fx_cut = 0.5 * min(1.0, nx_new / nx_old)
        
        # Smooth filter (Butterworth-like)
        fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
        filter_2d = 1.0 / (1.0 + (torch.abs(fy_grid/fy_cut)**6 + torch.abs(fx_grid/fx_cut)**6))
        
        field_fft = field_fft * filter_2d.unsqueeze(0)
    
    # Pad or crop in frequency domain
    if ny_new != ny_old or nx_new != nx_old:
        # Calculate padding/cropping
        pad_y = (ny_new - ny_old) // 2
        pad_x = (nx_new - nx_old) // 2
        
        if ny_new > ny_old or nx_new > nx_old:
            # Pad with zeros (interpolation)
            padding = [
                max(0, pad_x), max(0, nx_new - nx_old - pad_x),
                max(0, pad_y), max(0, ny_new - ny_old - pad_y)
            ]
            field_fft_resized = F.pad(field_fft, padding, mode='constant', value=0)
        else:
            # Crop (decimation)
            start_y = max(0, -pad_y)
            start_x = max(0, -pad_x)
            end_y = start_y + ny_new
            end_x = start_x + nx_new
            field_fft_resized = field_fft[:, start_y:end_y, start_x:end_x]
    else:
        field_fft_resized = field_fft
    
    # Shift back
    field_fft_resized = torch.fft.ifftshift(field_fft_resized, dim=(-2, -1))
    
    # Inverse FFT
    resampled = torch.fft.ifft2(field_fft_resized)
    
    # Scale to preserve energy
    scale_factor = (ny_new * nx_new) / (ny_old * nx_old)
    resampled = resampled * np.sqrt(scale_factor)
    
    return resampled


def _resample_spatial(field: torch.Tensor,
                     ny_new: int,
                     nx_new: int,
                     mode: str = 'bilinear',
                     anti_alias: bool = True) -> torch.Tensor:
    """Resample using spatial interpolation.
    
    Works separately on real and imaginary parts.
    
    Args:
        field: Complex field (batch, ny_old, nx_old)
        ny_new: New height
        nx_new: New width
        mode: Interpolation mode ('bilinear' or 'bicubic')
        anti_alias: Apply anti-aliasing
        
    Returns:
        Resampled field (batch, ny_new, nx_new)
    """
    batch, ny_old, nx_old = field.shape
    
    # Split into real and imaginary
    field_real = field.real
    field_imag = field.imag
    
    # Add channel dimension for F.interpolate
    field_real = field_real.unsqueeze(1)  # (batch, 1, ny, nx)
    field_imag = field_imag.unsqueeze(1)
    
    # Anti-aliasing pre-filter if downsampling
    if anti_alias and (ny_new < ny_old or nx_new < nx_old):
        # Apply Gaussian blur
        sigma = max(ny_old/ny_new, nx_old/nx_new) / 2
        field_real = _gaussian_blur(field_real, sigma)
        field_imag = _gaussian_blur(field_imag, sigma)
    
    # Interpolate
    resampled_real = F.interpolate(
        field_real, 
        size=(ny_new, nx_new),
        mode=mode,
        align_corners=False
    )
    resampled_imag = F.interpolate(
        field_imag,
        size=(ny_new, nx_new),
        mode=mode,
        align_corners=False
    )
    
    # Remove channel dimension and recombine
    resampled_real = resampled_real.squeeze(1)
    resampled_imag = resampled_imag.squeeze(1)
    resampled = torch.complex(resampled_real, resampled_imag)
    
    return resampled


def _gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur for anti-aliasing.
    
    Args:
        tensor: Input tensor (batch, channels, height, width)
        sigma: Gaussian standard deviation
        
    Returns:
        Blurred tensor
    """
    if sigma <= 0:
        return tensor
    
    # Create Gaussian kernel
    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # 1D Gaussian
    x = torch.arange(kernel_size, dtype=torch.float32, device=tensor.device)
    x = x - (kernel_size - 1) / 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # 2D kernel via outer product
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, k, k)
    
    # Apply convolution
    batch, channels, height, width = tensor.shape
    
    # Reshape for depthwise convolution
    tensor_reshaped = tensor.reshape(batch * channels, 1, height, width)
    
    # Pad to handle boundaries
    padding = kernel_size // 2
    tensor_padded = F.pad(tensor_reshaped, [padding] * 4, mode='reflect')
    
    # Convolve
    blurred = F.conv2d(tensor_padded, kernel_2d, padding=0)
    
    # Reshape back
    blurred = blurred.reshape(batch, channels, height, width)
    
    return blurred


def check_nyquist(field: torch.Tensor,
                 pitch_um: float,
                 wavelength_um: float,
                 na: float) -> bool:
    """Check if sampling satisfies Nyquist criterion.
    
    Args:
        field: Complex field tensor
        pitch_um: Grid spacing in micrometers
        wavelength_um: Wavelength in micrometers
        na: Numerical aperture
        
    Returns:
        True if Nyquist criterion is satisfied
    """
    # Nyquist limit
    pitch_nyquist = wavelength_um / (2 * na)
    
    # Check with 10% margin
    return pitch_um <= pitch_nyquist * 1.1


def compute_bandwidth(field: torch.Tensor,
                     pitch_um: float) -> Tuple[float, float]:
    """Compute spatial frequency bandwidth of field.
    
    Args:
        field: Complex field tensor (..., ny, nx)
        pitch_um: Grid spacing in micrometers
        
    Returns:
        Tuple of (fx_max, fy_max) in cycles/µm
    """
    # Get last two dimensions
    ny, nx = field.shape[-2:]
    
    # FFT
    field_fft = torch.fft.fft2(field, dim=(-2, -1))
    power = torch.abs(field_fft)**2
    
    # Frequency grids
    fx = torch.fft.fftfreq(nx, d=pitch_um, device=field.device)
    fy = torch.fft.fftfreq(ny, d=pitch_um, device=field.device)
    fy_grid, fx_grid = torch.meshgrid(fy, fx, indexing='ij')
    
    # Find maximum frequency with significant power
    # Use 1% of peak power as threshold
    threshold = 0.01 * power.max()
    mask = power > threshold
    
    if mask.any():
        fx_max = torch.abs(fx_grid[mask]).max().item()
        fy_max = torch.abs(fy_grid[mask]).max().item()
    else:
        fx_max = fy_max = 0.0
    
    return fx_max, fy_max


def adaptive_resample(field: torch.Tensor,
                     from_pitch_um: float,
                     to_pitch_um: float,
                     wavelength_um: float,
                     na: float) -> torch.Tensor:
    """Adaptively resample based on field content and NA.
    
    Automatically selects best method and parameters.
    
    Args:
        field: Complex field tensor
        from_pitch_um: Original grid spacing
        to_pitch_um: Target grid spacing
        wavelength_um: Wavelength
        na: Numerical aperture
        
    Returns:
        Resampled field
    """
    # Check current bandwidth
    fx_max, fy_max = compute_bandwidth(field, from_pitch_um)
    f_max = max(fx_max, fy_max)
    
    # NA-limited bandwidth
    f_na = na / wavelength_um
    
    # Choose method based on content
    if f_max > 0.8 * f_na:
        # High frequency content - use Fourier method
        method = 'fourier'
    else:
        # Lower frequency - spatial method is faster
        method = 'bicubic'
    
    # Always use anti-aliasing when downsampling
    anti_alias = to_pitch_um > from_pitch_um
    
    return resample(field, from_pitch_um, to_pitch_um, method, anti_alias)
