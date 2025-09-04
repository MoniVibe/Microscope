"""
M3 Angular Spectrum propagation solver with strict FP32 enforcement
For Microscope project integration
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class M3AngularSpectrumSolver:
    """
    M3-compliant Angular Spectrum solver with strict FP32 precision

    Implements the angular spectrum method for free-space propagation
    with enforced FP32/complex64 precision for deterministic GPU execution.
    """

    def __init__(
        self,
        wavelength: float,
        n_medium: float = 1.0,
        NA: float = 0.5,
        device: torch.device | None = None,
        enforce_fp32: bool = True,
        mixed_fft: bool = False,
        **kwargs,
    ):
        """
        Initialize M3 Angular Spectrum solver

        Args:
            wavelength: Wavelength in meters
            n_medium: Refractive index
            NA: Numerical aperture
            device: Compute device (cuda/cpu)
            enforce_fp32: Enforce strict FP32 (required for M3)
            mixed_fft: Allow mixed precision FFT (default False for M3)
        """
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.NA = NA
        self.k = 2 * np.pi * n_medium / wavelength
        self.k_max = 2 * np.pi * NA / wavelength

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enforce_fp32 = enforce_fp32
        self.mixed_fft = mixed_fft

        # M3: Enforce FP32 settings
        if self.enforce_fp32:
            torch.set_float32_matmul_precision("highest")
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

        logger.info("M3 Angular Spectrum solver initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Wavelength: {wavelength*1e9:.1f} nm")
        logger.info(f"  NA: {NA}")
        logger.info(f"  FP32 enforcement: {enforce_fp32}")

    def _ensure_fp32(self, tensor: torch.Tensor) -> torch.Tensor:
        """Ensure tensor is FP32/complex64 for M3 compliance"""
        if not self.enforce_fp32:
            return tensor

        if torch.is_complex(tensor):
            if tensor.dtype != torch.complex64:
                return tensor.to(torch.complex64)
        elif tensor.dtype != torch.float32:
            return tensor.to(torch.float32)
        return tensor

    def _create_transfer_function(self, N: int, dx: float, z: float) -> torch.Tensor:
        """
        Create angular spectrum transfer function with FP32 precision

        Args:
            N: Grid size
            dx: Grid spacing in meters
            z: Propagation distance in meters

        Returns:
            Transfer function H(kx, ky, z)
        """
        # Create frequency grid in FP32
        dfx = 1.0 / (N * dx)
        fx = torch.arange(N, dtype=torch.float32, device=self.device) - N // 2
        fx = fx * dfx

        # Create 2D frequency meshgrid
        FX, FY = torch.meshgrid(fx, fx, indexing="xy")

        # Shift for FFT convention
        FX = torch.fft.ifftshift(FX)
        FY = torch.fft.ifftshift(FY)

        # Convert to k-space
        kx = 2 * np.pi * FX
        ky = 2 * np.pi * FY

        # Calculate kz with proper handling of evanescent waves
        k_squared = self.k**2
        kz_squared = k_squared - kx**2 - ky**2

        # Create complex kz
        kz = torch.zeros_like(kz_squared, dtype=torch.complex64)

        # Propagating waves
        propagating = kz_squared >= 0
        kz[propagating] = torch.sqrt(kz_squared[propagating] + 0j)

        # Evanescent waves (exponential decay)
        kz[~propagating] = 1j * torch.sqrt(-kz_squared[~propagating])

        # Apply NA filtering
        k_transverse = torch.sqrt(kx**2 + ky**2)
        na_filter = k_transverse <= self.k_max

        # Transfer function
        H = torch.exp(1j * kz * z)
        H = H * na_filter.to(torch.complex64)

        # Ensure FP32/complex64
        H = self._ensure_fp32(H)

        return H

    def propagate(
        self, field: torch.Tensor, z_distance: float, dx: float | None = None
    ) -> torch.Tensor:
        """
        Propagate field using M3-compliant angular spectrum method

        Args:
            field: Input field (complex tensor)
            z_distance: Propagation distance in meters
            dx: Grid spacing in meters

        Returns:
            Propagated field with enforced FP32 precision
        """
        # Ensure FP32/complex64
        field = self._ensure_fp32(field)
        field = field.to(self.device)

        # Handle dimensions
        original_shape = field.shape
        if field.ndim == 2:
            field = field.unsqueeze(0)

        B, N, M = field.shape
        assert N == M, "Field must be square"

        # Default grid spacing
        if dx is None:
            dx = self.wavelength

        logger.debug(f"M3 AS propagation: z={z_distance*1e6:.1f} μm, grid={N}x{N}")

        # Create transfer function
        H = self._create_transfer_function(N, dx, z_distance)
        H = H.unsqueeze(0).expand(B, -1, -1)

        # Propagation with controlled precision
        if self.mixed_fft and not self.enforce_fp32:
            # Allow mixed precision (not recommended for M3)
            with torch.cuda.amp.autocast(enabled=True):
                field_fft = torch.fft.fft2(field, norm="ortho")
                field_fft = field_fft * H
                field_prop = torch.fft.ifft2(field_fft, norm="ortho")
        else:
            # Strict FP32 (M3 requirement)
            field_fft = torch.fft.fft2(field, norm="ortho")
            field_fft = self._ensure_fp32(field_fft)
            field_fft = field_fft * H
            field_prop = torch.fft.ifft2(field_fft, norm="ortho")
            field_prop = self._ensure_fp32(field_prop)

        # Restore original shape
        if len(original_shape) == 2:
            field_prop = field_prop.squeeze(0)

        # Verify energy conservation
        if logger.isEnabledFor(logging.DEBUG):
            energy_in = torch.sum(torch.abs(field) ** 2).item()
            energy_out = torch.sum(torch.abs(field_prop) ** 2).item()
            energy_ratio = energy_out / (energy_in + 1e-10)
            logger.debug(f"Energy conservation: {energy_ratio:.6f}")

            if abs(energy_ratio - 1.0) > 0.01:
                logger.warning(f"M3: Energy deviation {abs(1-energy_ratio)*100:.2f}%")

        return field_prop

    def get_memory_usage(self) -> dict[str, float]:
        """Get current GPU memory usage for M3 monitoring"""
        if torch.cuda.is_available() and self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            return {
                "allocated_bytes": allocated,
                "reserved_bytes": reserved,
                "allocated_gb": allocated / (1024**3),
                "reserved_gb": reserved / (1024**3),
            }
        return {"allocated_bytes": 0, "reserved_bytes": 0}

    def validate_m3_compliance(self) -> dict[str, Any]:
        """Validate M3 compliance settings"""
        compliance = {
            "fp32_enforced": self.enforce_fp32,
            "mixed_fft_disabled": not self.mixed_fft,
            "device": str(self.device),
            "deterministic": (
                torch.backends.cudnn.deterministic if torch.cuda.is_available() else True
            ),
            "matmul_precision": torch.get_float32_matmul_precision(),
        }

        # Check all requirements
        compliance["m3_compliant"] = (
            compliance["fp32_enforced"]
            and compliance["mixed_fft_disabled"]
            and compliance["deterministic"]
            and compliance["matmul_precision"] == "highest"
        )

        return compliance
