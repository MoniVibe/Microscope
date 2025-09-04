"""
M3 Split-Step BPM solver with strict FP32 enforcement
For Microscope project integration
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class M3SplitStepBPMSolver:
    """
    M3-compliant Split-Step Beam Propagation Method solver

    Implements split-step BPM for inhomogeneous media with
    enforced FP32/complex64 precision for deterministic GPU execution.
    """

    def __init__(
        self,
        wavelength: float,
        n_medium: float = 1.0,
        n_steps: int = 50,
        device: torch.device | None = None,
        enforce_fp32: bool = True,
        mixed_fft: bool = False,
        seed: int = 1337,
        **kwargs,
    ):
        """
        Initialize M3 Split-Step BPM solver

        Args:
            wavelength: Wavelength in meters
            n_medium: Background refractive index
            n_steps: Number of propagation steps
            device: Compute device (cuda/cpu)
            enforce_fp32: Enforce strict FP32 (required for M3)
            mixed_fft: Allow mixed precision FFT (default False for M3)
            seed: Random seed for determinism
        """
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.n_steps = n_steps
        self.k = 2 * np.pi * n_medium / wavelength
        self.seed = seed

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enforce_fp32 = enforce_fp32
        self.mixed_fft = mixed_fft

        # M3: Set deterministic mode
        self._set_deterministic()

        logger.info("M3 Split-Step BPM solver initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Wavelength: {wavelength*1e9:.1f} nm")
        logger.info(f"  Background n: {n_medium}")
        logger.info(f"  Steps: {n_steps}")
        logger.info(f"  FP32 enforcement: {enforce_fp32}")
        logger.info(f"  Seed: {seed}")

    def _set_deterministic(self):
        """Configure deterministic execution for M3"""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        if self.enforce_fp32:
            torch.set_float32_matmul_precision("highest")
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                torch.use_deterministic_algorithms(True, warn_only=True)

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

    def _create_propagator(self, N: int, dx: float, dz: float) -> torch.Tensor:
        """
        Create free-space propagator for split-step (paraxial approximation)

        Args:
            N: Grid size
            dx: Transverse grid spacing
            dz: Propagation step size

        Returns:
            Half-step propagator in k-space
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

        # Paraxial approximation: kz ≈ k - (kx^2 + ky^2)/(2k)
        kz = self.k - (kx**2 + ky**2) / (2 * self.k)

        # Half-step propagator for split-step method
        propagator = torch.exp(1j * kz * dz / 2)

        # Ensure complex64
        propagator = self._ensure_fp32(propagator)

        return propagator

    def _apply_phase_screen(
        self, field: torch.Tensor, n_profile: torch.Tensor, dz: float
    ) -> torch.Tensor:
        """
        Apply phase screen for refractive index variation

        Args:
            field: Input field
            n_profile: Refractive index profile (real)
            dz: Step size

        Returns:
            Field after phase screen
        """
        # Ensure FP32
        field = self._ensure_fp32(field)
        n_profile = self._ensure_fp32(n_profile)

        # Phase shift due to index variation
        k0 = 2 * np.pi / self.wavelength
        phase_shift = k0 * (n_profile - self.n_medium) * dz

        # Apply phase screen
        field_out = field * torch.exp(1j * phase_shift)

        # Ensure complex64
        field_out = self._ensure_fp32(field_out)

        return field_out

    def propagate(
        self,
        field: torch.Tensor,
        z_distance: float,
        dx: float | None = None,
        n_profile: Callable | None = None,
    ) -> torch.Tensor:
        """
        Propagate field using M3-compliant Split-Step BPM

        Args:
            field: Input field (complex tensor)
            z_distance: Total propagation distance in meters
            dx: Grid spacing in meters
            n_profile: Optional function returning n(z) profile

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

        # Step size
        dz = z_distance / self.n_steps

        logger.debug(
            f"M3 Split-step: z={z_distance*1e3:.1f} mm, steps={self.n_steps}, dz={dz*1e6:.1f} μm"
        )

        # Create half-step propagator
        propagator = self._create_propagator(N, dx, dz)
        propagator = propagator.unsqueeze(0).expand(B, -1, -1)

        # Initialize field
        current_field = field.clone()

        # Split-step propagation loop
        for step in range(self.n_steps):
            z = step * dz

            # Step 1: Half-step free-space propagation (in k-space)
            if self.mixed_fft and not self.enforce_fp32:
                with torch.cuda.amp.autocast(enabled=True):
                    field_fft = torch.fft.fft2(current_field, norm="ortho")
            else:
                field_fft = torch.fft.fft2(current_field, norm="ortho")
                field_fft = self._ensure_fp32(field_fft)

            field_fft = field_fft * propagator
            current_field = torch.fft.ifft2(field_fft, norm="ortho")
            current_field = self._ensure_fp32(current_field)

            # Step 2: Apply phase screen if n_profile provided
            if n_profile is not None:
                n_z = n_profile(z + dz / 2)
                if n_z is not None:
                    n_z = self._ensure_fp32(n_z)
                    if n_z.ndim == 2:
                        n_z = n_z.unsqueeze(0).expand(B, -1, -1)
                    current_field = self._apply_phase_screen(current_field, n_z, dz)

            # Step 3: Half-step free-space propagation
            field_fft = torch.fft.fft2(current_field, norm="ortho")
            field_fft = self._ensure_fp32(field_fft)
            field_fft = field_fft * propagator
            current_field = torch.fft.ifft2(field_fft, norm="ortho")
            current_field = self._ensure_fp32(current_field)

            # Progress logging
            if (step + 1) % max(1, self.n_steps // 10) == 0:
                logger.debug(f"  Step {step+1}/{self.n_steps} (z={z*1e6:.1f} μm)")

        # Restore original shape
        if len(original_shape) == 2:
            current_field = current_field.squeeze(0)

        # Verify energy conservation
        if logger.isEnabledFor(logging.DEBUG):
            energy_in = torch.sum(torch.abs(field) ** 2).item()
            energy_out = torch.sum(torch.abs(current_field) ** 2).item()
            energy_ratio = energy_out / (energy_in + 1e-10)
            logger.debug(f"Energy conservation: {energy_ratio:.6f}")

            if abs(energy_ratio - 1.0) > 0.01:
                logger.warning(f"M3: Energy deviation {abs(1-energy_ratio)*100:.2f}%")

        return current_field

    def propagate_grating(
        self,
        field: torch.Tensor,
        z_distance: float,
        dx: float,
        grating_period: float,
        grating_depth: float,
        grating_n: float = 1.5,
    ) -> torch.Tensor:
        """
        Propagate through a binary phase grating (M3-compliant)

        Args:
            field: Input field
            z_distance: Propagation distance after grating
            dx: Grid spacing
            grating_period: Grating period in meters
            grating_depth: Grating depth in meters
            grating_n: Grating refractive index

        Returns:
            Field after grating and propagation
        """
        field = self._ensure_fp32(field)
        field = field.to(self.device)

        N = field.shape[-1]

        # Create grating phase profile in FP32
        x = torch.arange(N, dtype=torch.float32, device=self.device) * dx
        x = x - x.mean()  # Center

        # Binary grating pattern
        grating_phase = torch.zeros(N, dtype=torch.float32, device=self.device)
        mask = (x % grating_period) < (grating_period / 2)
        grating_phase[mask] = 1.0

        # Calculate phase shift
        k0 = 2 * np.pi / self.wavelength
        phase_shift = k0 * (grating_n - self.n_medium) * grating_depth * grating_phase

        # Expand to 2D
        phase_screen = phase_shift.unsqueeze(0).expand(N, -1)

        # Apply grating
        field_after_grating = field * torch.exp(1j * phase_screen)
        field_after_grating = self._ensure_fp32(field_after_grating)

        # Propagate after grating
        return self.propagate(field_after_grating, z_distance, dx)

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
            "seed": self.seed,
        }

        # Check all requirements
        compliance["m3_compliant"] = (
            compliance["fp32_enforced"]
            and compliance["mixed_fft_disabled"]
            and compliance["deterministic"]
            and compliance["matmul_precision"] == "highest"
        )

        return compliance
