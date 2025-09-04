"""
M3 Wide-Angle BPM solver with strict FP32 enforcement
For Microscope project integration
"""

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


class M3WideAngleBPMSolver:
    """
    M3-compliant Wide-Angle Beam Propagation Method solver

    Implements wide-angle BPM using Padé approximants for high-NA systems
    with enforced FP32/complex64 precision for deterministic GPU execution.
    """

    def __init__(
        self,
        wavelength: float,
        n_medium: float = 1.0,
        NA: float = 0.85,
        pade_order: tuple[int, int] = (1, 1),
        n_steps: int = 100,
        device: torch.device | None = None,
        enforce_fp32: bool = True,
        mixed_fft: bool = False,
        seed: int = 1337,
        **kwargs,
    ):
        """
        Initialize M3 Wide-Angle BPM solver

        Args:
            wavelength: Wavelength in meters
            n_medium: Background refractive index
            NA: Numerical aperture
            pade_order: Padé approximant order (numerator, denominator)
            n_steps: Number of propagation steps
            device: Compute device (cuda/cpu)
            enforce_fp32: Enforce strict FP32 (required for M3)
            mixed_fft: Allow mixed precision FFT (default False for M3)
            seed: Random seed for determinism
        """
        self.wavelength = wavelength
        self.n_medium = n_medium
        self.NA = NA
        self.pade_order = pade_order
        self.n_steps = n_steps
        self.k = 2 * np.pi * n_medium / wavelength
        self.seed = seed

        # Padé coefficients for (1,1) approximant
        if pade_order == (1, 1):
            self.pade_a = 0.5
            self.pade_b = 0.25
        else:
            raise NotImplementedError(f"Padé order {pade_order} not implemented for M3")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.enforce_fp32 = enforce_fp32
        self.mixed_fft = mixed_fft

        # M3: Set deterministic mode
        self._set_deterministic()

        logger.info("M3 Wide-Angle BPM solver initialized")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Wavelength: {wavelength*1e9:.1f} nm")
        logger.info(f"  NA: {NA}")
        logger.info(f"  Padé order: {pade_order}")
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

    def _create_wide_angle_operator(
        self, N: int, dx: float, dz: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Create wide-angle propagation operators using Padé approximants

        Args:
            N: Grid size
            dx: Transverse grid spacing
            dz: Propagation step size

        Returns:
            Numerator and denominator operators for Padé approximant
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

        # Convert to normalized k-space
        kx_norm = 2 * np.pi * FX * self.wavelength / (2 * np.pi * self.n_medium)
        ky_norm = 2 * np.pi * FY * self.wavelength / (2 * np.pi * self.n_medium)

        # Ensure FP32
        kx_norm = self._ensure_fp32(kx_norm)
        ky_norm = self._ensure_fp32(ky_norm)

        # Transverse k-squared (normalized)
        k_perp_sq = kx_norm**2 + ky_norm**2

        # Padé (1,1) approximant for sqrt(1 + k_perp^2)
        # sqrt(1 + x) ≈ (1 + a*x) / (1 + b*x)
        # For (1,1): a = 0.5, b = 0.25

        # Phase factor
        phase = self.k * dz

        # Numerator: exp(i*k*dz * (1 + a*k_perp^2))
        numerator = torch.exp(1j * phase * (1 + self.pade_a * k_perp_sq))

        # Denominator: (1 + b*k_perp^2)
        denominator = 1 + self.pade_b * k_perp_sq

        # Ensure complex64
        numerator = self._ensure_fp32(numerator)
        denominator = self._ensure_fp32(denominator.to(torch.complex64))

        return numerator, denominator

    def _solve_implicit_step(
        self,
        field: torch.Tensor,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
        max_iter: int = 5,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        """
        Solve implicit propagation step using iterative method

        Args:
            field: Input field
            numerator: Numerator operator
            denominator: Denominator operator
            max_iter: Maximum iterations for implicit solve
            tol: Convergence tolerance

        Returns:
            Propagated field
        """
        # Transform to k-space
        if self.mixed_fft and not self.enforce_fp32:
            with torch.cuda.amp.autocast(enabled=True):
                field_fft = torch.fft.fft2(field, norm="ortho")
        else:
            field_fft = torch.fft.fft2(field, norm="ortho")
            field_fft = self._ensure_fp32(field_fft)

        # Apply Padé approximant
        # u_new = numerator * u_old / denominator
        field_fft_new = field_fft * numerator / denominator
        field_fft_new = self._ensure_fp32(field_fft_new)

        # Iterative refinement for higher accuracy (optional for M3)
        for iter_num in range(max_iter):
            field_fft_prev = field_fft_new.clone()

            # Refine solution
            residual = denominator * field_fft_new - numerator * field_fft
            correction = residual / (denominator + 1e-10)
            field_fft_new = field_fft_new - 0.5 * correction  # Relaxation factor
            field_fft_new = self._ensure_fp32(field_fft_new)

            # Check convergence
            error = torch.norm(field_fft_new - field_fft_prev) / (torch.norm(field_fft_new) + 1e-10)
            if error < tol:
                break

        # Transform back to real space
        field_new = torch.fft.ifft2(field_fft_new, norm="ortho")
        field_new = self._ensure_fp32(field_new)

        return field_new

    def propagate(
        self, field: torch.Tensor, z_distance: float, dx: float | None = None
    ) -> torch.Tensor:
        """
        Propagate field using M3-compliant Wide-Angle BPM

        Args:
            field: Input field (complex tensor)
            z_distance: Total propagation distance in meters
            dx: Grid spacing in meters

        Returns:
            Propagated field with enforced FP32 precision
        """
        # Ensure FP32/complex64
        field = self._ensure_fp32(field)
        field = field.to(self.device)

        # Handle dimensions
        original_shape = field.shape
        SPECTRAL_DIM = 2  # PLR2004 named constant
        if field.ndim == SPECTRAL_DIM:
            field = field.unsqueeze(0)

        B, N, M = field.shape
        assert N == M, "Field must be square"

        # Default grid spacing for high NA
        if dx is None:
            dx = self.wavelength / (4 * self.NA)

        # Step size
        dz = z_distance / self.n_steps

        logger.debug(
            f"M3 Wide-angle: z={z_distance*1e3:.1f} mm, steps={self.n_steps}, NA={self.NA}"
        )
        logger.debug(f"  Grid: N={N}, dx={dx*1e9:.1f} nm, dz={dz*1e6:.1f} μm")

        # Create wide-angle operators
        numerator, denominator = self._create_wide_angle_operator(N, dx, dz)
        numerator = numerator.unsqueeze(0).expand(B, -1, -1)
        denominator = denominator.unsqueeze(0).expand(B, -1, -1)

        # Initialize field
        current_field = field.clone()

        # Wide-angle propagation loop
        for step in range(self.n_steps):
            # Wide-angle propagation step
            current_field = self._solve_implicit_step(current_field, numerator, denominator)

            # Ensure FP32
            current_field = self._ensure_fp32(current_field)

            # Progress logging
            if (step + 1) % max(1, self.n_steps // 10) == 0:
                z = (step + 1) * dz
                logger.debug(f"  Step {step+1}/{self.n_steps} (z={z*1e6:.1f} μm)")

        # Restore original shape
        if len(original_shape) == SPECTRAL_DIM:
            current_field = current_field.squeeze(0)

        # Verify energy conservation
        if logger.isEnabledFor(logging.DEBUG):
            energy_in = torch.sum(torch.abs(field) ** 2).item()
            energy_out = torch.sum(torch.abs(current_field) ** 2).item()
            energy_ratio = energy_out / (energy_in + 1e-10)
            logger.debug(f"Energy conservation: {energy_ratio:.6f}")

            ENERGY_WARN_TOL = 0.02
            if abs(energy_ratio - 1.0) > ENERGY_WARN_TOL:
                logger.warning(
                    "M3: Energy deviation "
                    f"{abs(1-energy_ratio)*100:.2f}% (acceptable for wide-angle)"
                )

        return current_field

    def propagate_high_na_focus(
        self, field: torch.Tensor, focal_length: float, dx: float
    ) -> torch.Tensor:
        """
        Propagate to focus for high-NA lens (M3-compliant)

        Args:
            field: Input field (typically pupil field)
            focal_length: Focal length in meters
            dx: Grid spacing

        Returns:
            Field at focus
        """
        field = self._ensure_fp32(field)
        field = field.to(self.device)

        N = field.shape[-1]

        # Create coordinate grids in FP32
        x = torch.arange(N, dtype=torch.float32, device=self.device) - N // 2
        x = x * dx
        X, Y = torch.meshgrid(x, x, indexing="xy")

        # Apply lens phase (spherical)
        R_sq = X**2 + Y**2
        lens_phase = -self.k * R_sq / (2 * focal_length)

        # Apply lens
        field_after_lens = field * torch.exp(1j * lens_phase)
        field_after_lens = self._ensure_fp32(field_after_lens)

        # Propagate to focus
        return self.propagate(field_after_lens, focal_length, dx)

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
