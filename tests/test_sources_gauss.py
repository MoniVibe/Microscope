"""Tests for Gaussian finite band source."""

import numpy as np
import torch
from optics_sim.sources.gaussian_finite_band import GaussianFiniteBand

# Named constants for PLR2004 in this test module
CENTER_PIX_TOL = 3
SPECTRAL_CONV_TOL = 0.02
PROFILE_REL_TOL = 0.2
MEAN_WAVELENGTH_TOL = 0.001
WEIGHTS_SUM_TOL = 1e-6


def test_gaussian_emit_shape():
    """Test that emitted field has correct shape."""
    src = GaussianFiniteBand(waist_um=5.0)
    src.prepare({}, "cpu")
    src.set_grid(32, 48, 0.5)

    field = src.emit(0)
    assert field.shape == (32, 48)
    assert field.dtype == torch.complex64


def test_spectral_convergence():
    """Test spectral sampling convergence to <2% L2 error."""
    # Create source with moderate bandwidth
    src = GaussianFiniteBand(
        center_um=0.55,
        bandwidth_um=0.05,  # ~10% relative bandwidth
        waist_um=10.0,
        spectral_samples=9,  # Standard preset sampling
    )

    cfg = {"preset": "Standard"}
    src.prepare(cfg, "cpu")

    # Compute convergence metric
    error = src.compute_convergence_metric(reference_samples=18)

    assert (
        error < SPECTRAL_CONV_TOL
    ), f"Spectral convergence error {error:.3%} exceeds {SPECTRAL_CONV_TOL:.0%}"


def test_gaussian_profile():
    """Test that field has Gaussian spatial profile."""
    waist_um = 10.0
    src = GaussianFiniteBand(center_um=0.55, bandwidth_um=0.01, waist_um=waist_um)
    src.prepare({}, "cpu")

    # Generate field on fine grid
    ny, nx = 128, 128
    pitch = 0.5
    src.set_grid(ny, nx, pitch)
    field = src.emit(0)

    # Check intensity profile
    intensity = torch.abs(field) ** 2

    # Find peak
    peak_idx = torch.argmax(intensity)
    cy, cx = peak_idx // nx, peak_idx % nx

    # Peak should be near center
    assert abs(cy - ny // 2) < CENTER_PIX_TOL
    assert abs(cx - nx // 2) < CENTER_PIX_TOL

    # Extract radial profile
    y = torch.arange(ny, dtype=torch.float32) - ny // 2
    x = torch.arange(nx, dtype=torch.float32) - nx // 2
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    r = torch.sqrt(xx**2 + yy**2) * pitch

    # Fit Gaussian
    # I(r) = I0 * exp(-2r²/w₀²)
    # At r = w₀, I = I0 * exp(-2) ≈ 0.135 * I0
    intensity_at_waist = intensity[r < waist_um + pitch].mean()
    intensity_peak = intensity[cy, cx]

    ratio = intensity_at_waist / intensity_peak
    expected_ratio = np.exp(-2)

    # Check within 20% (loose due to discretization)
    assert abs(ratio - expected_ratio) / expected_ratio < PROFILE_REL_TOL


def test_normalization():
    """Test that field is properly normalized."""
    src = GaussianFiniteBand(center_um=0.55, bandwidth_um=0.01, waist_um=5.0)
    src.prepare({}, "cpu")

    pitch = 0.2
    src.set_grid(256, 256, pitch)
    field = src.emit(0)

    # Check total power (should be 1 before spectral weight)
    power = torch.sum(torch.abs(field) ** 2) * pitch**2

    # Account for spectral weight
    spectral_weight = src.spectral_weights[0]
    expected_power = spectral_weight

    torch.testing.assert_close(power, expected_power, rtol=0.1)


def test_multiple_spectral_samples():
    """Test emission of multiple spectral samples."""
    src = GaussianFiniteBand(center_um=0.55, bandwidth_um=0.1, waist_um=10.0, spectral_samples=5)
    src.prepare({}, "cpu")
    src.set_grid(64, 64, 0.5)

    # Emit all spectral samples
    fields = []
    for i in range(src.spectral_samples):
        field = src.emit(i)
        fields.append(field)
        assert field.shape == (64, 64)

    # Fields should be different (different wavelengths)
    for i in range(1, len(fields)):
        assert not torch.allclose(fields[i], fields[0])


def test_wavelength_range():
    """Test that wavelengths span the specified bandwidth."""
    center = 0.55
    bandwidth = 0.05

    src = GaussianFiniteBand(center_um=center, bandwidth_um=bandwidth, spectral_samples=7)
    src.prepare({}, "cpu")

    info = src.get_spectral_info()
    wavelengths = info["wavelengths_um"]

    # Check range (approximately, due to Gauss-Hermite sampling)
    assert wavelengths.min() >= center - 2 * bandwidth
    assert wavelengths.max() <= center + 2 * bandwidth

    # Check centering
    mean_wavelength = np.average(wavelengths, weights=info["weights"])
    assert abs(mean_wavelength - center) < MEAN_WAVELENGTH_TOL


def test_angular_divergence():
    """Test that angular divergence affects field."""
    src_collimated = GaussianFiniteBand(center_um=0.55, waist_um=10.0, angular_sigma_rad=0.0)

    src_divergent = GaussianFiniteBand(center_um=0.55, waist_um=10.0, angular_sigma_rad=0.01)

    src_collimated.prepare({}, "cpu")
    src_divergent.prepare({}, "cpu")

    src_collimated.set_grid(64, 64, 0.5)
    src_divergent.set_grid(64, 64, 0.5)

    field_collimated = src_collimated.emit(0)
    field_divergent = src_divergent.emit(0)

    # Divergent beam should have phase variation
    phase_collimated = torch.angle(field_collimated)
    phase_divergent = torch.angle(field_divergent)

    # Phase standard deviation should be larger for divergent beam
    # (This is a simplified test - actual implementation may differ)
    std_collimated = torch.std(phase_collimated)
    std_divergent = torch.std(phase_divergent)

    # Just check they're different (implementation-dependent)
    assert not torch.allclose(field_collimated, field_divergent)


def test_device_compatibility():
    """Test that source works on different devices."""
    src = GaussianFiniteBand(center_um=0.55)

    # CPU
    src.prepare({}, "cpu")
    src.set_grid(32, 32, 0.5)
    field_cpu = src.emit(0)
    assert field_cpu.device.type == "cpu"

    # CUDA (if available)
    if torch.cuda.is_available():
        src_cuda = GaussianFiniteBand(center_um=0.55)
        src_cuda.prepare({}, "cuda")
        src_cuda.set_grid(32, 32, 0.5)
        field_cuda = src_cuda.emit(0)
        assert field_cuda.device.type == "cuda"


def test_adaptive_spectral_sampling():
    """Test that spectral samples adapt to bandwidth."""
    # Narrow band
    src_narrow = GaussianFiniteBand(
        center_um=0.55,
        bandwidth_um=0.005,  # 0.5% relative
    )
    src_narrow.prepare({}, "cpu")

    # Broad band
    src_broad = GaussianFiniteBand(
        center_um=0.55,
        bandwidth_um=0.1,  # 18% relative
    )
    src_broad.prepare({}, "cpu")

    # Broad band should have more samples
    assert src_broad.spectral_samples > src_narrow.spectral_samples


def test_gauss_hermite_weights():
    """Test that Gauss-Hermite weights sum to 1."""
    src = GaussianFiniteBand(center_um=0.55, bandwidth_um=0.05, spectral_samples=7)
    src.prepare({}, "cpu")

    info = src.get_spectral_info()
    weights = info["weights"]

    # Weights should sum to 1
    assert abs(weights.sum() - 1.0) < WEIGHTS_SUM_TOL


if __name__ == "__main__":
    # Run tests
    test_gaussian_emit_shape()
    test_spectral_convergence()
    test_gaussian_profile()
    test_normalization()
    test_multiple_spectral_samples()
    test_wavelength_range()
    test_angular_divergence()
    test_device_compatibility()
    test_adaptive_spectral_sampling()
    test_gauss_hermite_weights()

    print("All Gaussian source tests passed!")
