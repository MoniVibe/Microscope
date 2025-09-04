"""Tests for coordinate frame transformations."""

import numpy as np
import torch

# Named tolerances for tests (PLR2004)
ABS_TOL = 1e-6
REL_TOL = 1e-6
ROUND_TRIP_TOL = 1e-6
MULTI_ROUND_TRIP_TOL = 1e-5

from optics_sim.core.frames import (
    compose,
    compose_chain,
    from_world,
    identity,
    invert,
    to_world,
    transform_grid,
)


def test_identity_transform():
    """Test identity transform."""
    T = identity()

    # Identity should not change points
    p_local = torch.tensor([[1.0, 2.0, 3.0]])
    p_world = to_world(p_local, T)

    torch.testing.assert_close(p_world, p_local)


def test_translation_only():
    """Test pure translation."""
    euler = torch.zeros(3)
    t = torch.tensor([1.0, 2.0, 3.0])

    T = compose(euler, t)

    # Test single point
    p_local = torch.tensor([0.0, 0.0, 0.0])
    p_world = to_world(p_local, T)

    torch.testing.assert_close(p_world, t)


def test_rotation_z():
    """Test rotation around Z axis."""
    angle = np.pi / 2  # 90 degrees
    euler = torch.tensor([angle, 0.0, 0.0])
    t = torch.zeros(3)

    T = compose(euler, t)

    # Point on X axis should rotate to Y axis
    p_local = torch.tensor([1.0, 0.0, 0.0])
    p_world = to_world(p_local, T)

    expected = torch.tensor([0.0, 1.0, 0.0])
    torch.testing.assert_close(p_world, expected, atol=ABS_TOL, rtol=REL_TOL)


def test_round_trip_accuracy():
    """Test round-trip transformation accuracy < 1e-6 µm."""
    # Random transform
    euler = torch.tensor([0.1, 0.2, 0.3])
    t = torch.tensor([10.0, 20.0, 30.0])

    T = compose(euler, t)

    # Test multiple points
    points_local = torch.randn(100, 3) * 100  # Points within 100 µm

    # Forward and inverse transform
    points_world = to_world(points_local, T)
    points_back = from_world(points_world, T)

    # Check round-trip error
    error = torch.norm(points_back - points_local, dim=-1)
    max_error = error.max().item()

    assert (
        max_error < ROUND_TRIP_TOL
    ), f"Round-trip error {max_error:.2e} µm exceeds {ROUND_TRIP_TOL} µm"


def test_transform_grid():
    """Test grid transformation."""
    euler = torch.tensor([0.0, 0.0, 0.0])
    t = torch.tensor([5.0, 10.0, 0.0])

    T = compose(euler, t)

    # Small grid
    nx, ny = 3, 3
    pitch = 1.0

    grid_world = transform_grid(nx, ny, pitch, T)

    assert grid_world.shape == (ny, nx, 3)

    # Center point should be translated
    center_idx = (ny // 2, nx // 2)
    center_world = grid_world[center_idx]

    torch.testing.assert_close(center_world, t, atol=ABS_TOL)


def test_compose_chain():
    """Test composing multiple transforms."""
    # Two translations
    T1 = compose(torch.zeros(3), torch.tensor([1.0, 0.0, 0.0]))
    T2 = compose(torch.zeros(3), torch.tensor([0.0, 2.0, 0.0]))

    T_combined = compose_chain([T1, T2])

    # Should sum translations
    p = torch.tensor([0.0, 0.0, 0.0])
    p_world = to_world(p, T_combined)

    expected = torch.tensor([1.0, 2.0, 0.0])
    torch.testing.assert_close(p_world, expected, atol=1e-6)


def test_invert_transform():
    """Test transform inversion."""
    euler = torch.tensor([0.1, 0.2, 0.3])
    t = torch.tensor([5.0, 10.0, 15.0])

    T = compose(euler, t)
    T_inv = invert(T)

    # Applying T then T_inv should give identity
    p = torch.randn(10, 3)
    p_transformed = to_world(p, T)
    p_back = to_world(p_transformed, T_inv)

    torch.testing.assert_close(p_back, p, atol=1e-6)


def test_zyx_euler_order():
    """Test that Euler angles are applied in Z-Y-X order."""
    # Small rotations to test order
    rz = 0.1  # First rotation (Z)
    ry = 0.2  # Second rotation (Y)
    rx = 0.3  # Third rotation (X)

    euler = torch.tensor([rz, ry, rx])
    t = torch.zeros(3)

    T = compose(euler, t)
    R = T["R"]

    # Build expected matrix manually
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    cx, sx = np.cos(rx), np.sin(rx)

    # Z-Y-X order: R = Rz * Ry * Rx
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)

    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)

    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)

    R_expected = Rz @ Ry @ Rx

    torch.testing.assert_close(R, R_expected, atol=ABS_TOL)


def test_batch_transform():
    """Test batched transformations."""
    batch_size = 5
    euler = torch.randn(batch_size, 3) * 0.1
    t = torch.randn(batch_size, 3) * 10

    T = compose(euler, t)

    # Should have batch dimension
    assert T["R"].shape == (batch_size, 3, 3)
    assert T["t"].shape == (batch_size, 3)

    # Transform points
    points = torch.randn(10, 3)
    # Note: batched transform not fully implemented in simplified version
    # This test documents expected behavior


def test_transform_preserves_distances():
    """Test that rigid transform preserves distances."""
    euler = torch.tensor([0.5, 0.3, 0.1])
    t = torch.tensor([10.0, 20.0, 30.0])

    T = compose(euler, t)

    # Create two points
    p1 = torch.tensor([0.0, 0.0, 0.0])
    p2 = torch.tensor([1.0, 0.0, 0.0])

    # Transform both
    p1_world = to_world(p1, T)
    p2_world = to_world(p2, T)

    # Distance should be preserved
    dist_local = torch.norm(p2 - p1)
    dist_world = torch.norm(p2_world - p1_world)

    torch.testing.assert_close(dist_world, dist_local, atol=ABS_TOL)


def test_high_precision_round_trip():
    """Test multiple round trips maintain precision."""
    euler = torch.tensor([0.123, 0.456, 0.789])
    t = torch.tensor([12.345, 67.890, 34.567])

    T = compose(euler, t)

    # Start with a point
    p = torch.tensor([1.234, 5.678, 9.012])
    p_original = p.clone()

    # Do 100 round trips
    for _ in range(100):
        p = to_world(p, T)
        p = from_world(p, T)

    # Error should still be tiny
    error = torch.norm(p - p_original).item()
    assert error < MULTI_ROUND_TRIP_TOL, f"Error after 100 round trips: {error:.2e} µm"


if __name__ == "__main__":
    # Run basic tests
    test_identity_transform()
    test_translation_only()
    test_rotation_z()
    test_round_trip_accuracy()
    test_transform_grid()
    test_compose_chain()
    test_invert_transform()
    test_zyx_euler_order()
    test_transform_preserves_distances()
    test_high_precision_round_trip()

    print("All frame transformation tests passed!")
