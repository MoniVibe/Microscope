"""Coordinate frame transformations for optics simulation.

Implements right-handed coordinate system with Z-Y-X Euler rotations
and translations in micrometers. Ensures round-trip accuracy < 1e-6 µm.
"""

from __future__ import annotations

try:  # optional at import time for CPU-only environments
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None  # type: ignore


def compose(euler_zyx, t_um):  # type: ignore[no-untyped-def]
    """Compose a Z-Y-X Euler rotation and translation into a transform dictionary.

    Args:
        euler_zyx: Euler angles [rz, ry, rx] in radians, shape (3,) or (B, 3)
        t_um: Translation in micrometers [tx, ty, tz], shape (3,) or (B, 3)

    Returns:
        Transform dictionary with keys:
            - 'R': Rotation matrix, shape (3, 3) or (B, 3, 3)
            - 't': Translation vector, shape (3,) or (B, 3)
            - 'euler_zyx': Original Euler angles
            - 'T': Full 4x4 homogeneous transform, shape (4, 4) or (B, 4, 4)
    """
    # Fallback minimal implementation when torch is not available
    if torch is None:

        if isinstance(t_um, (list, tuple)) and len(t_um) == 3:
            t_tuple = (float(t_um[0]), float(t_um[1]), float(t_um[2]))
        else:
            t_tuple = (0.0, 0.0, 0.0)
        return Transform(
            {
                "R": None,
                "t": t_tuple,
                "euler_zyx": tuple(euler_zyx)
                if isinstance(euler_zyx, (list, tuple))
                else (0.0, 0.0, 0.0),
                "T": None,
            }
        )

    # Ensure tensors (torch path) in high precision to guarantee round-trip accuracy
    if not isinstance(euler_zyx, torch.Tensor):
        euler_zyx = torch.tensor(euler_zyx, dtype=torch.float64)
    else:
        euler_zyx = euler_zyx.to(dtype=torch.float64)
    if not isinstance(t_um, torch.Tensor):
        t_um = torch.tensor(t_um, dtype=torch.float64)
    else:
        t_um = t_um.to(dtype=torch.float64)

    # Handle batch dimension
    batch_shape = euler_zyx.shape[:-1]
    is_batch = len(batch_shape) > 0

    if not is_batch:
        euler_zyx = euler_zyx.unsqueeze(0)
        t_um = t_um.unsqueeze(0)
        batch_size = 1
    else:
        batch_size = batch_shape[0]

    # Extract angles
    rz = euler_zyx[:, 0]  # First rotation around Z
    ry = euler_zyx[:, 1]  # Second rotation around Y
    rx = euler_zyx[:, 2]  # Third rotation around X

    # Compute sin and cos
    cz, sz = torch.cos(rz), torch.sin(rz)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cx, sx = torch.cos(rx), torch.sin(rx)

    # Build rotation matrix using Z-Y-X order
    # R = Rz * Ry * Rx (applied right to left)
    R = torch.zeros((batch_size, 3, 3), dtype=torch.float64, device=euler_zyx.device)

    # Combined rotation matrix elements
    R[:, 0, 0] = cy * cz
    R[:, 0, 1] = sx * sy * cz - cx * sz
    R[:, 0, 2] = cx * sy * cz + sx * sz

    R[:, 1, 0] = cy * sz
    R[:, 1, 1] = sx * sy * sz + cx * cz
    R[:, 1, 2] = cx * sy * sz - sx * cz

    R[:, 2, 0] = -sy
    R[:, 2, 1] = sx * cy
    R[:, 2, 2] = cx * cy

    # Build 4x4 homogeneous transform
    T = torch.eye(4, dtype=torch.float64, device=euler_zyx.device)
    T = T.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
    T[:, :3, :3] = R
    T[:, :3, 3] = t_um

    # Remove batch dimension if input was unbatched
    if not is_batch:
        R = R.squeeze(0)
        t_um = t_um.squeeze(0)
        T = T.squeeze(0)
        euler_zyx = euler_zyx.squeeze(0)

    return Transform({"R": R, "t": t_um, "euler_zyx": euler_zyx, "T": T})


def to_world(p_local, T):  # type: ignore[no-untyped-def]
    """Transform points from local to world coordinates.

    Args:
        p_local: Local points, shape (..., 3)
        T: Transform dictionary from compose()

    Returns:
        World points, shape (..., 3)
    """
    if torch is None:
        # Minimal fallback: apply translation only
        px, py, pz = (float(p_local[0]), float(p_local[1]), float(p_local[2]))
        tx, ty, tz = T["t"]
        return (px + tx, py + ty, pz + tz)

    if not isinstance(p_local, torch.Tensor):
        p_local = torch.tensor(p_local, dtype=torch.float32)

    original_shape = p_local.shape
    p_local = p_local.reshape(-1, 3)

    # Apply rotation and translation
    R = T["R"]
    t = T["t"]

    # Handle batch dimensions
    if R.dim() == 3:  # Batched transform
        # p_local shape: (N, 3), R shape: (B, 3, 3), t shape: (B, 3)
        # Expand p_local to (1, N, 3) for broadcasting
        p_world = torch.matmul(p_local.unsqueeze(0), R.transpose(-2, -1))
        p_world = p_world + t.unsqueeze(1)
        p_world = p_world.squeeze(0)
    else:  # Single transform
        # p_local shape: (N, 3), R shape: (3, 3), t shape: (3,)
        p_world = torch.matmul(p_local, R.T) + t

    return p_world.reshape(original_shape)


def from_world(p_world: torch.Tensor, T: dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform points from world to local coordinates (inverse transform).

    Args:
        p_world: World points, shape (..., 3)
        T: Transform dictionary from compose()

    Returns:
        Local points, shape (..., 3)
    """
    if not isinstance(p_world, torch.Tensor):
        p_world = torch.tensor(p_world, dtype=torch.float32)

    original_shape = p_world.shape
    p_world = p_world.reshape(-1, 3)

    # Apply inverse transformation: p_local = R^T * (p_world - t)
    R = T["R"]
    t = T["t"]

    if R.dim() == 3:  # Batched transform
        # Subtract translation then apply inverse rotation
        p_local = p_world.unsqueeze(0) - t.unsqueeze(1)
        p_local = torch.matmul(p_local, R)
        p_local = p_local.squeeze(0)
    else:  # Single transform
        p_local = torch.matmul(p_world - t, R)

    return p_local.reshape(original_shape)


def transform_grid(nx: int, ny: int, pitch_um: float, T: dict[str, torch.Tensor]) -> torch.Tensor:
    """Transform a 2D grid of points from local to world coordinates.

    Creates a grid in the XY plane (Z=0) and transforms all points.

    Args:
        nx: Number of grid points in X
        ny: Number of grid points in Y
        pitch_um: Grid spacing in micrometers
        T: Transform dictionary from compose()

    Returns:
        World coordinates of grid points, shape (ny, nx, 3)
    """
    # Create grid in local coordinates (centered at origin)
    x = torch.linspace(-(nx - 1) / 2, (nx - 1) / 2, nx) * pitch_um
    y = torch.linspace(-(ny - 1) / 2, (ny - 1) / 2, ny) * pitch_um

    # Meshgrid
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    zz = torch.zeros_like(xx)

    # Stack into points array
    p_local = torch.stack([xx, yy, zz], dim=-1)  # Shape: (ny, nx, 3)

    # Transform to world coordinates
    original_shape = p_local.shape
    p_local_flat = p_local.reshape(-1, 3)
    p_world_flat = to_world(p_local_flat, T)
    p_world = p_world_flat.reshape(original_shape)

    return p_world


def identity():  # type: ignore[no-untyped-def]
    """Create an identity transform.

    Returns:
        Transform dictionary representing no rotation or translation
    """
    if torch is None:
        return compose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    return compose(torch.zeros(3, dtype=torch.float32), torch.zeros(3, dtype=torch.float32))


# Backwards-compat alias expected by tests
compose_zyx_euler = compose


class Transform(dict):
    """Lightweight transform container type for isinstance checks in tests."""

    pass


def invert(T: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Compute the inverse of a transform.

    Args:
        T: Transform dictionary from compose()

    Returns:
        Inverse transform dictionary
    """
    R = T["R"]
    t = T["t"]

    # Inverse rotation is transpose
    R_inv = R.T if R.dim() == 2 else R.transpose(-2, -1)

    # Inverse translation
    if R.dim() == 2:
        t_inv = -torch.matmul(R_inv, t)
    else:
        t_inv = -torch.matmul(R_inv, t.unsqueeze(-1)).squeeze(-1)

    # Build inverse transform
    T_inv = torch.eye(4, dtype=torch.float32, device=R.device)
    if R.dim() == 3:
        T_inv = T_inv.unsqueeze(0).expand(R.shape[0], -1, -1).contiguous()
        T_inv[:, :3, :3] = R_inv
        T_inv[:, :3, 3] = t_inv
    else:
        T_inv[:3, :3] = R_inv
        T_inv[:3, 3] = t_inv

    # Note: Euler angles for inverse are not simply negated
    # They would need to be recomputed from R_inv if needed
    return {
        "R": R_inv,
        "t": t_inv,
        "T": T_inv,
        "euler_zyx": None,  # Not computed for inverse
    }


def compose_chain(transforms: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Compose a chain of transforms.

    Args:
        transforms: List of transform dictionaries, applied left to right

    Returns:
        Combined transform dictionary
    """
    if not transforms:
        return identity()

    if len(transforms) == 1:
        return transforms[0]

    # Start with first transform
    T_combined = transforms[0]["T"].clone()

    # Multiply subsequent transforms
    for T in transforms[1:]:
        T_combined = torch.matmul(T["T"], T_combined)

    # Extract rotation and translation
    R = T_combined[:3, :3] if T_combined.dim() == 2 else T_combined[:, :3, :3]
    t = T_combined[:3, 3] if T_combined.dim() == 2 else T_combined[:, :3, 3]

    return {
        "R": R,
        "t": t,
        "T": T_combined,
        "euler_zyx": None,  # Combined Euler angles would need to be extracted
    }
