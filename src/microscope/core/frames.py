"""Coordinate frames and transformations for microscope simulation.

Right-handed coordinate system with Z along optical axis.
"""

from .types import Position3D, Rotation3D, Transform


def compose_transform(
    rotation: Rotation3D = (0.0, 0.0, 0.0),
    translation: Position3D = (0.0, 0.0, 0.0),
    order: str = "ZYX",
) -> Transform:
    """Compose a transformation from Euler angles and translation.

    Args:
        rotation: Euler angles (rx, ry, rz) in degrees
        translation: Translation (x, y, z) in micrometers
        order: Rotation order (only ZYX supported)

    Returns:
        Transform dictionary
    """
    if order != "ZYX":
        raise ValueError(f"Unsupported rotation order: {order}")

    return {
        "rotation": rotation,
        "translation": translation,
        "order": order,
    }


def to_world(point_local: Position3D, transform: Transform) -> Position3D:
    """Transform a point from local to world coordinates.

    Args:
        point_local: Point in local coordinates
        transform: Transform from local to world

    Returns:
        Point in world coordinates
    """
    # Simplified implementation - just translation for now
    # TODO: Add rotation when needed
    tx, ty, tz = transform["translation"]
    px, py, pz = point_local

    return (px + tx, py + ty, pz + tz)


def from_world(point_world: Position3D, transform: Transform) -> Position3D:
    """Transform a point from world to local coordinates.

    Args:
        point_world: Point in world coordinates
        transform: Transform from local to world

    Returns:
        Point in local coordinates
    """
    # Simplified implementation - just translation for now
    # TODO: Add rotation when needed
    tx, ty, tz = transform["translation"]
    px, py, pz = point_world

    return (px - tx, py - ty, pz - tz)


__all__ = [
    "compose_transform",
    "to_world",
    "from_world",
]
