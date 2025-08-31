"""Plane-based field recorders for capturing simulation data.

Records intensity, phase, and complex field data at specified z-planes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class PlaneRecorder:
    """Records field data at specific z-planes during propagation.

    Attributes:
        planes: List of recording plane configurations
        recorded_data: Dictionary storing captured data
    """

    planes: list[dict] = field(default_factory=list)
    recorded_data: dict = field(default_factory=dict)

    def add_plane(self, z_um: float, what: set[str]) -> None:
        """Add a recording plane.

        Args:
            z_um: Z-position in micrometers
            what: Set of quantities to record {'intensity', 'complex', 'phase'}
        """
        # Validate recording types
        allowed = {"intensity", "complex", "phase", "amplitude"}
        invalid = what - allowed
        if invalid:
            raise ValueError(f"Invalid recording types: {invalid}. Allowed: {allowed}")

        # Add plane configuration
        self.planes.append({"z_um": z_um, "what": what, "index": len(self.planes)})

        # Initialize storage
        plane_key = f"plane_{len(self.planes) - 1:03d}_z{z_um:.3f}"
        self.recorded_data[plane_key] = {}

    def capture(self, state: dict) -> dict[str, torch.Tensor]:
        """Capture field data based on current propagation state.

        Args:
            state: Propagation state containing:
                - 'field': Current field tensor (complex)
                - 'z_um': Current z-position
                - 'wavelength_um': Current wavelength (optional)
                - 'spectral_idx': Spectral sample index (optional)

        Returns:
            Dictionary of captured data for this z-position
        """
        current_z = state.get("z_um", 0.0)
        field_data = state.get("field")

        if field_data is None:
            raise ValueError("No field data in state")

        # Find matching plane(s) within tolerance
        captured = {}
        z_tolerance = 1e-6  # 1 nm tolerance

        for plane_config in self.planes:
            if abs(plane_config["z_um"] - current_z) < z_tolerance:
                # This plane should be recorded
                plane_key = f"plane_{plane_config['index']:03d}_z{plane_config['z_um']:.3f}"

                # Record requested quantities
                for quantity in plane_config["what"]:
                    captured_data = self._extract_quantity(field_data, quantity)

                    # Store with optional spectral index
                    spectral_idx = state.get("spectral_idx", None)
                    if spectral_idx is not None:
                        key = f"{quantity}_s{spectral_idx:02d}"
                    else:
                        key = quantity

                    if plane_key not in self.recorded_data:
                        self.recorded_data[plane_key] = {}

                    self.recorded_data[plane_key][key] = captured_data
                    captured[f"{plane_key}_{key}"] = captured_data

        return captured

    def _extract_quantity(self, field: torch.Tensor, quantity: str) -> torch.Tensor:
        """Extract specific quantity from complex field.

        Args:
            field: Complex field tensor
            quantity: Type of data to extract

        Returns:
            Extracted data tensor
        """
        if quantity == "complex":
            return field.clone()
        elif quantity == "intensity":
            return torch.abs(field) ** 2
        elif quantity == "phase":
            return torch.angle(field)
        elif quantity == "amplitude":
            return torch.abs(field)
        else:
            raise ValueError(f"Unknown quantity: {quantity}")

    def get_plane_data(self, z_um: float = None, index: int = None) -> dict:
        """Retrieve recorded data for a specific plane.

        Args:
            z_um: Z-position of plane (optional)
            index: Plane index (optional)

        Returns:
            Dictionary of recorded data for the plane
        """
        if z_um is not None:
            # Find by z-position
            for key in self.recorded_data:
                if f"z{z_um:.3f}" in key:
                    return self.recorded_data[key]
        elif index is not None:
            # Find by index
            key = f"plane_{index:03d}_z"
            for full_key in self.recorded_data:
                if key in full_key:
                    return self.recorded_data[full_key]

        return {}

    def get_all_data(self) -> dict:
        """Get all recorded data.

        Returns:
            Complete dictionary of recorded data
        """
        return self.recorded_data

    def clear(self) -> None:
        """Clear all recorded data."""
        self.recorded_data.clear()

    def summary(self) -> dict:
        """Get summary of recording configuration and data.

        Returns:
            Summary dictionary
        """
        summary = {
            "num_planes": len(self.planes),
            "z_positions": [p["z_um"] for p in self.planes],
            "quantities": list(set().union(*[p["what"] for p in self.planes])),
            "data_keys": list(self.recorded_data.keys()),
        }

        # Add data shapes
        shapes = {}
        for plane_key, plane_data in self.recorded_data.items():
            shapes[plane_key] = {
                k: v.shape if isinstance(v, torch.Tensor) else None for k, v in plane_data.items()
            }
        summary["data_shapes"] = shapes

        return summary


class MultiPlaneRecorder:
    """Advanced recorder supporting multiple configurations and spectral samples."""

    def __init__(self):
        self.recorders = {}
        self.metadata = {}

    def add_configuration(self, name: str, planes: list[dict]) -> None:
        """Add a named recording configuration.

        Args:
            name: Configuration name
            planes: List of plane definitions
        """
        recorder = PlaneRecorder()
        for plane_def in planes:
            recorder.add_plane(plane_def["z_um"], set(plane_def["what"]))
        self.recorders[name] = recorder

    def capture_all(self, state: dict) -> dict:
        """Capture data for all configurations.

        Args:
            state: Current propagation state

        Returns:
            Combined captured data
        """
        all_captured = {}
        for name, recorder in self.recorders.items():
            captured = recorder.capture(state)
            for key, value in captured.items():
                all_captured[f"{name}_{key}"] = value
        return all_captured

    def get_recorder(self, name: str) -> PlaneRecorder:
        """Get a specific recorder by name."""
        return self.recorders.get(name)

    def set_metadata(self, key: str, value) -> None:
        """Set metadata for recordings."""
        self.metadata[key] = value

    def export_data(self) -> dict:
        """Export all data with metadata.

        Returns:
            Complete data dictionary including metadata
        """
        export = {"metadata": self.metadata, "configurations": {}}

        for name, recorder in self.recorders.items():
            export["configurations"][name] = {
                "planes": recorder.planes,
                "data": recorder.get_all_data(),
            }

        return export


def create_recorder_from_config(config: dict) -> PlaneRecorder:
    """Create recorder from configuration dictionary.

    Args:
        config: Configuration with 'recorders' key

    Returns:
        Configured PlaneRecorder
    """
    recorder = PlaneRecorder()

    recorders_config = config.get("recorders", [])
    for rec_def in recorders_config:
        z_um = rec_def.get("z_um", 0.0)
        what = rec_def.get("what", ["intensity"])

        # Ensure 'what' is a set
        if isinstance(what, str):
            what = {what}
        elif isinstance(what, list):
            what = set(what)

        recorder.add_plane(z_um, what)

    return recorder


def interpolate_to_plane(
    field: torch.Tensor, current_z: float, target_z: float, propagator
) -> torch.Tensor:
    """Interpolate field to recording plane if needed.

    Args:
        field: Current field
        current_z: Current z-position
        target_z: Target recording z-position
        propagator: Function to propagate field

    Returns:
        Field at target z
    """
    dz = target_z - current_z

    if abs(dz) < 1e-6:  # Already at target
        return field

    # Use propagator to reach exact plane
    return propagator(field, dz)
