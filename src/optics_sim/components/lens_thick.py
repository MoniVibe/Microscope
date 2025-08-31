from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ThickLens:
    """Placeholder for a thick lens model. No behavior in scaffold."""

    thickness_um: float = 1.0
