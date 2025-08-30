"""Base protocol and utilities for optical field sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Protocol, runtime_checkable

try:  # optional at import time for CPU CI
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None  # type: ignore


@runtime_checkable
class Source(Protocol):
    """Protocol for field sources.
    
    All sources must implement:
    - prepare(): Initialize with configuration and device
    - emit(): Generate complex field for a sample index
    """
    
    def prepare(self, cfg: Dict, device: str = "cpu") -> None:
        """Prepare source with configuration.
        
        Args:
            cfg: Configuration dictionary
            device: Computation device ('cpu' or 'cuda')
        """
        ...
    
    def emit(self, sample_idx: int = 0):  # type: ignore[no-untyped-def]
        """Emit complex field for given sample.
        
        Args:
            sample_idx: Sample index for spectral/angular sampling
            
        Returns:
            Complex field tensor of shape (ny, nx)
        """
        ...


class BaseSource(ABC):
    """Abstract base class for sources with common functionality."""
    
    def __init__(self):
        self.device = 'cpu'
        self._prepared = False
        self._grid_shape = None
        self._pitch_um = None
    
    @abstractmethod
    def prepare(self, cfg: Dict, device: str = "cpu") -> None:
        """Prepare source with configuration."""
        self.device = device
        self._prepared = True
    
    @abstractmethod  
    def emit(self, sample_idx: int = 0):  # type: ignore[no-untyped-def]
        """Emit complex field."""
        if not self._prepared:
            raise RuntimeError("Source not prepared. Call prepare() first.")
    
    def set_grid(self, ny: int, nx: int, pitch_um: float):
        """Set grid parameters for field generation.
        
        Args:
            ny: Number of grid points in Y
            nx: Number of grid points in X
            pitch_um: Grid spacing in micrometers
        """
        self._grid_shape = (ny, nx)
        self._pitch_um = pitch_um
    
    def get_grid(self) -> tuple:
        """Get current grid parameters.
        
        Returns:
            Tuple of (ny, nx, pitch_um) or (None, None, None) if not set
        """
        if self._grid_shape is None:
            return (None, None, None)
        return (*self._grid_shape, self._pitch_um)
