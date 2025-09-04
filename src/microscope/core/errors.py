"""Custom exception types for microscope simulation."""


class MicroscopeError(Exception):
    """Base exception for all microscope errors."""

    pass


class ConfigError(MicroscopeError):
    """Configuration-related errors."""

    pass


class SamplingError(MicroscopeError):
    """Sampling and grid-related errors."""

    pass


class BackendError(MicroscopeError):
    """Backend and device-related errors."""

    pass


class PhysicsError(MicroscopeError):
    """Physics simulation errors."""

    pass


class IOError(MicroscopeError):
    """Input/output errors."""

    pass


__all__ = [
    "MicroscopeError",
    "ConfigError",
    "SamplingError",
    "BackendError",
    "PhysicsError",
    "IOError",
]
