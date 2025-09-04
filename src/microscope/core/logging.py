"""Structured logging for microscope simulation.

Provides JSON lines logging for reproducibility and analysis.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Format log records as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as JSON."""
        log_data = {
            "timestamp": time.time(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }

        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_data.update(record.extra_data)

        return json.dumps(log_data, default=str)


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    """Setup structured logging.

    Args:
        log_path: Optional path for JSON lines log file
        level: Logging level
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with simple format
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler with JSON format if path provided
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> "StructuredLogger":
    """Get a structured logger instance.

    Args:
        name: Logger name (usually __name__)

    Returns:
        StructuredLogger instance
    """
    return StructuredLogger(logging.getLogger(name))


class StructuredLogger:
    """Wrapper for adding structured data to log messages."""

    def __init__(self, logger: logging.Logger):
        """Initialize with a standard logger."""
        self.logger = logger

    def _log(self, level: int, msg: str, data: dict[str, Any] | None = None) -> None:
        """Log with structured data."""
        extra = {"extra_data": data} if data else {}
        self.logger.log(level, msg, extra=extra)

    def debug(self, msg: str, data: dict[str, Any] | None = None) -> None:
        """Debug level log."""
        self._log(logging.DEBUG, msg, data)

    def info(self, msg: str, data: dict[str, Any] | None = None) -> None:
        """Info level log."""
        self._log(logging.INFO, msg, data)

    def warning(self, msg: str, data: dict[str, Any] | None = None) -> None:
        """Warning level log."""
        self._log(logging.WARNING, msg, data)

    def error(self, msg: str, data: dict[str, Any] | None = None) -> None:
        """Error level log."""
        self._log(logging.ERROR, msg, data)


__all__ = [
    "setup_logging",
    "get_logger",
    "StructuredLogger",
]
