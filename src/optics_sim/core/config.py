"""Configuration loading and validation for optics simulation.

This module handles YAML configuration files, normalizes units to micrometers,
validates parameters, and captures deterministic seeds.
"""

from __future__ import annotations

import copy
import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

try:  # optional at import time for CPU CI
    import torch  # type: ignore
except Exception:  # pragma: no cover - optional dependency at import time
    torch = None  # type: ignore

import yaml


def load(path: str | Path) -> dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        path: Path to YAML configuration file

    Returns:
        Raw configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path) as f:
        try:
            cfg = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Failed to parse YAML config: {e}")

    if cfg is None:
        cfg = {}

    return cfg


def validate(cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize configuration to standard units.

    Converts all units to micrometers, validates ranges, captures RNG seeds,
    and ensures all required fields are present.

    Args:
        cfg: Raw configuration dictionary

    Returns:
        Normalized Cfg dictionary with validated parameters

    Raises:
        ValueError: If required fields are missing or values are out of range
    """
    cfg = copy.deepcopy(cfg)

    # Convert wavelength from nm to µm if present
    if "lambda_nm" in cfg:
        if isinstance(cfg["lambda_nm"], dict):
            if "min" in cfg["lambda_nm"]:
                cfg["lambda_um"] = {"min": cfg["lambda_nm"]["min"] / 1000.0}
            if "max" in cfg["lambda_nm"]:
                if "lambda_um" not in cfg:
                    cfg["lambda_um"] = {}
                cfg["lambda_um"]["max"] = cfg["lambda_nm"]["max"] / 1000.0
        elif isinstance(cfg["lambda_nm"], (int, float)):
            cfg["lambda_um"] = cfg["lambda_nm"] / 1000.0

    # Handle legacy wavelength fields in sources
    if "sources" in cfg:
        for source in cfg["sources"]:
            if "center_nm" in source:
                source["center_um"] = source["center_nm"] / 1000.0
            if "bandwidth_nm" in source:
                source["bandwidth_um"] = source["bandwidth_nm"] / 1000.0

    # Ensure required fields with defaults
    required_defaults = {
        "NA_max": None,  # Must be provided
        "grid": {"target_px": 1024},
        "recorders": [],
        "components": [],
        "sources": [],
        "runtime": {"budget": {"vram_gb": 10.0, "time_s": 3600.0}},
    }

    # Check and apply required fields
    for key, default in required_defaults.items():
        if key not in cfg:
            if default is None:
                # Try to infer from other fields
                if key == "NA_max":
                    # Check if na is present (legacy field)
                    if "na" in cfg:
                        cfg["NA_max"] = cfg["na"]
                    else:
                        raise ValueError(f"Required field '{key}' is missing")
            else:
                cfg[key] = copy.deepcopy(default)
        elif key == "runtime" and isinstance(default, dict):
            # Merge runtime defaults
            if "budget" not in cfg["runtime"]:
                cfg["runtime"]["budget"] = copy.deepcopy(default["budget"])
            else:
                for bkey, bval in default["budget"].items():
                    if bkey not in cfg["runtime"]["budget"]:
                        cfg["runtime"]["budget"][bkey] = bval

    # Validate NA_max range
    if not 0.01 <= cfg["NA_max"] <= 1.4:
        raise ValueError(f"NA_max must be between 0.01 and 1.4, got {cfg['NA_max']}")

    # Validate and normalize grid
    if "grid" in cfg:
        grid = cfg["grid"]

        # Handle legacy nx, ny fields
        if "nx" in grid or "ny" in grid:
            nx = grid.get("nx", grid.get("target_px", 1024))
            ny = grid.get("ny", grid.get("target_px", 1024))
            grid["target_px"] = max(nx, ny)

        if "target_px" not in grid:
            grid["target_px"] = 1024

        # Ensure grid size is reasonable
        if not 64 <= grid["target_px"] <= 16384:
            raise ValueError(
                f"grid.target_px must be between 64 and 16384, got {grid['target_px']}"
            )

        # Convert pitch to micrometers if needed
        if "pitch_nm" in grid:
            grid["pitch_um"] = grid["pitch_nm"] / 1000.0
        elif "pitch_um" not in grid:
            # Calculate default pitch based on wavelength and NA
            if "lambda_um" in cfg:
                if isinstance(cfg["lambda_um"], dict):
                    lambda_min = cfg["lambda_um"].get("min", 0.4)
                else:
                    lambda_min = cfg["lambda_um"]
            else:
                lambda_min = 0.5  # Default visible light

            # Nyquist sampling criterion
            grid["pitch_um"] = lambda_min / (2.0 * cfg["NA_max"])

    # Capture deterministic seeds
    if "seeds" not in cfg:
        cfg["seeds"] = {}

    if "seed_tensor" not in cfg["seeds"]:
        cfg["seeds"]["seed_tensor"] = 42

    if "seed_sampler" not in cfg["seeds"]:
        cfg["seeds"]["seed_sampler"] = 1337

    # Capture environment snapshot for reproducibility
    if torch is not None:
        cfg["environment"] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_available": bool(getattr(torch.cuda, "is_available", lambda: False)()),
            "cuda_version": getattr(getattr(torch, "version", None), "cuda", None)
            if bool(getattr(torch.cuda, "is_available", lambda: False)())
            else None,
            "device_count": getattr(torch.cuda, "device_count", lambda: 0)()
            if bool(getattr(torch.cuda, "is_available", lambda: False)())
            else 0,
        }
    else:
        cfg["environment"] = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device_count": 0,
        }

    # Add config hash for tracking
    cfg_str = str(sorted(cfg.items()))
    cfg["config_hash"] = hashlib.sha256(cfg_str.encode()).hexdigest()[:8]

    # Handle wavelength ranges
    if "lambda_um" not in cfg:
        # Default visible range
        cfg["lambda_um"] = {"min": 0.4, "max": 0.7}
    elif not isinstance(cfg["lambda_um"], dict):
        # Single wavelength
        val = cfg["lambda_um"]
        cfg["lambda_um"] = {"min": val, "max": val}

    # Validate wavelength range
    lambda_min = cfg["lambda_um"].get("min", 0.4)
    lambda_max = cfg["lambda_um"].get("max", 0.7)

    if not 0.1 <= lambda_min <= 10.0:
        raise ValueError(f"lambda_min must be between 0.1 and 10.0 µm, got {lambda_min}")
    if not 0.1 <= lambda_max <= 10.0:
        raise ValueError(f"lambda_max must be between 0.1 and 10.0 µm, got {lambda_max}")
    if lambda_min > lambda_max:
        raise ValueError(f"lambda_min ({lambda_min}) must be <= lambda_max ({lambda_max})")

    # Validate memory budget
    vram_gb = cfg["runtime"]["budget"]["vram_gb"]
    if not 1.0 <= vram_gb <= 80.0:
        raise ValueError(f"vram_gb must be between 1.0 and 80.0, got {vram_gb}")

    time_s = cfg["runtime"]["budget"]["time_s"]
    if not 1.0 <= time_s <= 86400.0:  # 1 second to 24 hours
        raise ValueError(f"time_s must be between 1.0 and 86400.0, got {time_s}")

    return cfg
