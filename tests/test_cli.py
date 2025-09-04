from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_cli_help() -> None:
    out = subprocess.check_output(["python", "-m", "microscope.cli", "--help"]).decode()
    assert "run" in out and "validate" in out and "inspect" in out


def test_cli_validate_and_run(tmp_path: Path) -> None:
    cfg = Path("examples/minimal.yaml")
    # validate
    subprocess.check_call(["python", "-m", "microscope.cli", "validate", "-c", str(cfg)])
    # run
    outdir = tmp_path / "out"
    subprocess.check_call(
        ["python", "-m", "microscope.cli", "run", "-c", str(cfg), "-o", str(outdir)]
    )
    j = json.loads((outdir / "config.normalized.json").read_text())
    assert j["NA_max"] == 0.5


def test_cli_run_help() -> None:
    result = subprocess.run(
        ["python", "-m", "microscope.cli", "run", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "--out" in result.stdout


def test_cli_validate_help() -> None:
    result = subprocess.run(
        ["python", "-m", "microscope.cli", "validate", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0


def test_cli_inspect_help() -> None:
    result = subprocess.run(
        ["python", "-m", "microscope.cli", "inspect", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "--config" in result.stdout


def test_cli_missing_command() -> None:
    result = subprocess.run(
        ["python", "-m", "microscope.cli"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
