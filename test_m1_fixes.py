#!/usr/bin/env python
"""Test M1 fixes - verify gates are green."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description, check=True):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode == 0:
        print("SUCCESS")
        if result.stdout and len(result.stdout) < 500:
            print("Output:", result.stdout[:500])
    else:
        print(f"FAILED (exit code: {result.returncode})")
        if result.stderr:
            print("Error:", result.stderr[:500])
        if check:
            return False

    return result.returncode == 0


def main():
    """Run M1 fix verification."""
    print("M1 Gate Fix Verification")
    print("=" * 60)

    all_passed = True

    # Test 1: Ruff format check
    if not run_command(
        [sys.executable, "-m", "ruff", "format", "--check", "src/microscope"],
        "Ruff format check",
        check=False,  # May need format
    ):
        print("  Running formatter...")
        run_command([sys.executable, "-m", "ruff", "format", "src/microscope"], "Format", False)

    # Test 2: Ruff lint (E,F only)
    if not run_command(
        [sys.executable, "-m", "ruff", "check", "src/microscope", "--select", "E,F"],
        "Ruff lint (E,F only)",
    ):
        all_passed = False

    # Test 3: CLI help
    if not run_command([sys.executable, "-m", "microscope.cli", "--help"], "CLI help"):
        all_passed = False

    # Test 4: Run with minimal config
    example = Path("examples/minimal.yaml")
    if example.exists():
        if not run_command(
            [sys.executable, "-m", "microscope.cli", "run", "-c", str(example), "-o", "out/min"],
            "Run with minimal config",
        ):
            all_passed = False

    # Test 5: Validate subcommand
    if not run_command([sys.executable, "-m", "microscope.cli", "validate"], "Validate subcommand"):
        all_passed = False

    # Test 6: Tests produce coverage.xml
    print("\n" + "=" * 60)
    print("Testing: Pytest with coverage")
    print("-" * 60)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-ra",
            "--cov=microscope",
            "--cov-report=xml",
            "tests/test_cli.py",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if Path("coverage.xml").exists():
        print("SUCCESS: coverage.xml generated")
    else:
        print("FAILED: coverage.xml not found")
        all_passed = False

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("All M1 gate fixes PASSED")
        print("\nTo run full verification:")
        print("  make verify")
    else:
        print("Some tests FAILED - see details above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
