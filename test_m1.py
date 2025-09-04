#!/usr/bin/env python
"""Quick test script to verify M1 milestone functionality."""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and report results."""
    print(f"\n{'='*60}")
    print(f"Testing: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=True, text=True, check=False)

    if result.returncode == 0:
        print("✓ SUCCESS")
        if result.stdout:
            print("Output:", result.stdout[:500])
    else:
        print(f"✗ FAILED (exit code: {result.returncode})")
        if result.stderr:
            print("Error:", result.stderr[:500])

    return result.returncode == 0


def main():
    """Run M1 acceptance tests."""
    print("M1 Vertical Slice - Acceptance Test Suite")
    print("=" * 60)

    all_passed = True

    # Test 1: CLI help works
    if not run_command([sys.executable, "-m", "microscope.cli", "--help"], "CLI help"):
        all_passed = False

    # Test 2: CLI subcommands help
    for subcmd in ["run", "validate", "inspect"]:
        if not run_command(
            [sys.executable, "-m", "microscope.cli", subcmd, "--help"], f"CLI {subcmd} help"
        ):
            all_passed = False

    # Test 3: Config loading
    example_config = Path("examples/minimal.yaml")
    if example_config.exists():
        print(f"\n{'='*60}")
        print("Testing: Config loading")
        print("-" * 60)
        try:
            from microscope.core.config import load_config

            scene = load_config(example_config)
            print(
                f"✓ Loaded config with {len(scene.sources)} sources, "
                f"{len(scene.components)} components, {len(scene.recorders)} recorders"
            )
        except Exception as e:
            print(f"✗ Failed to load config: {e}")
            all_passed = False

    # Test 4: Run pytest on our new tests
    if not run_command([sys.executable, "-m", "pytest", "tests/test_cli.py", "-v"], "CLI tests"):
        all_passed = False

    if not run_command(
        [sys.executable, "-m", "pytest", "tests/test_config_roundtrip.py", "-v"],
        "Config round-trip tests",
    ):
        all_passed = False

    # Summary
    print(f"\n{'='*60}")
    if all_passed:
        print("✓ All M1 acceptance tests PASSED")
    else:
        print("✗ Some tests FAILED - see details above")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
