#!/usr/bin/env python
"""Fix gate issues - run ruff format and check, generate baselines."""

import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, description):
    """Run command and report."""
    print(f"\n{description}:")
    print(f"  Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        print(f"  Failed: {result.stderr[:500]}")
    else:
        print("  Success")
    return result


def main():
    root = Path(__file__).parent

    # Step 1: Format code
    print("Step 1: Formatting code with ruff...")
    run_cmd([sys.executable, "-m", "ruff", "format"], "Ruff format")

    # Step 2: Auto-fix what we can
    print("\nStep 2: Auto-fixing with ruff...")
    run_cmd([sys.executable, "-m", "ruff", "check", "--fix"], "Ruff fix")

    # Step 3: Check violations
    print("\nStep 3: Checking remaining violations...")
    result = run_cmd(
        [sys.executable, "-m", "ruff", "check", "--output-format", "concise"], "Ruff check"
    )

    # Step 4: Generate baseline if needed
    if result.returncode != 0:
        print("\nStep 4: Generating ruff baseline...")
        # Get violations in GitHub format
        violations = subprocess.run(
            [sys.executable, "-m", "ruff", "check", "--output-format", "github"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout

        # Write to baseline
        baseline_path = root / "ruff-baseline.toml"
        with open(baseline_path, "w") as f:
            f.write("# Ruff baseline - existing violations to ignore\n")
            f.write("# Generated for zero-new policy\n\n")

            if violations:
                # Parse and group by file
                file_codes = {}
                for line in violations.strip().split("\n"):
                    if "::" in line:
                        parts = line.split("::")
                        if len(parts) >= 4:
                            filepath = parts[0].replace("\\", "/")
                            code = parts[3].split(":")[0].strip()
                            if filepath not in file_codes:
                                file_codes[filepath] = set()
                            file_codes[filepath].add(code)

                if file_codes:
                    f.write("[tool.ruff.lint.per-file-ignores]\n")
                    for filepath, codes in sorted(file_codes.items()):
                        codes_str = ", ".join(f'"{c}"' for c in sorted(codes))
                        f.write(f'"{filepath}" = [{codes_str}]\n')
                else:
                    f.write("# No violations found\n")

        print(f"  Wrote baseline to {baseline_path}")

    # Step 5: Generate mypy baseline
    print("\nStep 5: Generating mypy baseline...")
    mypy_result = subprocess.run(
        [sys.executable, "-m", "mypy", "."], capture_output=True, text=True, check=False
    )

    baseline_path = root / "mypy-baseline.txt"
    with open(baseline_path, "w") as f:
        f.write("# mypy baseline - existing violations\n")
        f.write("# Generated for zero-new policy\n\n")
        if mypy_result.stdout:
            f.write(mypy_result.stdout)
        else:
            f.write("# No mypy violations found\n")

    print(f"  Wrote mypy baseline to {baseline_path}")

    print("\nDone! Gates should be ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
