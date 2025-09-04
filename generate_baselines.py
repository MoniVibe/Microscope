import subprocess
import sys
from pathlib import Path

# Get to the project root
project_root = Path(r"C:\Users\Moni\Documents\claudeprojects\Microscope")

# Run ruff check
result = subprocess.run(
    [sys.executable, "-m", "ruff", "check", "--output-format", "concise"],
    capture_output=True,
    text=True,
    cwd=str(project_root),
    check=False,
)

print("Ruff output:")
print(result.stdout if result.stdout else "No violations")
print("\nStderr:", result.stderr if result.stderr else "None")
print(f"Return code: {result.returncode}")

# Generate baseline if there are violations
if result.returncode != 0 and result.stdout:
    print("\nGenerating ruff baseline...")
    baseline_result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", "--output-format", "json"],
        capture_output=True,
        text=True,
        cwd=str(project_root),
        check=False,
    )

    # Write baseline
    import json

    if baseline_result.stdout:
        violations = json.loads(baseline_result.stdout)
        # Format as TOML for ruff-baseline.toml
        with open(project_root / "ruff-baseline.toml", "w") as f:
            f.write("# Ruff baseline - existing violations to ignore\n")
            f.write("# Generated for zero-new policy\n\n")
            f.write("[tool.ruff.lint.per-file-ignores]\n")

            file_violations = {}
            for v in violations:
                filepath = v.get("filename", "")
                code = v.get("code", "")
                if filepath and code:
                    if filepath not in file_violations:
                        file_violations[filepath] = []
                    if code not in file_violations[filepath]:
                        file_violations[filepath].append(code)

            for filepath, codes in file_violations.items():
                f.write(f'"{filepath}" = {codes}\n')
