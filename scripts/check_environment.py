
"""Check ResearchPilot-RAG setup requirements."""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_PYTHON = (3, 13)
REQUIRED_FILES = ("requirements.txt", ".env.example")


def collect_checks(
    version: tuple[int, int],
    project_root: Path,
) -> list[tuple[bool, str]]:
    """Check Python version and required project files."""

    version_ok = version == REQUIRED_PYTHON

    results = [
        (
            version_ok,
            f"Python {version[0]}.{version[1]} detected; "
            "documented version: 3.13",
        )
    ]

    for filename in REQUIRED_FILES:
        exists = (project_root / filename).is_file()
        results.append((exists, f"{filename} exists"))

    return results


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]

    version = (
        sys.version_info.major,
        sys.version_info.minor,
    )

    results = collect_checks(version, project_root)

    print("ResearchPilot-RAG environment preflight")

    for passed, message in results:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {message}")

    if not all(passed for passed, _ in results):
        print("Please fix failed checks before installation.")
        return 1

    print("Basic prerequisites look good.")
    print("Model downloads and API access are not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
