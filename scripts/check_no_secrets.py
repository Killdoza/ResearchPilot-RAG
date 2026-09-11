"""A small CI guard against accidentally committing common credential forms.

It is intentionally not a replacement for a server-side secret scanner. The
goal is to fail fast for the local project before a real `.env` or API key is
pushed to GitHub.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED_FILES = {".env.example"}
PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"^\s*(?:export\s+)?(?:DEEPSEEK|OPENAI)_API_KEY\s*=\s*"
        r"(?!replace_|\$\{|your_)[^\s#]{12,}",
    ),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
)


def repository_files() -> list[Path]:
    """Scan tracked files and non-ignored new files before the first commit."""
    output = subprocess.check_output(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    return [ROOT / item for item in output.decode("utf-8").split("\0") if item]


def main() -> int:
    findings: list[str] = []
    for path in repository_files():
        if path.name in IGNORED_FILES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(pattern.search(line) for pattern in PATTERNS):
                findings.append(f"{path.relative_to(ROOT)}:{line_number}")
    if findings:
        print("Potential secret detected in tracked files:", *findings, sep="\n  ")
        return 1
    print("Secret-pattern scan passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
