
"""Tests for the environment preflight checker."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.check_environment import collect_checks


class EnvironmentPreflightTests(unittest.TestCase):

    def test_supported_environment(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            for filename in (
                "requirements.txt",
                ".env.example",
            ):
                (root / filename).write_text(
                    "example",
                    encoding="utf-8",
                )

            results = collect_checks((3, 13), root)

            self.assertTrue(
                all(ok for ok, _ in results)
            )

    def test_wrong_python_version(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = collect_checks(
                (3, 12),
                Path(tmp),
            )

            self.assertFalse(results[0][0])

    def test_missing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = collect_checks(
                (3, 13),
                Path(tmp),
            )

            self.assertFalse(results[1][0])
            self.assertFalse(results[2][0])


if __name__ == "__main__":
    unittest.main()
