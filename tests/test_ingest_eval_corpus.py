"""评测语料导入脚本的文件布局解析测试。"""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location("ingest_eval_corpus", PROJECT_ROOT / "scripts" / "ingest_eval_corpus.py")
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
resolve_manifest_pdf = _MODULE.resolve_manifest_pdf


class IngestEvalCorpusTests(unittest.TestCase):
    def test_resolves_split_subdirectory_when_manifest_only_has_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = root / "dev" / "paper.pdf"
            expected.parent.mkdir()
            expected.write_bytes(b"pdf")
            self.assertEqual(resolve_manifest_pdf(raw_dir=root, local_filename="paper.pdf"), expected.resolve())

    def test_duplicate_nested_filename_is_rejected_instead_of_guessed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for split in ("dev", "test"):
                path = root / split / "paper.pdf"
                path.parent.mkdir()
                path.write_bytes(split.encode())
            with self.assertRaisesRegex(Exception, "多个同名 PDF"):
                resolve_manifest_pdf(raw_dir=root, local_filename="paper.pdf")


if __name__ == "__main__":
    unittest.main()
