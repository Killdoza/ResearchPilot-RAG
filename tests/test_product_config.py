"""产品 Profile 的加载、校验与 E5 对齐测试。"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.product_config import (  # noqa: E402
    ProductConfigurationError,
    load_product_profile,
    product_profile_to_dict,
)
from researchpilot.runtime_factory import build_service_from_components  # noqa: E402
from researchpilot.sparse import PersistentBM25Repository  # noqa: E402


class _NoopIndex:
    """Factory 测试只验证装配参数，不触发真实索引调用。"""

    def list_children(self, **kwargs: object) -> tuple[object, ...]:
        return ()


class _WhitespaceTokenizer:
    def tokenize(self, text: str) -> tuple[str, ...]:
        return tuple(text.split())


class ProductProfileTests(unittest.TestCase):
    def test_high_precision_promotes_e5_structure_and_rrf(self) -> None:
        profile = load_product_profile(project_root=PROJECT_ROOT, profile_name="high_precision")

        self.assertEqual(profile.ingestion.parser_name, "docling")
        self.assertEqual(profile.ingestion.chunker_version, "structure_aware_v1")
        self.assertEqual(profile.ingestion.fallback_parser_name, "pdfplumber")
        self.assertEqual(profile.retrieval.fusion, "rrf")
        self.assertEqual(profile.retrieval.rrf_k, 60)
        self.assertTrue(profile.retrieval.reranker_enabled)
        self.assertFalse(profile.retrieval.query_expansion)

    def test_standard_keeps_same_e5_ingestion_but_disables_reranker(self) -> None:
        profile = load_product_profile(project_root=PROJECT_ROOT, profile_name="standard")

        self.assertEqual(profile.ingestion.chunker_version, "structure_aware_v1")
        self.assertEqual(profile.retrieval.fusion, "rrf")
        self.assertFalse(profile.retrieval.reranker_enabled)
        self.assertIn("config_hash", product_profile_to_dict(profile))

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaises(ProductConfigurationError):
            load_product_profile(project_root=PROJECT_ROOT, profile_name="unknown")

    def test_invalid_profile_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            target = root / "configs" / "product"
            target.mkdir(parents=True)
            (target / "standard.yaml").write_text(
                "schema_version: wrong\nname: standard\n",
                encoding="utf-8",
            )
            with self.assertRaises(ProductConfigurationError):
                load_product_profile(project_root=root, profile_name="standard")

    def test_factory_uses_profile_fusion_and_reranker_boundary(self) -> None:
        high_precision = load_product_profile(
            project_root=PROJECT_ROOT,
            profile_name="high_precision",
        )
        standard = load_product_profile(project_root=PROJECT_ROOT, profile_name="standard")
        with tempfile.TemporaryDirectory() as temp_dir:
            sparse = PersistentBM25Repository(
                _NoopIndex(),  # type: ignore[arg-type]
                root_dir=Path(temp_dir),
                tokenizer=_WhitespaceTokenizer(),
            )
            high_precision_service = build_service_from_components(
                profile=high_precision,
                index_repository=_NoopIndex(),  # type: ignore[arg-type]
                sparse_repository=sparse,
                llm=object(),
                reranker_model=object(),
            )
            standard_service = build_service_from_components(
                profile=standard,
                index_repository=_NoopIndex(),  # type: ignore[arg-type]
                sparse_repository=sparse,
                llm=object(),
            )

        high_precision_retriever = high_precision_service._retrieval_service._retriever  # noqa: SLF001
        standard_retriever = standard_service._retrieval_service._retriever  # noqa: SLF001
        self.assertEqual(high_precision_retriever._fusion, "rrf")  # noqa: SLF001
        self.assertIsNotNone(high_precision_retriever._reranker)  # noqa: SLF001
        self.assertEqual(standard_retriever._fusion, "rrf")  # noqa: SLF001
        self.assertIsNone(standard_retriever._reranker)  # noqa: SLF001


if __name__ == "__main__":
    unittest.main()
