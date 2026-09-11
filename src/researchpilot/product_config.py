"""产品运行配置：把已验证的解析和检索策略从页面代码中移出。

实验 YAML 用于复现实验；本模块只读取 ``configs/product`` 下的版本化产品
Profile。Streamlit、FastAPI 和后续 CLI 通过同一 Profile 构建运行时，避免
各入口悄悄使用不同的 Parser、融合策略或 Reranker。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from .ingestion import IngestionConfig

PRODUCT_PROFILE_SCHEMA_VERSION = "researchpilot_product_profile_v1"
SUPPORTED_PRODUCT_PROFILES = frozenset({"standard", "high_precision", "full_paper", "strict_crossdoc"})


class ProductConfigurationError(ValueError):
    """产品 Profile 缺失或格式非法。"""


@dataclass(frozen=True)
class ProductRetrievalConfig:
    """一个产品模式下的可复现检索预算。"""

    query_expansion: bool
    dense_limit_per_query: int
    bm25_limit_per_query: int
    fusion: str
    rrf_k: int
    reranker_enabled: bool
    result_limit: int

    def __post_init__(self) -> None:
        if not isinstance(self.query_expansion, bool):
            raise ProductConfigurationError("retrieval.query_expansion 必须是布尔值。")
        if self.fusion not in {"set_union", "rrf"}:
            raise ProductConfigurationError("retrieval.fusion 必须是 set_union 或 rrf。")
        for value, field_name in (
            (self.dense_limit_per_query, "dense_limit_per_query"),
            (self.bm25_limit_per_query, "bm25_limit_per_query"),
            (self.rrf_k, "rrf_k"),
            (self.result_limit, "result_limit"),
        ):
            if not isinstance(value, int) or value < 1:
                raise ProductConfigurationError(f"retrieval.{field_name} 必须是正整数。")
        if not isinstance(self.reranker_enabled, bool):
            raise ProductConfigurationError("retrieval.reranker_enabled 必须是布尔值。")


@dataclass(frozen=True)
class ProductProfile:
    """供产品入口使用的不可变配置快照。"""

    name: str
    label: str
    description: str
    ingestion: IngestionConfig
    retrieval: ProductRetrievalConfig
    config_hash: str
    source_path: Path

    @property
    def is_experimental(self) -> bool:
        return self.name in {"full_paper", "strict_crossdoc"}

    def display_summary(self) -> str:
        reranker = "BGE Reranker" if self.retrieval.reranker_enabled else "未启用 Reranker"
        return (
            f"{self.label} · {self.ingestion.parser_name} + "
            f"{self.ingestion.chunker_version} · Dense + BM25 + "
            f"{self.retrieval.fusion.upper()} · {reranker}"
        )


def _require_mapping(value: object, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProductConfigurationError(f"{field_name} 必须是对象。")
    return value


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProductConfigurationError(f"{field_name} 必须是非空文本。")
    return value.strip()


def _profile_hash(raw: dict[str, Any]) -> str:
    encoded = json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:16]


def product_profile_path(*, project_root: Path, profile_name: str) -> Path:
    if profile_name not in SUPPORTED_PRODUCT_PROFILES:
        supported = "、".join(sorted(SUPPORTED_PRODUCT_PROFILES))
        raise ProductConfigurationError(f"未知产品模式 {profile_name!r}；支持：{supported}。")
    configured_path = project_root / "configs" / "product" / f"{profile_name}.yaml"
    if configured_path.is_file():
        return configured_path
    # 单元测试会使用临时 project_root；安装为包后也可能把运行数据与代码目录分开。
    # 此时读取随源代码发布的默认 Profile，而不是退回到页面内的魔法常量。
    packaged_path = Path(__file__).resolve().parents[2] / "configs" / "product" / f"{profile_name}.yaml"
    return packaged_path


def load_product_profile(*, project_root: Path, profile_name: str = "high_precision") -> ProductProfile:
    """加载、严格校验并计算一个产品 Profile 的内容哈希。"""
    path = product_profile_path(project_root=project_root, profile_name=profile_name)
    if not path.is_file():
        raise ProductConfigurationError(f"产品配置不存在：{path}")
    try:
        raw_value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ProductConfigurationError(f"无法读取产品配置 {path.name}：{exc}") from exc
    raw = _require_mapping(raw_value, path.name)
    if raw.get("schema_version") != PRODUCT_PROFILE_SCHEMA_VERSION:
        raise ProductConfigurationError(f"{path.name} 的 schema_version 必须为 {PRODUCT_PROFILE_SCHEMA_VERSION}。")
    name = _require_text(raw.get("name"), "name")
    if name != profile_name:
        raise ProductConfigurationError("产品配置文件名必须与 name 字段一致。")
    ingestion_raw = _require_mapping(raw.get("ingestion"), "ingestion")
    retrieval_raw = _require_mapping(raw.get("retrieval"), "retrieval")
    try:
        ingestion = IngestionConfig(
            min_document_text_length=int(ingestion_raw.get("min_document_text_length", 10)),
            head_page_limit=int(ingestion_raw.get("head_page_limit", 3)),
            parent_chunk_size=int(ingestion_raw["parent_chunk_size"]),
            parent_chunk_overlap=int(ingestion_raw["parent_chunk_overlap"]),
            child_chunk_size=int(ingestion_raw["child_chunk_size"]),
            child_chunk_overlap=int(ingestion_raw["child_chunk_overlap"]),
            parser_name=_require_text(ingestion_raw.get("parser_name"), "ingestion.parser_name"),
            parser_version=_require_text(ingestion_raw.get("parser_version"), "ingestion.parser_version"),
            fallback_parser_name=(
                _require_text(ingestion_raw["fallback_parser_name"], "ingestion.fallback_parser_name")
                if ingestion_raw.get("fallback_parser_name") is not None
                else None
            ),
            fallback_parser_version=(
                _require_text(ingestion_raw["fallback_parser_version"], "ingestion.fallback_parser_version")
                if ingestion_raw.get("fallback_parser_version") is not None
                else None
            ),
            enable_ocr=ingestion_raw.get("enable_ocr", False),
            chunker_version=_require_text(ingestion_raw.get("chunker_version"), "ingestion.chunker_version"),
        )
        retrieval = ProductRetrievalConfig(
            query_expansion=retrieval_raw.get("query_expansion", False),
            dense_limit_per_query=int(retrieval_raw["dense_limit_per_query"]),
            bm25_limit_per_query=int(retrieval_raw["bm25_limit_per_query"]),
            fusion=_require_text(retrieval_raw.get("fusion"), "retrieval.fusion"),
            rrf_k=int(retrieval_raw["rrf_k"]),
            reranker_enabled=retrieval_raw.get("reranker_enabled", False),
            result_limit=int(retrieval_raw["result_limit"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, ProductConfigurationError):
            raise
        raise ProductConfigurationError(f"{path.name} 的字段无效：{exc}") from exc
    return ProductProfile(
        name=name,
        label=_require_text(raw.get("label"), "label"),
        description=_require_text(raw.get("description"), "description"),
        ingestion=ingestion,
        retrieval=retrieval,
        config_hash=_profile_hash(raw),
        source_path=path,
    )


def product_profile_to_dict(profile: ProductProfile) -> dict[str, object]:
    """返回不含本机绝对路径的可展示、可审计 Profile 摘要。"""
    return {
        "name": profile.name,
        "label": profile.label,
        "description": profile.description,
        "config_hash": profile.config_hash,
        "ingestion": asdict(profile.ingestion),
        "retrieval": asdict(profile.retrieval),
    }
