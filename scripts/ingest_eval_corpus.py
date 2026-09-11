"""将冻结的评测论文导入独立 Scope；默认只执行只读 dry-run。"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.config import (  # noqa: E402
    ConfigurationError,
    build_settings,
    ensure_runtime_directories,
    require_ascii_chroma_path,
    validate_storage_id,
)
from researchpilot.evaluation import (  # noqa: E402
    CorpusManifest,
    EvaluationDataError,
    ExperimentConfig,
    load_corpus_manifest,
    load_experiment_config,
)
from researchpilot.index import ChromaVectorIndexRepository, ChunkKind  # noqa: E402
from researchpilot.ingestion import (  # noqa: E402
    IngestionConfig,
    IngestionResult,
    build_document_parser,
    ingest_pdf,
)
from researchpilot.schemas import sha256_bytes  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入 ResearchPilot AI/CS 评测语料。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/e0_dense.yaml",
        help="用于固定 Parser 与 Chunk 参数的实验配置。",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data/raw/ai_cs_eval_v0.1",
        help="被 Git 忽略的原始 PDF 目录。",
    )
    parser.add_argument("--user-id", default="eval_ai_cs_v01", help="独立评测 user_id。")
    parser.add_argument("--session-id", default="corpus_v01", help="独立评测 session_id。")
    parser.add_argument("--apply", action="store_true", help="确认写入 Chroma；未传入时仅检查文件和 SHA-256。")
    parser.add_argument(
        "--reset-scope",
        action="store_true",
        help="仅配合 --apply 使用：先清空指定评测 Scope，再重新导入。",
    )
    return parser.parse_args(argv)


def _ingestion_config(config: ExperimentConfig) -> IngestionConfig:
    return IngestionConfig(
        parent_chunk_size=config.parent_chunk_size,
        parent_chunk_overlap=config.parent_chunk_overlap,
        child_chunk_size=config.child_chunk_size,
        child_chunk_overlap=config.child_chunk_overlap,
        parser_name=config.parser_name,
        parser_version=config.parser_version,
        enable_ocr=config.enable_ocr,
        chunker_version=config.chunker_version,
    )


def resolve_manifest_pdf(*, raw_dir: Path, local_filename: str) -> Path:
    """支持扁平语料与按 Dev/Test 分目录保存的冻结论文。

    Manifest 仍只记录文件名，避免把本机目录结构写入版本化数据；若递归查找
    出现同名 PDF，则拒绝猜测，以免把 Test 文件误导入 Dev Scope。
    """
    root = raw_dir.resolve()
    direct = (root / local_filename).resolve()
    try:
        direct.relative_to(root)
    except ValueError as error:
        raise EvaluationDataError(f"论文文件路径越出 raw-dir：{local_filename}") from error
    if direct.is_file():
        return direct
    matches = tuple(path.resolve() for path in root.rglob(local_filename) if path.is_file())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise EvaluationDataError(f"找不到论文 PDF：{direct}")
    raise EvaluationDataError(
        f"raw-dir 下存在多个同名 PDF，拒绝猜测导入目标：{local_filename} -> {list(map(str, matches))}"
    )


def inspect_local_corpus(*, manifest: CorpusManifest, raw_dir: Path) -> tuple[dict[str, Any], ...]:
    """只读验证本地 PDF，且在未冻结哈希时输出待写回 Manifest 的实际值。"""
    if not manifest.papers:
        raise EvaluationDataError("论文 Manifest 为空；请先添加至少一篇开放获取论文。")
    resolved_raw_dir = raw_dir.resolve()
    inspected: list[dict[str, Any]] = []
    for paper in manifest.papers:
        pdf_path = resolve_manifest_pdf(raw_dir=resolved_raw_dir, local_filename=paper.local_filename)
        actual_hash = sha256_bytes(pdf_path.read_bytes())
        if paper.sha256 and paper.sha256 != actual_hash:
            raise EvaluationDataError(
                f"{paper.paper_key} 的 SHA-256 不匹配；Manifest={paper.sha256}，实际={actual_hash}。"
            )
        inspected.append(
            {
                "paper_key": paper.paper_key,
                "local_filename": paper.local_filename,
                "path": str(pdf_path),
                "sha256": actual_hash,
                "hash_frozen": paper.sha256 is not None,
            }
        )
    return tuple(inspected)


def _build_service(*, config: ExperimentConfig):
    """仅在 --apply 后导入重依赖；评测导入不调用 DeepSeek。"""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = build_settings()
    require_ascii_chroma_path(settings.evaluation_chroma_dir, "RESEARCHPILOT_EVALUATION_CHROMA_DIR")
    ensure_runtime_directories(settings)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model)
    vectorstore = Chroma(
        collection_name=settings.evaluation_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.evaluation_chroma_dir),
    )
    repository = ChromaVectorIndexRepository(vectorstore)
    return repository, settings


def parse_corpus_for_indexing(
    *, manifest: CorpusManifest, inspected: tuple[dict[str, Any], ...], config: ExperimentConfig
) -> tuple[tuple[dict[str, Any], ...], tuple[IngestionResult, ...]]:
    """先完整解析冻结语料，确保写入前所有 PDF 都能被解析。"""
    parsed_results: list[IngestionResult] = []
    imported: list[dict[str, Any]] = []
    ingestion_config = _ingestion_config(config)
    document_parser = build_document_parser(ingestion_config)
    for paper, inspected_item in zip(manifest.papers, inspected, strict=True):
        result = ingest_pdf(
            Path(str(inspected_item["path"])),
            source_name=paper.local_filename,
            config=ingestion_config,
            parser=document_parser,
        )
        parsed_results.append(result)
        imported.append(
            {
                "paper_key": paper.paper_key,
                "paper_id": result.paper.paper_id,
                "sha256": result.paper.content_hash,
                "page_count": result.paper.page_count,
                "block_count": len(result.blocks),
                "parent_chunk_count": len(result.parent_chunks),
                "child_chunk_count": len(result.child_chunks),
                "parse_quality": result.parse_quality.to_dict() if result.parse_quality else None,
            }
        )
    return tuple(imported), tuple(parsed_results)


def write_block_artifacts(
    *, root: Path, experiment: ExperimentConfig, results: tuple[IngestionResult, ...]
) -> tuple[Path, ...]:
    """保存可复用的结构 Block JSON；原始 PDF 不会被复制或提交。"""
    output_dir = root / "data/runtime/parsed_blocks" / experiment.name
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[Path] = []
    for result in results:
        payload = {
            "schema_version": "parsed_blocks_v1",
            "paper": {
                "paper_id": result.paper.paper_id,
                "source_name": result.paper.source_name,
                "content_hash": result.paper.content_hash,
                "page_count": result.paper.page_count,
            },
            "parse_quality": result.parse_quality.to_dict() if result.parse_quality else None,
            "blocks": [
                {
                    "block_id": block.block_id,
                    "page": block.page,
                    "block_type": block.block_type.value,
                    "section_path": list(block.section_path),
                    "text": block.text,
                    "parser_name": block.parser_name,
                    "parser_version": block.parser_version,
                    "ocr_used": block.ocr_used,
                    "extraction_quality": block.extraction_quality,
                }
                for block in result.blocks
            ],
        }
        path = output_dir / f"{result.paper.content_hash}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        artifacts.append(path)
    return tuple(artifacts)


def write_ingestion_report(*, root: Path, record: dict[str, Any]) -> Path:
    output_dir = root / "data/runtime/evaluations"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"ingestion_{timestamp}.json"
    output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.reset_scope and not args.apply:
        print("--reset-scope 必须与 --apply 同时使用。", file=sys.stderr)
        return 2
    try:
        validate_storage_id(args.user_id, "user_id")
        validate_storage_id(args.session_id, "session_id")
        manifest = load_corpus_manifest(args.manifest)
        config = load_experiment_config(args.config)
        inspected = inspect_local_corpus(manifest=manifest, raw_dir=args.raw_dir)
    except (EvaluationDataError, ConfigurationError) as error:
        print(f"评测语料检查失败：{error}", file=sys.stderr)
        return 2

    unfrozen = [item for item in inspected if not item["hash_frozen"]]
    for item in inspected:
        print(f"{item['paper_key']}: SHA-256={item['sha256']}")
    if not args.apply:
        print("dry-run 完成：未写入 Chroma。确认每个哈希已写入 Manifest 后，使用 --apply 导入。")
        return 0
    if unfrozen:
        paper_keys = [str(item["paper_key"]) for item in unfrozen]
        print(
            f"拒绝导入：以下论文尚未在 Manifest 冻结 SHA-256：{paper_keys}。请将上方实际哈希写入 Manifest 后重试。",
            file=sys.stderr,
        )
        return 2

    try:
        repository, settings = _build_service(config=config)
        deleted_count = 0
        if args.reset_scope:
            deleted_count = repository.delete_session(user_id=args.user_id, session_id=args.session_id)
        imported, parsed_results = parse_corpus_for_indexing(manifest=manifest, inspected=inspected, config=config)
        for paper, result in zip(manifest.papers, parsed_results, strict=True):
            # source_name 是引用链的一部分，不能在跨论文批量写入时退化为语料集合名。
            repository.upsert(
                chunks=result.parent_chunks,
                chunk_kind=ChunkKind.PARENT,
                source_name=paper.local_filename,
                user_id=args.user_id,
                session_id=args.session_id,
            )
            repository.upsert(
                chunks=result.child_chunks,
                chunk_kind=ChunkKind.CHILD,
                source_name=paper.local_filename,
                user_id=args.user_id,
                session_id=args.session_id,
            )
        block_artifacts = write_block_artifacts(root=settings.project_root, experiment=config, results=parsed_results)
        report_path = write_ingestion_report(
            root=settings.project_root,
            record={
                "schema_version": "eval_ingestion_run_v1",
                "created_at": datetime.now(UTC).isoformat(),
                "experiment": config.name,
                "user_id": args.user_id,
                "session_id": args.session_id,
                "parser_name": config.parser_name,
                "parser_version": config.parser_version,
                "chunker_version": config.chunker_version,
                "enable_ocr": config.enable_ocr,
                "deleted_before_import": deleted_count,
                "papers": imported,
                "block_artifacts": [str(path) for path in block_artifacts],
            },
        )
    except Exception as error:  # 保留原异常信息，方便定位真实 PDF/模型/Chroma 问题。
        print(f"评测语料导入失败：{error}", file=sys.stderr)
        return 1
    print(f"已导入 {len(imported)} 篇论文到独立评测 Scope；报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
