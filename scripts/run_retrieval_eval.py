"""运行独立、可复现的论文检索评测，并输出逐题 JSONL 与聚合报告。"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.config import (  # noqa: E402
    build_settings,
    ensure_runtime_directories,
    require_ascii_chroma_path,
    require_deepseek_api_key,
    validate_storage_id,
)
from researchpilot.evaluation import (  # noqa: E402
    CaseEvaluation,
    EvalCase,
    EvaluationDataError,
    EvaluationSummary,
    ExperimentConfig,
    case_evaluation_to_dict,
    evaluate_case,
    load_corpus_manifest,
    load_eval_cases,
    load_experiment_config,
    paper_ids_from_manifest,
    render_summary_markdown,
    summarize_evaluations,
    validate_dataset_manifest_coverage,
)
from researchpilot.index import ChromaVectorIndexRepository  # noqa: E402
from researchpilot.retrieval import (  # noqa: E402
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    JiebaTokenizer,
    LangChainQueryExpander,
    PersistentHybridRetriever,
    RetrievalResult,
    RetrievalService,
    SentenceTransformerReranker,
)
from researchpilot.schemas import to_jsonable  # noqa: E402
from researchpilot.sparse import PersistentBM25Repository  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ResearchPilot 页级检索评测。")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "eval/datasets/ai_cs_eval_v0.1.jsonl",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/e0_dense.yaml",
    )
    parser.add_argument("--split", choices=("dev", "test", "all"), default="dev")
    parser.add_argument("--user-id", default="eval_ai_cs_v01", help="评测专用 user_id。")
    parser.add_argument("--session-id", default="corpus_v01", help="评测专用 session_id。")
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        default=PROJECT_ROOT / "data/runtime/evaluations",
        help="逐题 JSONL 的忽略目录。",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "eval/results",
        help="聚合 CSV/Markdown 输出目录。",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的非占位聚合结果。",
    )
    return parser.parse_args(argv)


def select_cases(cases: Sequence[EvalCase], split: str) -> tuple[EvalCase, ...]:
    selected = tuple(cases) if split == "all" else tuple(case for case in cases if case.split == split)
    if not selected:
        raise EvaluationDataError(f"数据集中没有 split={split} 的评测问题。")
    return selected


def build_retrieval_service(*, config: ExperimentConfig):
    """按 YAML 组装检索器；E0 路径不会导入或调用 DeepSeek。"""
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    settings = build_settings()
    require_ascii_chroma_path(settings.evaluation_chroma_dir, "RESEARCHPILOT_EVALUATION_CHROMA_DIR")
    ensure_runtime_directories(settings)
    embeddings = HuggingFaceEmbeddings(model_name=config.embedding_model) if config.strategy != "bm25" else None
    vectorstore = Chroma(
        collection_name=settings.evaluation_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.evaluation_chroma_dir),
    )
    index_repository = ChromaVectorIndexRepository(vectorstore)
    sparse_repository = PersistentBM25Repository(
        index_repository,
        root_dir=settings.sparse_index_dir,
        tokenizer=JiebaTokenizer(),
    )
    if config.strategy == "dense":
        return RetrievalService(
            DenseRetriever(index_repository, candidate_limit=config.dense_limit_per_query)
        ), settings

    if config.strategy == "bm25":
        return (
            RetrievalService(
                BM25Retriever(index_repository, sparse_repository, candidate_limit=config.bm25_limit_per_query)
            ),
            settings,
        )

    query_expander = None
    reranker = None
    if config.query_expansion:
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            api_key=require_deepseek_api_key(settings),
            base_url=settings.deepseek_base_url,
            model=settings.deepseek_model,
            streaming=False,
        )
        query_expander = LangChainQueryExpander(llm, expansion_count=config.query_expansion_count)
    if config.reranker:
        from sentence_transformers import CrossEncoder

        reranker = SentenceTransformerReranker(CrossEncoder(str(config.reranker_model)))
    if config.strategy == "hybrid_union" and config.fusion == "parent_id_set_union":
        retriever = HybridRetriever(
            index_repository,
            query_expander=query_expander,
            reranker=reranker,
            dense_limit_per_query=config.dense_limit_per_query,
            bm25_limit_per_query=config.bm25_limit_per_query,
            result_limit=config.result_limit,
        )
    else:
        retriever = PersistentHybridRetriever(
            index_repository,
            sparse_repository,
            fusion="rrf" if config.strategy == "hybrid_rrf" else "set_union",
            query_expander=query_expander,
            reranker=reranker,
            dense_limit_per_query=config.dense_limit_per_query,
            bm25_limit_per_query=config.bm25_limit_per_query,
            result_limit=config.result_limit,
            rrf_k=config.rrf_k or 60,
        )
    return RetrievalService(retriever), settings


def build_run_record(
    *,
    case: EvalCase,
    retrieval: RetrievalResult,
    outcome: CaseEvaluation,
    experiment: ExperimentConfig,
    user_id: str,
    session_id: str,
    experiment_config_sha256: str | None = None,
) -> dict[str, Any]:
    """构造逐题运行记录；原始 PDF 与生成答案均不写入评测日志。"""
    return {
        "schema_version": "retrieval_eval_run_v1",
        "run_id": f"eval_{uuid.uuid4().hex}",
        "created_at": datetime.now(UTC).isoformat(),
        "experiment": experiment.name,
        "experiment_config_sha256": experiment_config_sha256,
        "user_id": user_id,
        "session_id": session_id,
        "case_id": case.case_id,
        "split": case.split,
        "task_type": case.task_type,
        "question": case.question,
        "retrieval": to_jsonable(retrieval.trace),
        "evaluation": case_evaluation_to_dict(outcome),
    }


def write_jsonl_records(
    *, runtime_dir: Path, experiment_name: str, split: str, records: Sequence[dict[str, Any]]
) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output_path = runtime_dir / f"{experiment_name}_{split}_{timestamp}.jsonl"
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def _summary_row(*, experiment: ExperimentConfig, split: str, summary: EvaluationSummary) -> dict[str, str]:
    row = {
        "experiment": experiment.name,
        "split": split,
        "evaluated_cases": str(summary.evaluated_cases),
        "answerable_cases": str(summary.answerable_cases),
        "unanswerable_cases": str(summary.unanswerable_cases),
        "mrr": f"{summary.mrr:.6f}",
        "mean_retrieval_latency_ms": f"{summary.mean_retrieval_latency_ms:.6f}",
        "p50_retrieval_latency_ms": f"{summary.p50_retrieval_latency_ms:.6f}",
        "p95_retrieval_latency_ms": f"{summary.p95_retrieval_latency_ms:.6f}",
        "unanswerable_top1_retrieval_rate": ""
        if summary.unanswerable_top1_retrieval_rate is None
        else f"{summary.unanswerable_top1_retrieval_rate:.6f}",
    }
    for cutoff in sorted(summary.question_recall_at):
        row[f"question_recall_at_{cutoff}"] = f"{summary.question_recall_at[cutoff]:.6f}"
        row[f"evidence_recall_at_{cutoff}"] = f"{summary.evidence_recall_at[cutoff]:.6f}"
        row[f"ndcg_at_{cutoff}"] = f"{summary.ndcg_at[cutoff]:.6f}"
    return row


def write_aggregate_results(
    *, results_dir: Path, experiment: ExperimentConfig, split: str, summary: EvaluationSummary, overwrite: bool
) -> tuple[Path, Path]:
    results_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = results_dir / f"{experiment.name}_{split}.md"
    csv_path = results_dir / f"{experiment.name}_{split}.csv"
    for path in (markdown_path, csv_path):
        if path.exists() and not overwrite:
            content = path.read_text(encoding="utf-8")
            if "pending_real_corpus_and_annotations" not in content and "状态：尚未运行" not in content:
                raise EvaluationDataError(f"聚合结果已存在：{path}。如确认覆盖，请显式传入 --overwrite。")
    markdown_path.write_text(
        render_summary_markdown(experiment=experiment, split=split, summary=summary),
        encoding="utf-8",
    )
    row = _summary_row(experiment=experiment, split=split, summary=summary)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    return markdown_path, csv_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validate_storage_id(args.user_id, "user_id")
        validate_storage_id(args.session_id, "session_id")
        manifest = load_corpus_manifest(args.manifest)
        cases = load_eval_cases(args.dataset)
        validate_dataset_manifest_coverage(cases, manifest)
        paper_ids = paper_ids_from_manifest(manifest, require_hashes=True)
        selected_cases = select_cases(cases, args.split)
        experiment = load_experiment_config(args.config)
        config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
        retrieval_service, _ = build_retrieval_service(config=experiment)
    except Exception as error:
        # 这里统一返回清晰的命令行失败码；真实模型/索引异常的原文仍会显示在终端。
        print(f"评测启动失败：{error}", file=sys.stderr)
        return 2

    outcomes: list[CaseEvaluation] = []
    records: list[dict[str, Any]] = []
    try:
        for case in selected_cases:
            retrieval = retrieval_service.retrieve(
                query=case.question,
                search_query=case.question,
                user_id=args.user_id,
                session_id=args.session_id,
                recent_history="",
            )
            outcome = evaluate_case(
                case,
                retrieval.results,
                paper_ids=paper_ids,
                retrieval_latency_ms=retrieval.trace.elapsed_ms,
            )
            outcomes.append(outcome)
            records.append(
                build_run_record(
                    case=case,
                    retrieval=retrieval,
                    outcome=outcome,
                    experiment=experiment,
                    user_id=args.user_id,
                    session_id=args.session_id,
                    experiment_config_sha256=config_sha256,
                )
            )
        summary = summarize_evaluations(outcomes)
        runtime_path = write_jsonl_records(
            runtime_dir=args.runtime_dir,
            experiment_name=experiment.name,
            split=args.split,
            records=records,
        )
        markdown_path, csv_path = write_aggregate_results(
            results_dir=args.results_dir,
            experiment=experiment,
            split=args.split,
            summary=summary,
            overwrite=args.overwrite,
        )
    except Exception as error:
        print(f"评测运行失败：{error}", file=sys.stderr)
        return 1
    print(f"评测完成：{experiment.name} / {args.split} / {summary.evaluated_cases} 条问题。")
    print(f"逐题记录：{runtime_path}")
    print(f"聚合报告：{markdown_path}")
    print(f"聚合 CSV：{csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
