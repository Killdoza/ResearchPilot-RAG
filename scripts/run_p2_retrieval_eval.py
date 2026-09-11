# ruff: noqa: E501  # Long Markdown report prose is intentionally kept readable.
"""运行 P2 的真实检索层对照；不调用 DeepSeek，也不生成答案。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.config import build_settings, ensure_runtime_directories, require_ascii_chroma_path  # noqa: E402
from researchpilot.evaluation import (  # noqa: E402
    EvaluationDataError,
    load_corpus_manifest,
    paper_ids_from_manifest,
)
from researchpilot.evidence_control import EvidenceCandidate, QueryAwareCandidateController  # noqa: E402
from researchpilot.index import ChromaVectorIndexRepository  # noqa: E402
from researchpilot.p2_dataset import (  # noqa: E402
    load_p2_cases,
    validate_frozen_p2_manifest,
    validate_p2_cases,
)
from researchpilot.p2_evaluation import (  # noqa: E402
    evaluate_p2_retrieval_case,
    summarize_p2_retrieval,
)
from researchpilot.p2_retrieval import PersistentHybridPerPaperProvider  # noqa: E402
from researchpilot.p2_runner import _runtime_outcome, load_p2_execution_config  # noqa: E402
from researchpilot.p2_runtime import P2RuntimePlanner  # noqa: E402
from researchpilot.query_plan import P2RunMode  # noqa: E402
from researchpilot.retrieval import JiebaTokenizer, PersistentHybridRetriever, SentenceTransformerReranker  # noqa: E402
from researchpilot.scope_resolution import PaperScopeCatalog  # noqa: E402
from researchpilot.sparse import PersistentBM25Repository  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P2 检索层 Dev/Test 对照（无 LLM 调用）")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_p2_v0.2.jsonl")
    parser.add_argument(
        "--dataset-manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.manifest.json"
    )
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.corpus.json")
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "data/runtime/evaluations")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "eval/results")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def build_persistent_retriever(*, embedding_model: str, reranker_model: str | None, rerank_enabled: bool):
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

    settings = build_settings()
    require_ascii_chroma_path(settings.evaluation_chroma_dir, "RESEARCHPILOT_EVALUATION_CHROMA_DIR")
    ensure_runtime_directories(settings)
    vectorstore = Chroma(
        collection_name=settings.evaluation_collection_name,
        embedding_function=HuggingFaceEmbeddings(model_name=embedding_model),
        persist_directory=str(settings.evaluation_chroma_dir),
    )
    index = ChromaVectorIndexRepository(vectorstore)
    sparse = PersistentBM25Repository(index, root_dir=settings.sparse_index_dir, tokenizer=JiebaTokenizer())
    reranker = SentenceTransformerReranker(CrossEncoder(reranker_model)) if rerank_enabled and reranker_model else None
    return PersistentHybridRetriever(
        index,
        sparse,
        fusion="rrf",
        reranker=reranker,
        dense_limit_per_query=10,
        bm25_limit_per_query=10,
        result_limit=10,
        rrf_k=60,
    ), settings


def _candidate_dict(item: EvidenceCandidate) -> dict[str, Any]:
    chunk = item.result.chunk
    return {
        "paper_key": item.paper_key,
        "chunk_id": chunk.chunk_id,
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "section_path": list(chunk.section_path),
        "dense_rank": item.result.dense_rank,
        "dense_score": item.result.dense_score,
        "sparse_rank": item.result.sparse_rank,
        "sparse_score": item.result.sparse_score,
        "fusion_score": item.result.fusion_score,
        "rerank_score": item.result.rerank_score,
    }


def _render_markdown(
    *, config_name: str, split: str, summary: Any, config_sha256: str, coverage_gate_minimum_score: float
) -> str:
    def metric(value):
        return "N/A" if value is None else f"{value:.3f}"

    return (
        "\n".join(
            [
                f"# P2 检索层结果：{config_name} / {split}",
                "",
                "本报告只评估真实检索与证据覆盖，未调用 DeepSeek、未生成答案。Gold 仅用于离线评分。",
                "",
                "| 指标 | 结果 |",
                "|---|---:|",
                f"| 题数 | {summary.evaluated_cases} |",
                f"| 可回答 / 不可回答 | {summary.answerable_cases} / {summary.unanswerable_cases} |",
                f"| Per-paper Recall | {metric(summary.per_paper_recall)} |",
                f"| Evidence Coverage | {metric(summary.evidence_coverage_rate)} |",
                f"| Coverage Gate Pass | {metric(summary.coverage_gate_pass_rate)} |",
                f"| Retry Rate | {metric(summary.retry_rate)} |",
                f"| 生成前拒答率 | {metric(summary.pre_generation_refusal_rate)} |",
                f"| 平均检索延迟 (ms) | {summary.mean_retrieval_latency_ms:.1f} |",
                f"| P50 / P95 检索延迟 (ms) | {summary.p50_retrieval_latency_ms:.1f} / {summary.p95_retrieval_latency_ms:.1f} |",
                "",
                f"Coverage Gate 最低重排分数：`{coverage_gate_minimum_score:.3f}`",
                f"配置 SHA-256：`{config_sha256}`",
            ]
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        config = load_p2_execution_config(args.config)
        frozen = validate_frozen_p2_manifest(dataset_path=args.dataset, manifest_path=args.dataset_manifest)
        corpus = load_corpus_manifest(args.corpus)
        cases = tuple(case for case in load_p2_cases(args.dataset) if case.split == args.split)
        validate_p2_cases(
            cases=cases,
            corpus=corpus,
            paper_splits=frozen["paper_splits"],
            page_counts=frozen["pdf_page_counts"],
        )
        if not cases:
            raise EvaluationDataError(f"P2 数据集中不存在 {args.split} 题目")
        paper_ids = paper_ids_from_manifest(corpus, require_hashes=True)
        catalog = PaperScopeCatalog.from_corpus(corpus)
        planner = P2RuntimePlanner(catalog=catalog)
        retriever, _ = build_persistent_retriever(
            embedding_model=config.retrieval_experiment.embedding_model,
            reranker_model=config.retrieval_experiment.reranker_model,
            rerank_enabled=config.retrieval_experiment.reranker,
        )
    except Exception as error:
        print(f"P2 检索评测启动失败：{error}", file=sys.stderr)
        return 2

    quality = {paper.paper_key: "normal" for paper in corpus.papers}
    provider = PersistentHybridPerPaperProvider(
        retriever=retriever,
        paper_id_by_key=paper_ids,
        user_id=config.evaluation_user_id,
        session_id=config.evaluation_session_id,
        extraction_quality_by_key=quality,
    )
    controller = QueryAwareCandidateController(
        provider=provider,
        minimum_score=config.coverage_gate_minimum_score,
        per_paper_quota=config.per_paper_quota,
    )
    inverse_ids = {paper_id: paper_key for paper_key, paper_id in paper_ids.items()}
    records: list[dict[str, Any]] = []
    evaluations = []
    try:
        for case in cases:
            plan, fallback = _runtime_outcome(case=case, config=config, planner=planner)
            started = time.perf_counter()
            decision = None
            if plan is None:
                selected: tuple[EvidenceCandidate, ...] = ()
                refusal_reason = fallback
            elif config.mode == P2RunMode.STATIC:
                execution = retriever.retrieve(
                    search_query=case.question,
                    user_id=config.evaluation_user_id,
                    session_id=config.evaluation_session_id,
                    dense_limit_per_query=config.retrieval_experiment.dense_limit_per_query,
                    bm25_limit_per_query=config.retrieval_experiment.bm25_limit_per_query,
                    result_limit=config.retrieval_experiment.result_limit,
                    rerank_enabled=config.retrieval_experiment.reranker,
                )
                selected = tuple(
                    EvidenceCandidate(paper_key=inverse_ids[item.chunk.paper_id], result=item)
                    for item in execution.results
                    if item.chunk.paper_id in inverse_ids
                )
                refusal_reason = None
            elif not plan.needs_retrieval or not plan.coverage_gate_required:
                # End-to-End rules may intentionally identify a message as
                # general chat / insufficiently cross-document.  It must not
                # be forced through a cross-document Coverage Gate: record a
                # safe pre-generation refusal instead.
                selected = ()
                refusal_reason = "scope_or_query_type_insufficient"
            else:
                decision = controller.retrieve_with_retry(
                    query=case.question, plan=plan, final_limit=config.retrieval_experiment.result_limit
                )
                selected = decision.selected
                refusal_reason = decision.refusal_reason
            elapsed_ms = (time.perf_counter() - started) * 1000
            outcome = evaluate_p2_retrieval_case(
                case=case,
                mode=config.mode.value,
                selected=selected,
                retrieval_latency_ms=elapsed_ms,
                decision=decision,
            )
            evaluations.append(outcome)
            records.append(
                {
                    "schema_version": "p2_retrieval_eval_run_v1",
                    "created_at": datetime.now(UTC).isoformat(),
                    "experiment": config.retrieval_experiment.name,
                    "mode": config.mode.value,
                    "case_id": case.case_id,
                    "split": case.split,
                    "question": case.question,
                    "plan": plan.to_dict() if plan else None,
                    "fallback_reason": fallback,
                    "decision": decision.trace_fields()
                    if decision
                    else {"refused": bool(fallback), "refusal_reason": refusal_reason},
                    "selected": [_candidate_dict(item) for item in selected],
                    "evaluation": outcome.to_dict(),
                }
            )
        summary = summarize_p2_retrieval(evaluations)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.runtime_dir.mkdir(parents=True, exist_ok=True)
        run_path = args.runtime_dir / f"p2_{config.retrieval_experiment.name}_{args.split}_{stamp}.jsonl"
        run_path.write_text(
            "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in records), encoding="utf-8"
        )
        args.results_dir.mkdir(parents=True, exist_ok=True)
        report_path = args.results_dir / f"p2_{config.retrieval_experiment.name}_{args.split}_retrieval.md"
        if report_path.exists() and not args.overwrite:
            raise EvaluationDataError(f"结果已存在：{report_path}；如确认覆盖请传入 --overwrite")
        report_path.write_text(
            _render_markdown(
                config_name=config.retrieval_experiment.name,
                split=args.split,
                summary=summary,
                config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                coverage_gate_minimum_score=config.coverage_gate_minimum_score,
            ),
            encoding="utf-8",
        )
    except Exception as error:
        print(f"P2 检索评测失败：{error}", file=sys.stderr)
        return 1
    print(f"P2 检索评测完成：{config.retrieval_experiment.name} / {args.split} / {len(evaluations)} 题")
    print(f"逐题记录：{run_path}")
    print(f"聚合报告：{report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
