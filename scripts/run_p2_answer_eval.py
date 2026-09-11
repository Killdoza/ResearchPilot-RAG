# ruff: noqa: E402, E501  # Runtime path bootstrap and Markdown report prose.
"""P2 Dev/Test 回答实验：真实检索、一次受约束生成与确定性引用评分。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_p2_retrieval_eval import build_persistent_retriever  # noqa:E402

from researchpilot.answer_evaluation import (
    AnswerCitation,
    GeneratedAnswer,
    evaluate_generated_answer,
    summarize_answer_evaluations,
)  # noqa:E402
from researchpilot.answer_runner import LangChainStructuredAnswerClient  # noqa:E402
from researchpilot.config import build_settings, require_deepseek_api_key  # noqa:E402
from researchpilot.evaluation import load_corpus_manifest, paper_ids_from_manifest  # noqa:E402
from researchpilot.evidence_control import EvidenceCandidate, QueryAwareCandidateController  # noqa:E402
from researchpilot.p2_answer_runner import generate_p2_answer  # noqa:E402
from researchpilot.p2_dataset import (  # noqa:E402
    load_p2_cases,
    validate_frozen_p2_manifest,
    validate_p2_cases,
)
from researchpilot.p2_retrieval import PersistentHybridPerPaperProvider  # noqa:E402
from researchpilot.p2_runner import _runtime_outcome, load_p2_execution_config  # noqa:E402
from researchpilot.p2_runtime import P2RuntimePlanner  # noqa:E402
from researchpilot.query_plan import P2RunMode  # noqa:E402
from researchpilot.scope_resolution import PaperScopeCatalog  # noqa:E402


def args_parser(argv=None):
    parser = argparse.ArgumentParser(description="运行 P2 受证据约束回答评测（会调用 DeepSeek）")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_p2_v0.2.jsonl")
    parser.add_argument(
        "--dataset-manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.manifest.json"
    )
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.corpus.json")
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "data/runtime/p2_evaluations")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "eval/results")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _candidate_dict(item):
    c = item.result.chunk
    return {
        "paper_key": item.paper_key,
        "chunk_id": c.chunk_id,
        "page_start": c.page_start,
        "page_end": c.page_end,
        "section_path": list(c.section_path),
        "rerank_score": item.result.rerank_score,
    }


def _markdown(name, split, summary, wrong_comparison_rate, protocol_rejections, config_hash):
    def val(x):
        return "N/A" if x is None else f"{x:.3f}"

    return (
        "\n".join(
            [
                f"# P2 回答与引用结果：{name} / {split}",
                "",
                "此报告的 Gold 仅用于离线评分；Planner、Retriever 与模型 Prompt 均不读取 Gold。",
                "",
                "| 指标 | 结果 |",
                "|---|---:|",
                f"| 题数 | {summary.evaluated_cases} |",
                f"| Citation Precision | {val(summary.citation_precision)} |",
                f"| Citation Recall | {val(summary.citation_recall)} |",
                f"| Citation F1 | {val(summary.citation_f1)} |",
                f"| 不可回答题正确拒答率 | {val(summary.correct_refusal_rate)} |",
                f"| 违规比较率 | {val(wrong_comparison_rate)} |",
                f"| G2 协议拦截数 | {protocol_rejections} |",
                f"| 平均 Prompt Token | {val(summary.mean_prompt_tokens)} |",
                f"| 平均 Completion Token | {val(summary.mean_completion_tokens)} |",
                f"| P50 / P95 端到端延迟 (ms) | {summary.p50_end_to_end_latency_ms:.1f} / {summary.p95_end_to_end_latency_ms:.1f} |",
                "",
                f"配置 SHA-256：`{config_hash}`",
                "",
                "语义正确性不由另一个模型伪评分；将在 Dev 失败审计中逐题复核。",
            ]
        )
        + "\n"
    )


def main(argv=None):
    args = args_parser(argv)
    try:
        config = load_p2_execution_config(args.config)
        frozen = validate_frozen_p2_manifest(dataset_path=args.dataset, manifest_path=args.dataset_manifest)
        corpus = load_corpus_manifest(args.corpus)
        cases = tuple(x for x in load_p2_cases(args.dataset) if x.split == args.split)
        validate_p2_cases(
            cases=cases, corpus=corpus, paper_splits=frozen["paper_splits"], page_counts=frozen["pdf_page_counts"]
        )
        paper_ids = paper_ids_from_manifest(corpus, require_hashes=True)
        planner = P2RuntimePlanner(catalog=PaperScopeCatalog.from_corpus(corpus))
        retriever, _ = build_persistent_retriever(
            embedding_model=config.retrieval_experiment.embedding_model,
            reranker_model=config.retrieval_experiment.reranker_model,
            rerank_enabled=config.retrieval_experiment.reranker,
        )
        from langchain_openai import ChatOpenAI

        settings = build_settings()
        client = LangChainStructuredAnswerClient(
            ChatOpenAI(
                api_key=require_deepseek_api_key(settings),
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                streaming=False,
            )
        )
    except Exception as error:
        print(f"P2 回答评测启动失败：{error}", file=sys.stderr)
        return 2
    provider = PersistentHybridPerPaperProvider(
        retriever=retriever,
        paper_id_by_key=paper_ids,
        user_id=config.evaluation_user_id,
        session_id=config.evaluation_session_id,
        extraction_quality_by_key={x.paper_key: "normal" for x in corpus.papers},
    )
    controller = QueryAwareCandidateController(
        provider=provider, minimum_score=config.coverage_gate_minimum_score, per_paper_quota=config.per_paper_quota
    )
    inverse = {v: k for k, v in paper_ids.items()}
    records = []
    answer_evals = []
    generated = []
    wrong = []
    protocol = 0
    for case in cases:
        plan, fallback = _runtime_outcome(case=case, config=config, planner=planner)
        started = time.perf_counter()
        decision = None
        if plan is None:
            selected = ()
            local_reason = fallback
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
                EvidenceCandidate(paper_key=inverse[x.chunk.paper_id], result=x)
                for x in execution.results
                if x.chunk.paper_id in inverse
            )
            local_reason = None
        elif not plan.needs_retrieval or not plan.coverage_gate_required:
            selected = ()
            local_reason = "scope_or_query_type_insufficient"
        else:
            decision = controller.retrieve_with_retry(
                query=case.question, plan=plan, final_limit=config.retrieval_experiment.result_limit
            )
            selected = decision.selected
            local_reason = decision.refusal_reason
        retrieval_ms = (time.perf_counter() - started) * 1000
        if local_reason and decision is None:
            from researchpilot.p2_answer_runner import _local_refusal

            answer_exec = _local_refusal(local_reason)
        else:
            answer_exec = generate_p2_answer(
                question=case.question,
                selected=selected,
                required_paper_keys=case.required_paper_keys,
                decision=decision,
                config=config,
                client=client,
            )
        p2answer = answer_exec.answer
        converted = GeneratedAnswer(
            answer=p2answer.answer,
            citations=tuple(
                AnswerCitation(x.paper_key, x.page_start, x.page_end, x.context_id) for x in p2answer.citations
            ),
            refused=p2answer.refused,
            refusal_reason=p2answer.refusal_reason,
            prompt_tokens=answer_exec.prompt_tokens,
            completion_tokens=answer_exec.completion_tokens,
            cost=answer_exec.cost,
            retrieval_latency_ms=retrieval_ms,
            generation_latency_ms=answer_exec.generation_latency_ms,
        )
        ev = evaluate_generated_answer(case=case, answer=converted)
        answer_evals.append(ev)
        generated.append(converted)
        invalid_comparison = not case.comparison_allowed and any(x.comparative for x in p2answer.claims)
        wrong.append(invalid_comparison)
        protocol += bool(answer_exec.validation_violations)
        records.append(
            {
                "schema_version": "p2_answer_eval_run_v1",
                "created_at": datetime.now(UTC).isoformat(),
                "experiment": config.retrieval_experiment.name,
                "mode": config.mode.value,
                "case_id": case.case_id,
                "split": case.split,
                "question": case.question,
                "plan": plan.to_dict() if plan else None,
                "fallback_reason": fallback,
                "decision": decision.trace_fields() if decision else {"refusal_reason": local_reason},
                "selected": [_candidate_dict(x) for x in selected],
                "answer": {
                    "answer": p2answer.answer,
                    "citations": [x.__dict__ for x in p2answer.citations],
                    "claims": [x.__dict__ for x in p2answer.claims],
                    "refused": p2answer.refused,
                    "refusal_reason": p2answer.refusal_reason,
                },
                "usage": {
                    "prompt_tokens": answer_exec.prompt_tokens,
                    "completion_tokens": answer_exec.completion_tokens,
                    "cost": answer_exec.cost,
                    "retrieval_latency_ms": retrieval_ms,
                    "generation_latency_ms": answer_exec.generation_latency_ms,
                },
                "evaluation": ev.to_dict(),
                "g2_violations": list(answer_exec.validation_violations),
            }
        )
    summary = summarize_answer_evaluations(outcomes=answer_evals, answers=generated)
    args.runtime_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run = args.runtime_dir / f"p2_answer_{config.retrieval_experiment.name}_{args.split}_{stamp}.jsonl"
    run.write_text("".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in records), encoding="utf-8")
    args.results_dir.mkdir(parents=True, exist_ok=True)
    report = args.results_dir / f"p2_{config.retrieval_experiment.name}_{args.split}_answer.md"
    report.write_text(
        _markdown(
            config.retrieval_experiment.name,
            args.split,
            summary,
            sum(wrong) / len(wrong) if wrong else None,
            protocol,
            __import__("hashlib").sha256(args.config.read_bytes()).hexdigest(),
        ),
        encoding="utf-8",
    )
    print(
        f"P2 回答评测完成：{config.retrieval_experiment.name}/{args.split}/{len(cases)} 题\n逐题记录：{run}\n报告：{report}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
