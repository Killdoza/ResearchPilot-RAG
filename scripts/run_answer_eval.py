"""运行 P1-E Long-context / RAG 回答对照；默认 --dry-run，避免误耗 API。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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

from researchpilot.answer_evaluation import (  # noqa: E402
    AnswerCitation,
    GeneratedAnswer,
    evaluate_generated_answer,
    render_answer_summary_markdown,
    summarize_answer_evaluations,
)
from researchpilot.answer_runner import (  # noqa: E402
    LangChainStructuredAnswerClient,
    build_context_for_case,
    estimate_prompt_tokens,
    generate_structured_answer,
    load_answer_experiment_config,
)
from researchpilot.config import (  # noqa: E402
    build_settings,
    require_deepseek_api_key,
    validate_storage_id,
)
from researchpilot.context_modes import E5BlockRepository  # noqa: E402
from researchpilot.crossdoc import (  # noqa: E402
    DEFAULT_PAPER_SPLITS,
    load_crossdoc_cases,
    validate_crossdoc_cases,
    validate_frozen_crossdoc_manifest,
)
from researchpilot.evaluation import (  # noqa: E402
    EvaluationDataError,
    load_corpus_manifest,
    load_eval_cases,
    validate_dataset_manifest_coverage,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 P1-E 回答与引用评测。")
    parser.add_argument("--config", type=Path, required=True, help="LC1/LC2/N0/A0 实验 YAML。")
    parser.add_argument("--dataset-kind", choices=("single", "crossdoc"), required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--single-dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_eval_v0.1.jsonl")
    parser.add_argument(
        "--crossdoc-dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_crossdoc_v0.1.jsonl"
    )
    parser.add_argument(
        "--crossdoc-manifest",
        type=Path,
        default=PROJECT_ROOT / "eval/manifests/ai_cs_crossdoc_v0.1.manifest.json",
    )
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json")
    parser.add_argument(
        "--case-ids",
        help="逗号分隔的固定题目 ID 子集；未传时运行该 split 全部题目。",
    )
    parser.add_argument(
        "--case-id-file",
        type=Path,
        help="冻结的题目选择 Manifest；会校验数据集 SHA-256 与 split，不能与 --case-ids 同用。",
    )
    parser.add_argument("--user-id", help="可选覆盖 YAML 中冻结的 E5 评测 user_id。")
    parser.add_argument("--session-id", help="可选覆盖 YAML 中冻结的 E5 评测 session_id。")
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "data/runtime/answer_evaluations")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "eval/results")
    parser.add_argument("--run", action="store_true", help="实际调用 DeepSeek；默认只输出 Token/调用次数 dry-run。")
    parser.add_argument("--overwrite", action="store_true", help="允许覆盖同名聚合 Markdown 报告。")
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="从失败时保留的 in_progress JSONL 继续，只跳过已成功写入的题目。",
    )
    return parser.parse_args(argv)


def _select_cases(cases: Sequence[Any], *, split: str, case_ids: str | None) -> tuple[Any, ...]:
    selected = tuple(case for case in cases if case.split == split)
    if case_ids:
        wanted = tuple(item.strip() for item in case_ids.split(",") if item.strip())
        if len(set(wanted)) != len(wanted):
            raise EvaluationDataError("--case-ids 包含重复 ID。")
        selected_by_id = {case.case_id: case for case in selected}
        missing = [case_id for case_id in wanted if case_id not in selected_by_id]
        if missing:
            raise EvaluationDataError(f"指定题目不在 split={split} 中：{missing}")
        selected = tuple(selected_by_id[case_id] for case_id in wanted)
    if not selected:
        raise EvaluationDataError("选中的评测问题为空。")
    return selected


def _case_ids_from_file(*, path: Path, dataset_path: Path, split: str) -> str:
    if not path.is_file():
        raise EvaluationDataError(f"题目选择 Manifest 不存在：{path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise EvaluationDataError(f"题目选择 Manifest 不是合法 JSON：{error}") from error
    if not isinstance(raw, dict) or raw.get("schema_version") != "researchpilot_p1e_case_selection_v0.1":
        raise EvaluationDataError("题目选择 Manifest schema_version 不正确。")
    if raw.get("split") != split:
        raise EvaluationDataError("题目选择 Manifest 的 split 与命令行不一致。")
    if raw.get("dataset") != dataset_path.name:
        raise EvaluationDataError("题目选择 Manifest 指向了错误的数据集文件。")
    actual_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    if raw.get("dataset_sha256") != actual_hash:
        raise EvaluationDataError("题目选择 Manifest 的数据集哈希不一致；拒绝运行。")
    ids = raw.get("case_ids")
    if not isinstance(ids, list) or not ids or any(not isinstance(item, str) or not item.strip() for item in ids):
        raise EvaluationDataError("题目选择 Manifest 的 case_ids 必须是非空字符串数组。")
    return ",".join(ids)


def _load_cases(args: argparse.Namespace, corpus: Any) -> tuple[Any, ...]:
    if args.dataset_kind == "single":
        cases = load_eval_cases(args.single_dataset)
        validate_dataset_manifest_coverage(cases, corpus)
        return _select_cases(cases, split=args.split, case_ids=args.case_ids)
    frozen = validate_frozen_crossdoc_manifest(dataset_path=args.crossdoc_dataset, manifest_path=args.crossdoc_manifest)
    page_counts = frozen.get("pdf_page_counts")
    if not isinstance(page_counts, dict):
        raise EvaluationDataError("跨论文 Manifest 缺少 pdf_page_counts。")
    cases = load_crossdoc_cases(args.crossdoc_dataset)
    validate_crossdoc_cases(
        cases=cases,
        corpus=corpus,
        paper_splits=DEFAULT_PAPER_SPLITS,
        page_counts=page_counts,
    )
    return _select_cases(cases, split=args.split, case_ids=args.case_ids)


def _build_retrieval_service(config: Any):
    """复用已验证的检索评测组装逻辑；只在 N0/A0 需要它。"""
    from run_retrieval_eval import build_retrieval_service

    return build_retrieval_service(config=config)


def _context_record(*, case: Any, context: Any, prompt_tokens: int, config_sha256: str) -> dict[str, Any]:
    return {
        "schema_version": "p1e_answer_dry_run_v1",
        "run_id": f"dry_{uuid.uuid4().hex}",
        "case_id": case.case_id,
        "split": case.split,
        "task_type": case.task_type,
        "answerable": case.answerable,
        "question": case.question,
        "context_mode": context.mode,
        "selection_note": context.selection_note,
        "context_reference_count": len(context.references),
        "context_characters": context.character_count,
        "estimated_prompt_tokens": prompt_tokens,
        "max_input_tokens": None,  # 后续由调用侧填充，保持记录字段稳定。
        "model_calls": 1,
        "retrieval_latency_ms": context.retrieval_latency_ms,
        "retrieval_trace_id": context.retrieval_trace_id,
        "experiment_config_sha256": config_sha256,
        "within_context_limit": None,
        "references": [
            {
                "context_id": item.reference_id,
                "paper_key": item.paper_key,
                "page_start": item.page_start,
                "page_end": item.page_end,
                "block_id": item.block_id,
                "source_kind": item.source_kind,
            }
            for item in context.references
        ],
    }


def _write_jsonl(*, directory: Path, stem: str, records: Sequence[dict[str, Any]]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = directory / f"{stem}_{timestamp}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return path


def _build_client() -> LangChainStructuredAnswerClient:
    from langchain_openai import ChatOpenAI

    settings = build_settings()
    chat_model = ChatOpenAI(
        api_key=require_deepseek_api_key(settings),
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        streaming=False,
    )
    return LangChainStructuredAnswerClient(chat_model)


def _write_summary(*, path: Path, content: str, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise EvaluationDataError(f"结果报告已存在：{path}；如要覆盖请显式传入 --overwrite。")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _load_resume_records(*, path: Path, selected_cases: Sequence[Any], config_sha256: str) -> dict[str, dict[str, Any]]:
    """读取逐题检查点；拒绝混入另一份配置或另一组题目。"""
    if not path.is_file():
        raise EvaluationDataError(f"--resume-from 文件不存在：{path}")
    selected_ids = {case.case_id for case in selected_cases}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationDataError(f"检查点第 {line_number} 行不是合法 JSON。") from error
        if not isinstance(record, dict) or record.get("schema_version") != "p1e_answer_run_v1":
            raise EvaluationDataError(f"检查点第 {line_number} 行 schema 不正确。")
        case_id = record.get("case_id")
        if not isinstance(case_id, str) or case_id not in selected_ids:
            raise EvaluationDataError(f"检查点第 {line_number} 行包含当前运行外题目。")
        if record.get("experiment_config_sha256") != config_sha256:
            raise EvaluationDataError("检查点与当前实验配置哈希不同，拒绝混合结果。")
        if case_id in records:
            raise EvaluationDataError(f"检查点包含重复题目：{case_id}")
        records[case_id] = record
    return records


def _answer_from_record(record: dict[str, Any]) -> GeneratedAnswer:
    raw = record.get("answer")
    if not isinstance(raw, dict):
        raise EvaluationDataError("检查点缺少 answer 对象。")
    citations_raw = raw.get("citations")
    if not isinstance(citations_raw, list):
        raise EvaluationDataError("检查点 answer.citations 必须是数组。")
    try:
        citations = tuple(
            AnswerCitation(
                paper_key=item["paper_key"],
                page_start=item["page_start"],
                page_end=item["page_end"],
                context_id=item.get("context_id"),
            )
            for item in citations_raw
            if isinstance(item, dict)
        )
        if len(citations) != len(citations_raw):
            raise ValueError("citation 不是对象")
        return GeneratedAnswer(
            answer=raw["answer"],
            citations=citations,
            refused=raw["refused"],
            refusal_reason=raw.get("refusal_reason"),
            prompt_tokens=raw.get("prompt_tokens"),
            completion_tokens=raw.get("completion_tokens"),
            cost=raw.get("cost"),
            retrieval_latency_ms=raw["retrieval_latency_ms"],
            generation_latency_ms=raw["generation_latency_ms"],
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationDataError("检查点 answer 字段非法。") from error


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.case_ids and args.case_id_file:
            raise EvaluationDataError("--case-ids 与 --case-id-file 不能同时使用。")
        corpus = load_corpus_manifest(args.manifest)
        config = load_answer_experiment_config(args.config, project_root=PROJECT_ROOT)
        args.user_id = args.user_id or config.evaluation_user_id
        args.session_id = args.session_id or config.evaluation_session_id
        validate_storage_id(args.user_id, "user_id")
        validate_storage_id(args.session_id, "session_id")
        if args.case_id_file:
            if args.dataset_kind != "single":
                raise EvaluationDataError("--case-id-file 当前只用于单论文评测集。")
            args.case_ids = _case_ids_from_file(
                path=args.case_id_file, dataset_path=args.single_dataset, split=args.split
            )
        cases = _load_cases(args, corpus)
        if args.dataset_kind == "crossdoc" and config.context_mode == "lc1":
            raise EvaluationDataError("跨论文题不得用 LC1；请使用 LC2、N0 或 A0。")
        if args.dataset_kind == "single" and config.context_mode == "lc2":
            raise EvaluationDataError("单论文题不得用 LC2；请使用 LC1、N0 或 A0。")
        repository = E5BlockRepository(
            artifact_dir=config.artifact_dir,
            corpus=corpus,
            expected_parser_name=config.parser_name,
            expected_parser_version=config.parser_version,
        )
        retrieval_service = None
        if config.context_mode in {"n0", "a0"}:
            retrieval_service, _ = _build_retrieval_service(config.retrieval_experiment)
        config_sha256 = hashlib.sha256(args.config.read_bytes()).hexdigest()
    except Exception as error:
        print(f"P1-E 启动失败：{error}", file=sys.stderr)
        return 2

    contexts: list[tuple[Any, Any, int]] = []
    try:
        for case in cases:
            context = build_context_for_case(
                case=case,
                config=config,
                corpus=corpus,
                block_repository=repository,
                retrieval_service=retrieval_service,
                user_id=args.user_id,
                session_id=args.session_id,
            )
            contexts.append((case, context, estimate_prompt_tokens(question=case.question, context=context)))
    except Exception as error:
        print(f"P1-E 上下文构建失败：{error}", file=sys.stderr)
        return 1

    if not args.run:
        records: list[dict[str, Any]] = []
        for case, context, estimate in contexts:
            record = _context_record(case=case, context=context, prompt_tokens=estimate, config_sha256=config_sha256)
            record["max_input_tokens"] = config.max_input_tokens
            record["within_context_limit"] = estimate <= config.max_input_tokens
            records.append(record)
        path = _write_jsonl(
            directory=args.runtime_dir,
            stem=f"dry_run_{config.retrieval_experiment.name}_{args.dataset_kind}_{args.split}",
            records=records,
        )
        risky = [record["case_id"] for record in records if not record["within_context_limit"]]
        print(
            f"P1-E dry-run 完成：{config.retrieval_experiment.name} / {args.dataset_kind} / "
            f"{args.split} / {len(records)} 题；记录：{path}"
        )
        if risky:
            print(f"以下题目预计超过上下文上限，实际运行会拒绝自动截断：{', '.join(risky)}")
            return 3
        return 0

    if any(estimate > config.max_input_tokens for _, _, estimate in contexts):
        print("拒绝实际运行：dry-run 已发现上下文超限；请先在 Dev 固定解决方案。", file=sys.stderr)
        return 2
    try:
        client = _build_client()
        resumed = (
            _load_resume_records(path=args.resume_from, selected_cases=cases, config_sha256=config_sha256)
            if args.resume_from
            else {}
        )
        records = []
        answers: list[GeneratedAnswer] = []
        outcomes = []
        args.runtime_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        final_runtime_path = args.runtime_dir / (
            f"run_{config.retrieval_experiment.name}_{args.dataset_kind}_{args.split}_{timestamp}.jsonl"
        )
        checkpoint_path = args.runtime_dir / (
            f"in_progress_{config.retrieval_experiment.name}_{args.dataset_kind}_{args.split}_{timestamp}.jsonl"
        )
        with checkpoint_path.open("w", encoding="utf-8", newline="\n") as handle:
            for case, context, _ in contexts:
                existing = resumed.get(case.case_id)
                answer = (
                    _answer_from_record(existing)
                    if existing is not None
                    else generate_structured_answer(
                        question=case.question, context=context, config=config, client=client
                    )
                )
                outcome = evaluate_generated_answer(case=case, answer=answer)
                answers.append(answer)
                outcomes.append(outcome)
                record = existing or {
                    "schema_version": "p1e_answer_run_v1",
                    "run_id": f"answer_{uuid.uuid4().hex}",
                    "created_at": datetime.now(UTC).isoformat(),
                    "case_id": case.case_id,
                    "split": case.split,
                    "task_type": case.task_type,
                    "answerable": case.answerable,
                    "question": case.question,
                    "context_mode": context.mode,
                    "context_reference_count": len(context.references),
                    "context_characters": context.character_count,
                    "retrieval_trace_id": context.retrieval_trace_id,
                    "answer": answer.to_dict(),
                    "evaluation": outcome.to_dict(),
                    "experiment_config_sha256": config_sha256,
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        checkpoint_path.replace(final_runtime_path)
        runtime_path = final_runtime_path
        summary = summarize_answer_evaluations(outcomes=outcomes, answers=answers)
        summary_path = (
            args.results_dir / f"{config.retrieval_experiment.name}_{args.dataset_kind}_{args.split}_answers.md"
        )
        _write_summary(
            path=summary_path,
            content=render_answer_summary_markdown(
                experiment_name=config.retrieval_experiment.name,
                split=f"{args.dataset_kind}_{args.split}",
                summary=summary,
            ),
            overwrite=args.overwrite,
        )
    except Exception as error:
        if "checkpoint_path" in locals() and checkpoint_path.exists():
            print(f"已保留可诊断的逐题检查点：{checkpoint_path}", file=sys.stderr)
        print(f"P1-E 实际运行失败：{error}", file=sys.stderr)
        return 1
    print(f"P1-E 实际运行完成：{len(records)} 题。")
    print(f"逐题 JSONL：{runtime_path}")
    print(f"聚合报告：{summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
