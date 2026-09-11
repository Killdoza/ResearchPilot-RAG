# ruff: noqa: E501  # Long Markdown report prose is intentionally kept readable.
"""生成 E4 固定分块与 E5 结构感知分块的受控对照报告。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _read_summary(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"聚合 CSV 必须恰有一行：{path}")
    return rows[0]


def _latest_run(runtime_dir: Path, experiment: str, split: str) -> Path:
    matches = sorted(runtime_dir.glob(f"{experiment}_{split}_*.jsonl"))
    if not matches:
        raise ValueError(f"未找到 {experiment} 的 {split} 逐题记录。")
    return matches[-1]


def _read_records(path: Path) -> dict[str, dict[str, object]]:
    return {
        str(record["case_id"]): record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    }


def _is_hit(record: dict[str, object]) -> bool:
    evaluation = record["evaluation"]
    assert isinstance(evaluation, dict)
    return bool(evaluation["relevant_ranks"])


def _type_metrics(records: dict[str, dict[str, object]]) -> dict[str, tuple[int, float, float]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for record in records.values():
        evaluation = record["evaluation"]
        assert isinstance(evaluation, dict)
        if not evaluation["answerable"]:
            continue
        grouped.setdefault(str(record["task_type"]), []).append(evaluation)
    metrics: dict[str, tuple[int, float, float]] = {}
    for task_type, items in grouped.items():
        recall_at_5 = sum(bool(item["relevant_ranks"] and item["relevant_ranks"][0] <= 5) for item in items) / len(
            items
        )
        mrr = sum(1 / item["relevant_ranks"][0] if item["relevant_ranks"] else 0 for item in items) / len(items)
        metrics[task_type] = (len(items), recall_at_5, mrr)
    return metrics


def _latest_e5_ingestion(runtime_dir: Path) -> dict[str, object] | None:
    candidates = []
    for path in runtime_dir.glob("ingestion_*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if record.get("experiment") == "e5_structure_aware":
            candidates.append((path.name, record))
    return sorted(candidates, key=lambda item: item[0])[-1][1] if candidates else None


def build_report(
    *,
    e4_summary: dict[str, str],
    e5_summary: dict[str, str],
    e4_records: dict[str, dict[str, object]],
    e5_records: dict[str, dict[str, object]],
    e5_ingestion: dict[str, object] | None,
    e4_run: Path,
    e5_run: Path,
    e4_test_summary: dict[str, str],
    e5_test_summary: dict[str, str],
    e4_test_run: Path,
    e5_test_run: Path,
) -> str:
    metric_rows = [
        ("MRR", "mrr"),
        ("Recall@5", "question_recall_at_5"),
        ("nDCG@5", "ndcg_at_5"),
        ("P95 检索耗时（ms）", "p95_retrieval_latency_ms"),
    ]
    table = ["| 指标 | E4 固定分块 | E5 结构感知 | 变化 |", "|---|---:|---:|---:|"]
    for label, key in metric_rows:
        left, right = float(e4_summary[key]), float(e5_summary[key])
        table.append(f"| {label} | {left:.4f} | {right:.4f} | {right - left:+.4f} |")

    test_table = ["| 指标 | E4 固定分块 | E5 结构感知 | 变化 |", "|---|---:|---:|---:|"]
    for label, key in metric_rows:
        left, right = float(e4_test_summary[key]), float(e5_test_summary[key])
        test_table.append(f"| {label} | {left:.4f} | {right:.4f} | {right - left:+.4f} |")

    e4_types, e5_types = _type_metrics(e4_records), _type_metrics(e5_records)
    type_table = [
        "| 类型 | 题数 | E4 Recall@5 | E5 Recall@5 | E4 MRR | E5 MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task_type in sorted(set(e4_types) | set(e5_types)):
        count, e4_recall, e4_mrr = e4_types.get(task_type, (0, 0.0, 0.0))
        _, e5_recall, e5_mrr = e5_types.get(task_type, (0, 0.0, 0.0))
        type_table.append(
            f"| {task_type} | {count} | {e4_recall:.4f} | {e5_recall:.4f} | {e4_mrr:.4f} | {e5_mrr:.4f} |"
        )

    recovered = [
        record
        for case_id, record in e5_records.items()
        if case_id in e4_records and not _is_hit(e4_records[case_id]) and _is_hit(record)
    ]
    regressed = [
        record
        for case_id, record in e4_records.items()
        if case_id in e5_records and _is_hit(record) and not _is_hit(e5_records[case_id])
    ]
    change_table = ["| 题目 ID | 类型 | 问题 |", "|---|---|---|"]
    for record in recovered[:20]:
        change_table.append(
            f"| {record['case_id']} | {record['task_type']} | {str(record['question']).replace('|', '\\|')} |"
        )
    if len(change_table) == 2:
        change_table.append("| 无 | - | E5 未新增找回 E4 完全漏召回的题目 |")
    regression_table = ["| 题目 ID | 类型 | 问题 |", "|---|---|---|"]
    for record in regressed[:20]:
        regression_table.append(
            f"| {record['case_id']} | {record['task_type']} | {str(record['question']).replace('|', '\\|')} |"
        )
    if len(regression_table) == 2:
        regression_table.append("| 无 | - | E5 未产生 E4 命中而 E5 完全漏召回的题目 |")

    parsing_section = ["未找到 E5 导入 Manifest。"]
    if e5_ingestion is not None:
        papers = e5_ingestion.get("papers", [])
        block_count = sum(int(paper["block_count"]) for paper in papers)
        parent_count = sum(int(paper["parent_chunk_count"]) for paper in papers)
        child_count = sum(int(paper["child_chunk_count"]) for paper in papers)
        empty_pages = sum(int(paper["parse_quality"]["empty_page_count"]) for paper in papers)
        parsing_section = [
            f"- Parser：`{e5_ingestion['parser_name']} {e5_ingestion['parser_version']}`；OCR：`{e5_ingestion['enable_ocr']}`。",
            f"- 10 篇论文共生成 {block_count} 个 Block、{parent_count} 个 Parent Chunk、{child_count} 个 Child Chunk；空页数为 {empty_pages}。",
            "- Block JSON 与导入 Manifest 均在 Git 忽略的 `data/runtime/` 中，原始 PDF 未被复制或提交。",
        ]

    return (
        "\n".join(
            [
                "# P1-D：结构感知解析与 E5 对照报告",
                "",
                "E4 与 E5 使用同一份 AI/CS 论文、同一页级评测集、Embedding、Dense/BM25 候选数、RRF、BGE Reranker 与 Top-K；唯一实验变量是 Parser 与 Chunker。所有配置选择只基于 Dev。",
                "",
                "## 解析与索引质量",
                "",
                *parsing_section,
                "",
                "## Dev 总体结果",
                "",
                *table,
                "",
                "## 按问题类型的 Dev 结果",
                "",
                *type_table,
                "",
                "## E5 新增找回的 E4 完全漏召回案例",
                "",
                *change_table,
                "",
                "## E5 新增完全漏召回案例",
                "",
                *regression_table,
                "",
                "## 一次性留出 Test 结果（配置已冻结）",
                "",
                "E5 配置已在页面级人工复核和 Dev 结果后，以提交 `099b406` 冻结；以下 24 条 Test 只运行一次，结果不再用于调节 Parser、Chunker 或检索参数。E4 Test 为此前已冻结的固定分块对照，两个 Test 均含 20 条可回答题和 4 条不可回答题。",
                "",
                *test_table,
                "",
                f"- E5 Test 的不可回答题 Top-1 仍有召回比例为 `{float(e5_test_summary['unanswerable_top1_retrieval_rate']):.4f}`。这不是拒答准确率：当前评测尚未校准拒答阈值，后续需单独开展拒答/置信度实验。",
                "- Test 论文按论文级留出，但问题由项目作者预先编写，因此它不是完全盲测；从本次运行起不再依据 Test 指标改动 E5。",
                "- 页面级视觉复核见 `docs/p1d_page_audit.md`：本次修复了公式 Block 漏失；图像内容、复杂双栏顺序和公式无损转写仍是已记录的边界。",
                "",
                "## 当前决策",
                "",
                "- E5 已完成页面级人工复核并冻结，作为当前高精度结构感知检索基线；P1-D 不再继续使用 Test 调参。",
                "- 交互产品可继续以 E3 为低延迟检索策略，但其底层 Parser/Chunker 可在 E5 验证后复用结构感知 Block；E5 的 Reranker 链路作为高精度对照。",
                "",
                f"Dev 逐题记录：E4 `{e4_run.relative_to(PROJECT_ROOT).as_posix()}`；E5 `{e5_run.relative_to(PROJECT_ROOT).as_posix()}`。",
                f"Test 逐题记录：E4 `{e4_test_run.relative_to(PROJECT_ROOT).as_posix()}`；E5 `{e5_test_run.relative_to(PROJECT_ROOT).as_posix()}`。",
            ]
        )
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="生成 P1-D E4/E5 对照报告")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "eval/results")
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "data/runtime/evaluations")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/reports/p1d_parser_chunking_report.md")
    args = parser.parse_args(argv)
    try:
        e4_summary = _read_summary(args.results_dir / "e4_dense_bm25_rrf_rerank_dev.csv")
        e5_summary = _read_summary(args.results_dir / "e5_structure_aware_dev.csv")
        e4_test_summary = _read_summary(args.results_dir / "e4_dense_bm25_rrf_rerank_test.csv")
        e5_test_summary = _read_summary(args.results_dir / "e5_structure_aware_test.csv")
        e4_run = _latest_run(args.runtime_dir, "e4_dense_bm25_rrf_rerank", "dev")
        e5_run = _latest_run(args.runtime_dir, "e5_structure_aware", "dev")
        e4_test_run = _latest_run(args.runtime_dir, "e4_dense_bm25_rrf_rerank", "test")
        e5_test_run = _latest_run(args.runtime_dir, "e5_structure_aware", "test")
        content = build_report(
            e4_summary=e4_summary,
            e5_summary=e5_summary,
            e4_records=_read_records(e4_run),
            e5_records=_read_records(e5_run),
            e5_ingestion=_latest_e5_ingestion(args.runtime_dir),
            e4_run=e4_run,
            e5_run=e5_run,
            e4_test_summary=e4_test_summary,
            e5_test_summary=e5_test_summary,
            e4_test_run=e4_test_run,
            e5_test_run=e5_test_run,
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"P1-D 报告生成失败：{error}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8")
    print(f"已生成 P1-D 报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
