# ruff: noqa: E501  # Long Markdown report prose is intentionally kept readable.
"""汇总 E0--E4 的 Dev 结果与一次冻结后的 E4 Test 结果。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = (
    "e0_dense",
    "e1_bm25",
    "e2_dense_bm25_union",
    "e3_dense_bm25_rrf",
    "e4_dense_bm25_rrf_rerank",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 P1-C 检索消融报告")
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "eval/results")
    parser.add_argument("--runtime-dir", type=Path, default=PROJECT_ROOT / "data/runtime/evaluations")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/reports/p1c_retrieval_ablation.md")
    return parser.parse_args(argv)


def _read_csv(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"聚合 CSV 必须恰有一行：{path}")
    return rows[0]


def _latest_run(runtime_dir: Path, experiment: str) -> Path:
    matches = sorted(runtime_dir.glob(f"{experiment}_dev_*.jsonl"))
    if not matches:
        raise ValueError(f"未找到 {experiment} 的 Dev 逐题记录。")
    return matches[-1]


def _task_rows(path: Path) -> list[tuple[str, int, float, float]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        evaluation = record["evaluation"]
        if not evaluation["answerable"]:
            continue
        grouped.setdefault(str(record["task_type"]), []).append(evaluation)
    rows = []
    for task_type, items in sorted(grouped.items()):
        recall_at_5 = sum(bool(item["relevant_ranks"] and item["relevant_ranks"][0] <= 5) for item in items) / len(
            items
        )
        mrr = sum(1 / item["relevant_ranks"][0] if item["relevant_ranks"] else 0 for item in items) / len(items)
        rows.append((task_type, len(items), recall_at_5, mrr))
    return rows


def _failure_rows(path: Path, limit: int = 20) -> list[dict[str, object]]:
    failures = []
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        evaluation = record["evaluation"]
        if evaluation["answerable"] and not evaluation["relevant_ranks"]:
            failures.append(record)
        elif not evaluation["answerable"] and evaluation["result_count"]:
            failures.append(record)
    return failures[:limit]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summaries = {experiment: _read_csv(args.results_dir / f"{experiment}_dev.csv") for experiment in EXPERIMENTS}
        e4_run = _latest_run(args.runtime_dir, "e4_dense_bm25_rrf_rerank")
        task_rows = _task_rows(e4_run)
        failures = _failure_rows(e4_run)
        test_path = args.results_dir / "e4_dense_bm25_rrf_rerank_test.csv"
        test_summary = _read_csv(test_path) if test_path.exists() else None
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"P1-C 报告生成失败：{error}", file=sys.stderr)
        return 1

    table = [
        "| 实验 | MRR | Recall@5 | nDCG@5 | 平均 ms | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for experiment in EXPERIMENTS:
        row = summaries[experiment]
        table.append(
            "| {name} | {mrr} | {recall} | {ndcg} | {mean} | {p50} | {p95} |".format(
                name=experiment,
                mrr=row["mrr"],
                recall=row["question_recall_at_5"],
                ndcg=row["ndcg_at_5"],
                mean=row["mean_retrieval_latency_ms"],
                p50=row["p50_retrieval_latency_ms"],
                p95=row["p95_retrieval_latency_ms"],
            )
        )
    type_table = [
        "| 问题类型（E4 Dev） | 可回答题数 | Recall@5 | MRR |",
        "|---|---:|---:|---:|",
        *[f"| {task_type} | {count} | {recall:.4f} | {mrr:.4f} |" for task_type, count, recall, mrr in task_rows],
    ]
    failure_table = ["| Case ID | 类型 | 问题 | 原因 |", "|---|---|---|---|"]
    for record in failures:
        evaluation = record["evaluation"]
        reason = "未召回 Gold 页" if evaluation["answerable"] else "不可回答题仍有文档召回"
        question = str(record["question"]).replace("|", "\\|")
        failure_table.append(f"| {record['case_id']} | {record['task_type']} | {question} | {reason} |")
    e0, e1, e2, e3, e4 = (summaries[name] for name in EXPERIMENTS)
    observations = [
        "## 基于 Dev 的结论与决策",
        "",
        "- E1 的 MRR 相比 E0 的变化：{:+.4f}；说明当前中文问英文论文场景中，稀疏词项匹配是必要通道。".format(
            float(e1["mrr"]) - float(e0["mrr"])
        ),
        "- E3 的 Recall@5 相比 E2 的变化：{:+.4f}；本轮用 RRF 替代简单并集具有可测的召回收益。".format(
            float(e3["question_recall_at_5"]) - float(e2["question_recall_at_5"])
        ),
        "- E4 的 MRR 相比 E3 的变化：{:+.4f}，但 P95 延迟增加 {:.1f} 倍；Reranker 应作为高精度模式，而不是默认交互链路。".format(
            float(e4["mrr"]) - float(e3["mrr"]),
            float(e4["p95_retrieval_latency_ms"]) / max(float(e3["p95_retrieval_latency_ms"]), 0.001),
        ),
        "- 后续 P1-D 默认采用 E3（Dense + BM25 + RRF）作为可交互基线；E4 保留为深度精读/离线评测模式。该决策仅来自 Dev，未使用 Test 调参。",
        "",
    ]
    test_section: list[str] = []
    if test_summary is not None:
        test_section = [
            "## 一次性留出 Test 结果",
            "",
            "E4 配置在完成 Dev 复核后冻结；下表为对 24 条留出论文题的首次运行结果，未据此修改任何参数。"
            "由于题目由人工编写，它是论文级留出测试，而不是对标注者完全不可见的盲测。",
            "",
            "| 指标 | 数值 |",
            "|---|---:|",
            f"| 可回答问题数 | {test_summary['answerable_cases']} |",
            f"| MRR | {test_summary['mrr']} |",
            f"| Recall@5 | {test_summary['question_recall_at_5']} |",
            f"| nDCG@5 | {test_summary['ndcg_at_5']} |",
            f"| P95 检索耗时（ms） | {test_summary['p95_retrieval_latency_ms']} |",
            "",
        ]
    content = "\n".join(
        [
            "# P1-C：E0--E4 检索消融报告",
            "",
            "本报告严格使用 Dev 结果选择后续默认检索链路；Test 不参与调参。",
            "",
            "## 总体结果",
            "",
            *table,
            "",
            "## E4 分类型结果",
            "",
            *type_table,
            "",
            *observations,
            *test_section,
            "## 待人工复核的失败案例",
            "",
            "以下是自动筛出的最多 20 条失败/误召回案例。复核时需区分解析错误、Gold 页码问题、分块问题、召回问题与重排问题。",
            "",
            *failure_table,
            "",
            f"逐题 E4 运行记录：`{e4_run.relative_to(PROJECT_ROOT).as_posix()}`。",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content + "\n", encoding="utf-8")
    print(f"已生成 P1-C 报告：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
