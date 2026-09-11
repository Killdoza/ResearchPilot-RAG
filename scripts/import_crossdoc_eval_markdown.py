"""将人工维护的跨论文 Markdown 标注导入冻结 JSONL。

该脚本只支持标注指南约定的受控模板；标注格式有误时宁可明确失败，也不会
静默丢掉结论或证据。公开仓库提供冻结 JSONL，个人 Markdown 工作稿通过
``--input`` 显式传入；PDF 始终留在 ``data/raw/`` 的忽略目录。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.crossdoc import (  # noqa: E402
    CROSSDOC_SCHEMA_VERSION,
    DEFAULT_PAPER_SPLITS,
    CrossDocumentCase,
    build_crossdoc_manifest,
    validate_crossdoc_cases,
)
from researchpilot.evaluation import EvaluationDataError, load_corpus_manifest  # noqa: E402

_CASE_HEADING = re.compile(r"(?m)^##\s+(CD\d+)\s*$")
_SUBHEADING = re.compile(r"(?m)^###\s+(.+?)\s*$")
_SCALAR_FIELD = re.compile(r"(?m)^\*\s+(split|task_type|answerable|comparison_dimension|question):\s*(.*?)\s*$")
_PAPER_KEY = re.compile(r"(?m)^\s*\*\s+([A-Za-z][A-Za-z0-9_.-]*)\s*$")
_CLAIM = re.compile(r"(?m)^\s*\*\s*(C\d+)：\s*(.+?)\s*$")
_EVIDENCE = re.compile(r"(?ms)^\s*\*\s*evidence_id:\s*(E\d+)\s*$\n(.*?)(?=^\s*\*\s*evidence_id:|^###\s|\Z)")
_EVIDENCE_FIELD = re.compile(
    r"(?m)^\s*\*\s*(claim_ids|paper_key|page_start|page_end|section_hint|supporting_text_summary):\s*(.*?)\s*$"
)


def _section(content: str, heading: str) -> str:
    match = re.search(rf"(?ms)^###\s+{re.escape(heading)}\s*$\n(.*?)(?=^###\s|\Z)", content)
    if not match:
        raise EvaluationDataError(f"缺少 ### {heading} 小节。")
    return match.group(1).strip()


def _scalar_fields(content: str, case_id: str) -> dict[str, str]:
    fields = {name: value.strip() for name, value in _SCALAR_FIELD.findall(content)}
    required = {"split", "task_type", "answerable", "question"}
    missing = sorted(required - set(fields))
    if missing:
        raise EvaluationDataError(f"{case_id} 缺少字段：{', '.join(missing)}")
    return fields


def _parse_paper_keys(content: str, case_id: str) -> list[str]:
    match = re.search(r"(?ms)^\*\s+paper_keys:\s*$\n(.*?)(?=^###\s|\Z)", content)
    if not match:
        raise EvaluationDataError(f"{case_id} 缺少 paper_keys。")
    keys = _PAPER_KEY.findall(match.group(1))
    if len(keys) < 2:
        raise EvaluationDataError(f"{case_id} 必须列出至少两篇论文。")
    return keys


def _plain_expected_answer(section: str) -> str:
    lines = [re.sub(r"^\s*\*\s*", "", line).strip() for line in section.splitlines()]
    answer = "\n".join(line for line in lines if line)
    if not answer:
        raise EvaluationDataError("不可回答题的预期拒答说明不能为空。")
    return answer


def _parse_evidence(section: str, case_id: str) -> list[dict[str, Any]]:
    if "无需伪造不存在的证据" in section:
        return []
    evidence: list[dict[str, Any]] = []
    for evidence_id, body in _EVIDENCE.findall(section):
        fields = {name: value.strip() for name, value in _EVIDENCE_FIELD.findall(body)}
        required = {
            "claim_ids",
            "paper_key",
            "page_start",
            "page_end",
            "section_hint",
            "supporting_text_summary",
        }
        missing = sorted(required - set(fields))
        if missing:
            raise EvaluationDataError(f"{case_id}/{evidence_id} 缺少字段：{', '.join(missing)}")
        claim_match = re.fullmatch(r"\[\s*(C\d+(?:\s*,\s*C\d+)*)\s*\]", fields["claim_ids"])
        if not claim_match:
            raise EvaluationDataError(f"{case_id}/{evidence_id} 的 claim_ids 必须形如 [C1, C2]。")
        try:
            page_start = int(fields["page_start"])
            page_end = int(fields["page_end"])
        except ValueError as error:
            raise EvaluationDataError(f"{case_id}/{evidence_id} 页码必须是整数。") from error
        evidence.append(
            {
                "evidence_id": evidence_id,
                "claim_ids": [part.strip() for part in claim_match.group(1).split(",")],
                "paper_key": fields["paper_key"],
                "page_start": page_start,
                "page_end": page_end,
                "section_hint": fields["section_hint"],
                "supporting_text_summary": fields["supporting_text_summary"],
            }
        )
    if not evidence:
        raise EvaluationDataError(f"{case_id} 未解析到任何 evidence_id。")
    return evidence


def _parse_notes(content: str) -> list[str]:
    match = re.search(r"(?ms)^###\s+标注备注\s*$\n(.*?)(?=^###\s|\Z)", content)
    if not match:
        # 标注备注是建议字段；缺失不影响已有证据的可复现性。
        return []
    section = match.group(1).strip()
    return [line.strip()[2:].strip() for line in section.splitlines() if line.strip().startswith("* ")]


def parse_crossdoc_markdown(markdown: str) -> tuple[CrossDocumentCase, ...]:
    """解析受控 Markdown 模板，并立即套用 JSONL 数据契约。"""
    headings = list(_CASE_HEADING.finditer(markdown))
    if not headings:
        raise EvaluationDataError("未找到形如 ## CD001 的跨论文题目标题。")
    cases: list[CrossDocumentCase] = []
    for index, heading in enumerate(headings):
        case_id = heading.group(1)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        content = markdown[heading.end() : end]
        fields = _scalar_fields(content, case_id)
        answerable = fields["answerable"].lower()
        if answerable not in {"true", "false"}:
            raise EvaluationDataError(f"{case_id} 的 answerable 必须为 true 或 false。")
        expected_section = _section(content, "预期答案")
        is_answerable = answerable == "true"
        claim_matches = _CLAIM.findall(expected_section)
        expected_claims = [{"claim_id": claim_id, "text": text.strip()} for claim_id, text in claim_matches]
        if is_answerable and not expected_claims:
            raise EvaluationDataError(f"{case_id} 可回答题必须至少有一个 C1/C2… 原子结论。")
        record: dict[str, Any] = {
            "schema_version": CROSSDOC_SCHEMA_VERSION,
            "id": case_id,
            "source_id": case_id,
            "split": fields["split"],
            "task_type": fields["task_type"],
            "answerable": is_answerable,
            "paper_keys": _parse_paper_keys(content, case_id),
            "comparison_dimension": fields.get("comparison_dimension"),
            "question": fields["question"],
            "expected_answer": (
                "\n".join(f"{claim_id}：{text}" for claim_id, text in claim_matches)
                if is_answerable
                else _plain_expected_answer(expected_section)
            ),
            "expected_claims": expected_claims if is_answerable else [],
            "gold_evidence": _parse_evidence(_section(content, "页级证据"), case_id) if is_answerable else [],
            "annotation_notes": _parse_notes(content),
        }
        cases.append(CrossDocumentCase.from_dict(record))
    return tuple(cases)


def _page_counts_from_raw(*, corpus_path: Path, raw_dir: Path) -> dict[str, int]:
    """从本地 PDF 获得真实页数，让导入时可以拒绝超范围 Gold 页码。"""
    import pdfplumber

    corpus = load_corpus_manifest(corpus_path)
    page_counts: dict[str, int] = {}
    for paper in corpus.papers:
        path = raw_dir / paper.local_filename
        if not path.is_file():
            raise EvaluationDataError(f"未找到用于页码校验的 PDF：{path}")
        with pdfplumber.open(path) as pdf:
            page_counts[paper.paper_key] = len(pdf.pages)
    return page_counts


def write_crossdoc_dataset(*, cases: tuple[CrossDocumentCase, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入 AI/CS 跨论文 Markdown 评测题。")
    parser.add_argument("--input", type=Path, required=True, help="本地跨论文 Markdown 标注稿路径。")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_crossdoc_v0.1.jsonl")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_eval_v0.1.corpus.json")
    parser.add_argument(
        "--crossdoc-manifest",
        type=Path,
        default=PROJECT_ROOT / "eval/manifests/ai_cs_crossdoc_v0.1.manifest.json",
    )
    parser.add_argument("--raw-dir", type=Path, default=PROJECT_ROOT / "data/raw/ai_cs_eval_v0.1")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        markdown = args.input.read_text(encoding="utf-8")
        cases = parse_crossdoc_markdown(markdown)
        corpus = load_corpus_manifest(args.manifest)
        page_counts = _page_counts_from_raw(corpus_path=args.manifest, raw_dir=args.raw_dir)
        validate_crossdoc_cases(
            cases=cases,
            corpus=corpus,
            paper_splits=DEFAULT_PAPER_SPLITS,
            page_counts=page_counts,
        )
        write_crossdoc_dataset(cases=cases, output_path=args.output)
        frozen_manifest = build_crossdoc_manifest(
            dataset_path=args.output,
            source_markdown_path=args.input,
            base_corpus_path=args.manifest,
            paper_splits=DEFAULT_PAPER_SPLITS,
            cases=cases,
        )
        frozen_manifest["pdf_page_counts"] = dict(sorted(page_counts.items()))
        args.crossdoc_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.crossdoc_manifest.write_text(
            json.dumps(frozen_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, EvaluationDataError) as error:
        print(f"跨论文评测集导入失败：{error}", file=sys.stderr)
        return 2
    print(
        "跨论文评测集已冻结："
        f"总计 {len(cases)} 条，Dev={sum(case.split == 'dev' for case in cases)}，"
        f"Test={sum(case.split == 'test' for case in cases)}。"
    )
    print(f"JSONL：{args.output}")
    print(f"Manifest：{args.crossdoc_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
