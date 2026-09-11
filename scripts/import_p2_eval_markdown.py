"""把 P2 v0.2 的受控 Markdown 题单导入为冻结 JSONL。

此脚本不调用 LLM，也不会创建索引。它只负责把人工标注转为数据契约，
并在写入前核验论文范围、Dev/Test 隔离和 PDF 物理页码。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.evaluation import EvaluationDataError, load_corpus_manifest  # noqa: E402
from researchpilot.p2_dataset import (  # noqa: E402
    P2_PAPER_SPLITS,
    P2_SCHEMA_VERSION,
    P2Case,
    build_p2_manifest,
    validate_p2_cases,
)

_CASE_HEADING = re.compile(r"(?m)^##\s+(CD\d+)\s*$")
_SCALAR_FIELD = re.compile(
    r"(?m)^\*\s+(split|task_type|answerable|comparison_dimension|comparison_allowed|"
    r"refusal_reason_target|question):\s*(.*?)\s*$"
)
_PAPER_KEY = re.compile(r"(?m)^\s{2,}\*\s+([A-Za-z][A-Za-z0-9_.-]*)\s*$")
_CLAIM = re.compile(r"(?m)^\s*\*\s*(C\d+)：\s*(.+?)\s*$")
_EVIDENCE = re.compile(r"(?ms)^\s*\*\s*evidence_id:\s*(E\d+)\s*$\n(.*?)(?=^\s*\*\s*evidence_id:|^###\s|\Z)")
_EVIDENCE_FIELD = re.compile(
    r"(?m)^\s*\*\s*(claim_ids|paper_key|page_start|page_end|section_hint|"
    r"supporting_text_summary):\s*(.*?)\s*$"
)


def _section(content: str, heading: str) -> str:
    match = re.search(rf"(?ms)^###\s+{re.escape(heading)}\s*$\n(.*?)(?=^###\s|\Z)", content)
    if not match:
        raise EvaluationDataError(f"缺少 ### {heading} 小节。")
    return match.group(1).strip()


def _scalar_fields(content: str, case_id: str) -> dict[str, str]:
    fields = {key: value.strip() for key, value in _SCALAR_FIELD.findall(content)}
    required = {
        "split",
        "task_type",
        "answerable",
        "comparison_dimension",
        "comparison_allowed",
        "question",
    }
    missing = sorted(required - set(fields))
    if missing:
        raise EvaluationDataError(f"{case_id} 缺少字段：{', '.join(missing)}")
    return fields


def _parse_key_list(content: str, *, label: str, case_id: str) -> list[str]:
    match = re.search(rf"(?ms)^\*\s+{re.escape(label)}:\s*$\n(.*?)(?=^\*\s+\w|^###\s|\Z)", content)
    if not match:
        raise EvaluationDataError(f"{case_id} 缺少 {label}。")
    keys = _PAPER_KEY.findall(match.group(1))
    if len(keys) < 2:
        raise EvaluationDataError(f"{case_id} 的 {label} 至少需要两篇论文。")
    return keys


def _parse_evidence(section: str, *, case_id: str) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for evidence_id, body in _EVIDENCE.findall(section):
        fields = {key: value.strip() for key, value in _EVIDENCE_FIELD.findall(body)}
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
            raise EvaluationDataError(f"{case_id}/{evidence_id} 的页码必须是整数。") from error
        evidence.append(
            {
                "evidence_id": evidence_id,
                "claim_ids": [item.strip() for item in claim_match.group(1).split(",")],
                "paper_key": fields["paper_key"],
                "page_start": page_start,
                "page_end": page_end,
                "section_hint": fields["section_hint"],
                "supporting_text_summary": fields["supporting_text_summary"],
            }
        )
    return evidence


def _parse_notes(content: str) -> list[str]:
    match = re.search(r"(?ms)^###\s+标注备注\s*$\n(.*?)(?=^###\s|\Z)", content)
    if not match:
        return []
    return [line.removeprefix("* ").strip() for line in match.group(1).splitlines() if line.strip().startswith("* ")]


def parse_p2_markdown(markdown: str) -> tuple[P2Case, ...]:
    """按 P2 固定模板解析 Markdown，并转换为 P2Case。"""
    headings = list(_CASE_HEADING.finditer(markdown))
    if not headings:
        raise EvaluationDataError("未找到形如 ## CD201 的 P2 题目标题。")
    cases: list[P2Case] = []
    for index, match in enumerate(headings):
        case_id = match.group(1)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        content = markdown[match.end() : end]
        fields = _scalar_fields(content, case_id)
        answerable_text = fields["answerable"].lower()
        if answerable_text not in {"true", "false"}:
            raise EvaluationDataError(f"{case_id} 的 answerable 必须为 true 或 false。")
        comparison_text = fields["comparison_allowed"].lower()
        if comparison_text not in {"true", "false"}:
            raise EvaluationDataError(f"{case_id} 的 comparison_allowed 必须为 true 或 false。")
        answerable = answerable_text == "true"
        expected_section = _section(content, "预期答案")
        claims = _CLAIM.findall(expected_section)
        if answerable and not claims:
            raise EvaluationDataError(f"{case_id} 可回答题必须给出 C1/C2/C3。")
        if not answerable and claims:
            raise EvaluationDataError(f"{case_id} 不可回答题不应伪造 C1/C2/C3。")
        expected_answer = (
            "\n".join(f"{claim_id}：{text.strip()}" for claim_id, text in claims)
            if answerable
            else "\n".join(line.removeprefix("* ").strip() for line in expected_section.splitlines() if line.strip())
        )
        raw: dict[str, Any] = {
            "schema_version": P2_SCHEMA_VERSION,
            "id": case_id,
            "source_id": case_id,
            "split": fields["split"],
            "task_type": fields["task_type"],
            "answerable": answerable,
            "paper_keys": _parse_key_list(content, label="paper_keys", case_id=case_id),
            "required_paper_keys": _parse_key_list(content, label="required_paper_keys", case_id=case_id),
            "comparison_dimension": fields["comparison_dimension"],
            "comparison_allowed": comparison_text == "true",
            "refusal_reason_target": fields.get("refusal_reason_target") or None,
            "question": fields["question"],
            "expected_answer": expected_answer,
            "expected_claims": [{"claim_id": claim_id, "text": text.strip()} for claim_id, text in claims],
            "gold_evidence": _parse_evidence(_section(content, "页级证据"), case_id=case_id) if answerable else [],
            "annotation_notes": _parse_notes(content),
        }
        cases.append(P2Case.from_dict(raw))
    return tuple(cases)


def p2_pdf_page_counts(*, corpus_path: Path, raw_root: Path) -> dict[str, int]:
    """读取本地 PDF 的真实物理页数；目录按 Dev/Test 自动解析。"""
    import pdfplumber

    corpus = load_corpus_manifest(corpus_path)
    page_counts: dict[str, int] = {}
    for paper in corpus.papers:
        split = P2_PAPER_SPLITS.get(paper.paper_key)
        if split is None:
            raise EvaluationDataError(f"P2 没有定义 {paper.paper_key} 的语料 Split。")
        path = raw_root / split / paper.local_filename
        if not path.is_file():
            raise EvaluationDataError(f"未找到用于页码校验的 PDF：{path}")
        if paper.sha256 is not None:
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != paper.sha256:
                raise EvaluationDataError(f"PDF SHA-256 与 Corpus Manifest 不一致：{paper.paper_key}。")
        with pdfplumber.open(path) as pdf:
            page_counts[paper.paper_key] = len(pdf.pages)
    return page_counts


def write_p2_dataset(*, cases: tuple[P2Case, ...], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入 P2 v0.2 跨论文评测 Markdown。")
    parser.add_argument("--input", type=Path, required=True, help="本地 P2 Markdown 标注稿路径。")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_p2_v0.2.jsonl")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.corpus.json")
    parser.add_argument("--p2-manifest", type=Path, default=PROJECT_ROOT / "eval/manifests/ai_cs_p2_v0.2.manifest.json")
    parser.add_argument("--raw-root", type=Path, default=PROJECT_ROOT / "data/raw/ai_cs_eval_v0.2")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        cases = parse_p2_markdown(args.input.read_text(encoding="utf-8"))
        corpus = load_corpus_manifest(args.corpus)
        page_counts = p2_pdf_page_counts(corpus_path=args.corpus, raw_root=args.raw_root)
        validate_p2_cases(cases=cases, corpus=corpus, paper_splits=P2_PAPER_SPLITS, page_counts=page_counts)
        write_p2_dataset(cases=cases, output_path=args.output)
        manifest = build_p2_manifest(
            dataset_path=args.output,
            source_markdown_path=args.input,
            corpus_path=args.corpus,
            paper_splits=P2_PAPER_SPLITS,
            cases=cases,
            page_counts=page_counts,
        )
        args.p2_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.p2_manifest.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except (EvaluationDataError, OSError, UnicodeError) as error:
        print(f"P2 导入失败：{error}", file=sys.stderr)
        return 2
    print(
        f"P2 导入完成：{len(cases)} 题，输出 {args.output.relative_to(PROJECT_ROOT)}，"
        f"冻结清单 {args.p2_manifest.relative_to(PROJECT_ROOT)}。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
