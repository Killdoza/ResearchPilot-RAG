"""将人工编写的 Markdown 评测题草稿安全地转换为版本化 JSONL。"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.evaluation import (  # noqa: E402
    EVALUATION_TASK_TYPES,
    EvalCase,
    EvaluationDataError,
    GoldEvidence,
    load_eval_cases,
)


class MarkdownEvalDraftError(ValueError):
    """Markdown 草稿不符合可导入的人工标注格式。"""


@dataclass(frozen=True)
class DraftQuestion:
    paper_key: str
    ordinal: int
    question: str
    expected_answer: str
    evidence: tuple[GoldEvidence, ...]
    task_type: str
    split: str
    answerable: bool


_PAPER_KEY_BY_TITLE = {
    "Deep Residual Learning for Image Recognition": "resnet_2015",
    "Attention Is All You Need": "transformer_2017",
    "Proximal Policy Optimization Algorithms": "ppo_2017",
    "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks": "rag_2020",
    "LoRA: Low-Rank Adaptation of Large Language Models": "lora_2021",
    "MapReduce: Simplified Data Processing on Large Clusters": "mapreduce_2004",
    "Spectre Attacks: Exploiting Speculative Execution": "spectre_2018",
    "A Simple Framework for Contrastive Learning of Visual Representations": "simclr_2020",
    "Denoising Diffusion Probabilistic Models": "ddpm_2020",
    "Segment Anything": "sam_2023",
}

# 评测类型记录的是题目主要考察能力，而不是论文的原始语言。所有草稿问题均为中文，
# 因此它们天然还会参与跨语言检索难度分析，但不应把 48 条全部伪装成同一种题型。
_TASK_TYPES_BY_PAPER = {
    "resnet_2015": (
        "semantic_explanation",
        "semantic_explanation",
        "semantic_explanation",
        "semantic_explanation",
        "experimental_setup",
        "experimental_setup",
        "experimental_setup",
        "experimental_setup",
    ),
    "transformer_2017": (
        "semantic_explanation",
        "semantic_explanation",
        "keyword_exact",
        "keyword_exact",
        "semantic_explanation",
        "multi_section",
        "experimental_setup",
        "experimental_setup",
    ),
    "ppo_2017": (
        "semantic_explanation",
        "local_fact",
        "semantic_explanation",
        "local_fact",
        "multi_section",
        "experimental_setup",
        "experimental_setup",
        "experimental_setup",
    ),
    "rag_2020": (
        "cross_lingual",
        "semantic_explanation",
        "semantic_explanation",
        "semantic_explanation",
        "semantic_explanation",
        "experimental_setup",
        "experimental_setup",
        "multi_section",
    ),
    "lora_2021": (
        "semantic_explanation",
        "semantic_explanation",
        "local_fact",
        "semantic_explanation",
        "semantic_explanation",
        "experimental_setup",
        "experimental_setup",
        "experimental_setup",
    ),
    "mapreduce_2004": (
        "semantic_explanation",
        "keyword_exact",
        "multi_section",
        "semantic_explanation",
        "semantic_explanation",
        "experimental_setup",
        "semantic_explanation",
        "experimental_setup",
    ),
    "spectre_2018": (
        "semantic_explanation",
        "local_fact",
        "experimental_setup",
        "keyword_exact",
        "unanswerable",
        "multi_section",
    ),
    "simclr_2020": (
        "semantic_explanation",
        "local_fact",
        "experimental_setup",
        "keyword_exact",
        "unanswerable",
        "experimental_setup",
    ),
    "ddpm_2020": (
        "semantic_explanation",
        "local_fact",
        "experimental_setup",
        "semantic_explanation",
        "unanswerable",
        "multi_section",
    ),
    "sam_2023": (
        "semantic_explanation",
        "local_fact",
        "experimental_setup",
        "keyword_exact",
        "unanswerable",
        "semantic_explanation",
    ),
}

_SPLIT_BY_PAPER = {
    "resnet_2015": "dev",
    "transformer_2017": "dev",
    "ppo_2017": "dev",
    "rag_2020": "dev",
    "lora_2021": "dev",
    "mapreduce_2004": "dev",
    "spectre_2018": "test",
    "simclr_2020": "test",
    "ddpm_2020": "test",
    "sam_2023": "test",
}

_PAPER_HEADER = re.compile(r"(?m)^## 论文：(?P<title>.+?)\s*$")
_QUESTION_HEADER = re.compile(r"(?m)^###\s+(?P<ordinal>[1-9]\d*)\s*(?:[.。]\s*)?(?:【[^】]+】)?\s*$")
_QUESTION_BLOCK = re.compile(
    r"\*\*问题：(?P<question>.*?)\*\*\s*\n\s*\*\*标准答案：\*\*\s*\n"
    r"(?P<answer>.*?)\n\s*\*\*证据 PDF 页：(?P<evidence>.*?)\*\*",
    re.DOTALL,
)
_PAGE_RANGE = re.compile(r"(?P<start>\d+)(?:\s*[–-]\s*(?P<end>\d+))?")
_CASE_ID = re.compile(r"^ai_cs_v01_q(?P<number>\d+)$")


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _parse_evidence(*, paper_key: str, raw: str) -> tuple[GoldEvidence, ...]:
    pages: list[GoldEvidence] = []
    for match in _PAGE_RANGE.finditer(raw):
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        pages.append(
            GoldEvidence(
                paper_key=paper_key,
                page_start=start,
                page_end=end,
                section_hint="人工 Markdown 草稿标注",
            )
        )
    if not pages:
        raise MarkdownEvalDraftError(f"{paper_key} 的证据页无法解析：{raw!r}")
    return tuple(pages)


def parse_markdown_draft(markdown: str) -> tuple[DraftQuestion, ...]:
    """解析本项目采用的“论文 / 问题 / 标准答案 / 证据页”草稿格式。"""
    if not isinstance(markdown, str) or not markdown.strip():
        raise MarkdownEvalDraftError("Markdown 草稿不能为空。")
    paper_headers = tuple(_PAPER_HEADER.finditer(markdown))
    if not paper_headers:
        raise MarkdownEvalDraftError("未找到“## 论文：...”标题。")

    parsed: list[DraftQuestion] = []
    seen_papers: set[str] = set()
    for index, paper_header in enumerate(paper_headers):
        title = paper_header.group("title").strip()
        paper_key = _PAPER_KEY_BY_TITLE.get(title)
        if paper_key is None:
            raise MarkdownEvalDraftError(f"草稿论文不在 v0.1 评测语料中：{title}")
        if paper_key in seen_papers:
            raise MarkdownEvalDraftError(f"草稿中论文重复：{title}")
        seen_papers.add(paper_key)
        section_end = paper_headers[index + 1].start() if index + 1 < len(paper_headers) else len(markdown)
        section = markdown[paper_header.end() : section_end]
        question_headers = tuple(_QUESTION_HEADER.finditer(section))
        task_types = _TASK_TYPES_BY_PAPER[paper_key]
        expected_question_count = len(task_types)
        if len(question_headers) != expected_question_count:
            raise MarkdownEvalDraftError(
                f"{title} 应包含 {expected_question_count} 道题，实际解析到 {len(question_headers)} 道。"
            )
        expected_ordinals = list(range(1, expected_question_count + 1))
        actual_ordinals = [int(header.group("ordinal")) for header in question_headers]
        if actual_ordinals != expected_ordinals:
            raise MarkdownEvalDraftError(
                f"{title} 的题号必须连续为 1 到 {expected_question_count}，实际为 {actual_ordinals}。"
            )
        for question_header, task_type in zip(question_headers, task_types, strict=True):
            question_end = next(
                (header.start() for header in question_headers if header.start() > question_header.start()),
                len(section),
            )
            raw_question = section[question_header.end() : question_end]
            block = _QUESTION_BLOCK.search(raw_question)
            if block is None:
                raise MarkdownEvalDraftError(
                    f"{title} 第 {question_header.group('ordinal')} 题缺少问题、答案或证据页字段。"
                )
            question = _normalized_text(block.group("question"))
            answer = block.group("answer").strip()
            if not question or not answer:
                raise MarkdownEvalDraftError(f"{title} 第 {question_header.group('ordinal')} 题的问题或答案为空。")
            if task_type not in EVALUATION_TASK_TYPES:
                raise MarkdownEvalDraftError(f"未知评测类型：{task_type}")
            answerable = task_type != "unanswerable"
            parsed.append(
                DraftQuestion(
                    paper_key=paper_key,
                    ordinal=int(question_header.group("ordinal")),
                    question=question,
                    expected_answer=answer,
                    evidence=(_parse_evidence(paper_key=paper_key, raw=block.group("evidence")) if answerable else ()),
                    task_type=task_type,
                    split=_SPLIT_BY_PAPER[paper_key],
                    answerable=answerable,
                )
            )
    expected_total = sum(len(task_types) for task_types in _TASK_TYPES_BY_PAPER.values())
    if len(parsed) != expected_total:
        raise MarkdownEvalDraftError(f"草稿应解析出 {expected_total} 道题，实际为 {len(parsed)}。")
    return tuple(parsed)


def _next_case_number(existing_cases: tuple[EvalCase, ...]) -> int:
    numbers = [
        int(match.group("number")) for case in existing_cases if (match := _CASE_ID.fullmatch(case.case_id)) is not None
    ]
    return max(numbers, default=0) + 1


def build_import_cases(
    *, draft_questions: tuple[DraftQuestion, ...], existing_cases: tuple[EvalCase, ...]
) -> tuple[EvalCase, ...]:
    """仅导入未出现过的题目；允许同一份累计 Markdown 被安全地重复执行。"""
    existing_questions = {_normalized_text(case.question) for case in existing_cases}
    seen_draft_questions: set[str] = set()
    next_number = _next_case_number(existing_cases)
    imported: list[EvalCase] = []
    for draft in draft_questions:
        normalized_question = _normalized_text(draft.question)
        if normalized_question in seen_draft_questions:
            raise MarkdownEvalDraftError(f"Markdown 草稿内存在重复问题：{draft.question}")
        seen_draft_questions.add(normalized_question)
        if normalized_question in existing_questions:
            continue
        imported.append(
            EvalCase(
                case_id=f"ai_cs_v01_q{next_number + len(imported):03d}",
                split=draft.split,
                paper_keys=(draft.paper_key,),
                question=draft.question,
                question_language="zh",
                task_type=draft.task_type,
                answerable=draft.answerable,
                gold_evidence=draft.evidence,
                expected_answer=draft.expected_answer,
            )
        )
    return tuple(imported)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="导入人工 Markdown 评测题草稿。")
    parser.add_argument("--source", type=Path, required=True, help="本地单论文 Markdown 标注稿路径。")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "eval/datasets/ai_cs_eval_v0.1.jsonl")
    parser.add_argument("--apply", action="store_true", help="确认覆盖写入 JSONL；默认仅 dry-run。")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        markdown = args.source.read_text(encoding="utf-8")
        draft_questions = parse_markdown_draft(markdown)
        existing_cases = load_eval_cases(args.dataset)
        imported_cases = build_import_cases(draft_questions=draft_questions, existing_cases=existing_cases)
    except (OSError, EvaluationDataError, MarkdownEvalDraftError) as error:
        print(f"评测题草稿校验失败：{error}", file=sys.stderr)
        return 2
    print(
        f"草稿校验通过：{len(draft_questions)} 条；将新增 {len(imported_cases)} 条题，"
        f"导入后共 {len(existing_cases) + len(imported_cases)} 条。"
    )
    if not args.apply:
        print("dry-run 完成：未改写 JSONL。确认后使用 --apply。")
        return 0
    all_cases = existing_cases + imported_cases
    rendered = "".join(
        json.dumps(
            {
                "id": case.case_id,
                "split": case.split,
                "paper_keys": list(case.paper_keys),
                "question": case.question,
                "question_language": case.question_language,
                "task_type": case.task_type,
                "answerable": case.answerable,
                "gold_evidence": [
                    {
                        "paper_key": evidence.paper_key,
                        "page_start": evidence.page_start,
                        "page_end": evidence.page_end,
                        "section_hint": evidence.section_hint,
                    }
                    for evidence in case.gold_evidence
                ],
                "expected_answer": case.expected_answer,
            },
            ensure_ascii=False,
        )
        + "\n"
        for case in all_cases
    )
    args.dataset.write_text(rendered, encoding="utf-8", newline="\n")
    print(f"已写入：{args.dataset}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
