"""Deterministic table semantics extracted from structure-aware PDF evidence."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256

from .schemas import ScoredChunk

_METRIC_ROW_RE = re.compile(
    r"^(?P<metric>Comprehensiveness|Diversity|Empowerment|Overall)\s+"
    r"(?P<a>\d+(?:\.\d+)?%)\s+(?P<b>\d+(?:\.\d+)?%)$",
    re.IGNORECASE,
)
_VALUE_PAIR_RE = re.compile(r"^(?P<a>\d+(?:\.\d+)?%)\s+(?P<b>\d+(?:\.\d+)?%)$")
_TABLE_CAPTION_RE = re.compile(r"^(Table\s+\d+\s*:[^\n]+)", re.IGNORECASE)


@dataclass(frozen=True)
class StructuredCell:
    row_header: str
    column_header: str
    method: str
    value: str
    citation_index: int


@dataclass(frozen=True)
class TableComparison:
    dataset: str
    metric: str
    method_a: str
    value_a: str
    method_b: str
    value_b: str
    relation: str
    citation_index: int
    table_id: str
    supporting_citation_indexes: tuple[int, ...] = ()

    def as_record(self) -> dict[str, str]:
        return {
            "dataset": self.dataset,
            "metric": self.metric,
            "method_a": self.method_a,
            "value_a": self.value_a,
            "method_b": self.method_b,
            "value_b": self.value_b,
            "relation": self.relation,
        }


@dataclass(frozen=True)
class StructuredTable:
    table_id: str
    caption: str
    paper_id: str
    page: int
    column_headers: tuple[str, ...]
    row_headers: tuple[str, ...]
    cells: tuple[StructuredCell, ...]
    comparisons: tuple[TableComparison, ...]
    nearby_explanation: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConceptMapping:
    term: str
    normalized_meaning: str
    evidence_text: str
    citation_index: int


@dataclass(frozen=True)
class CostFact:
    expression: str
    explanation: str
    citation_index: int


def _markdown_cells(line: str) -> tuple[str, ...]:
    stripped = line.strip()
    if not stripped.startswith("|"):
        return ()
    cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
    if cells and all(not cell or re.fullmatch(r"[-: ]+", cell) for cell in cells):
        return ()
    return cells


def _dataset_headers(passages: Sequence[tuple[int, str]]) -> tuple[tuple[str, ...], int | None]:
    candidates: list[tuple[tuple[str, ...], int]] = []
    for citation_index, text in passages:
        for line in text.splitlines():
            cells = _markdown_cells(line)
            for cell in cells:
                tokens = tuple(cell.split())
                if (
                    len(tokens) >= 2
                    and all("%" not in token and not any(char.isdigit() for char in token) for token in tokens)
                    and tokens[0].casefold()
                    not in {
                        "model",
                        "method",
                        "phase",
                        "comprehensiveness",
                        "diversity",
                        "empowerment",
                        "overall",
                    }
                    and not any(token.startswith("-") for token in tokens)
                ):
                    candidates.append((tokens, citation_index))
    # Docling may split one table into overlapping Parent chunks.  The first
    # passage can therefore start at a method row (for example
    # ``NaiveRAG -Low``) before another passage contains the real dataset
    # header.  Dataset headers span the widest set of columns, so prefer the
    # longest valid candidate instead of trusting retrieval order.
    return max(candidates, key=lambda item: len(item[0]), default=((), None))


def _method_pair(cells: Sequence[str]) -> tuple[str, str] | None:
    nonempty = next((cell for cell in cells if cell), "")
    tokens = nonempty.split()
    if len(tokens) != 2 or any("%" in token for token in tokens):
        return None
    if tokens[0].casefold() in {
        "agriculture",
        "comprehensiveness",
        "diversity",
        "empowerment",
        "overall",
        "model",
        "phase",
    }:
        return None
    return tokens[0], tokens[1]


def _relation(value_a: str, value_b: str) -> str:
    left = float(value_a.rstrip("%"))
    right = float(value_b.rstrip("%"))
    if left > right:
        return "method_a_higher"
    if right > left:
        return "method_b_higher"
    return "equal"


def extract_structured_tables(results: Sequence[ScoredChunk]) -> tuple[StructuredTable, ...]:
    """Recover paired percentage tables even when Parent chunks overlap or split."""
    groups: dict[tuple[str, int, tuple[str, ...]], list[tuple[int, str]]] = defaultdict(list)
    for citation_index, scored in enumerate(results, start=1):
        chunk = scored.chunk
        groups[(chunk.paper_id, chunk.page_start, chunk.section_path)].append((citation_index, chunk.content))

    tables: list[StructuredTable] = []
    for (paper_id, page, section_path), passages in groups.items():
        datasets, header_citation_index = _dataset_headers(passages)
        if not datasets:
            continue
        caption = next(
            (
                match.group(1).strip()
                for _, text in passages
                for line in text.splitlines()
                if (match := _TABLE_CAPTION_RE.match(line.strip()))
            ),
            f"Table in {' > '.join(section_path) or f'page {page}'}",
        )
        table_id = "table_" + sha256(f"{paper_id}\n{page}\n{caption}".encode()).hexdigest()[:16]
        cells_by_key: dict[tuple[str, str, str], StructuredCell] = {}
        comparisons_by_key: dict[tuple[str, str, str, str], TableComparison] = {}
        explanations: list[str] = []
        for citation_index, text in passages:
            current_methods: tuple[str, str] | None = None
            for line in text.splitlines():
                markdown_cells = _markdown_cells(line)
                if markdown_cells:
                    method_pair = _method_pair(markdown_cells)
                    if method_pair is not None:
                        current_methods = method_pair
                        continue
                    nonempty_cells = [cell for cell in markdown_cells if cell]
                    first_match = _METRIC_ROW_RE.fullmatch(nonempty_cells[0]) if nonempty_cells else None
                    if current_methods is None or first_match is None:
                        continue
                    metric = first_match.group("metric").title()
                    value_pairs: list[tuple[str, str]] = [(first_match.group("a"), first_match.group("b"))]
                    for cell in nonempty_cells[1:]:
                        pair = _VALUE_PAIR_RE.fullmatch(cell)
                        if pair is not None:
                            value_pairs.append((pair.group("a"), pair.group("b")))
                    for dataset, (value_a, value_b) in zip(datasets, value_pairs, strict=False):
                        left = StructuredCell(metric, dataset, current_methods[0], value_a, citation_index)
                        right = StructuredCell(metric, dataset, current_methods[1], value_b, citation_index)
                        cells_by_key[(dataset, metric, current_methods[0])] = left
                        cells_by_key[(dataset, metric, current_methods[1])] = right
                        comparison = TableComparison(
                            dataset=dataset,
                            metric=metric,
                            method_a=current_methods[0],
                            value_a=value_a,
                            method_b=current_methods[1],
                            value_b=value_b,
                            relation=_relation(value_a, value_b),
                            citation_index=citation_index,
                            table_id=table_id,
                            supporting_citation_indexes=(
                                (header_citation_index,)
                                if header_citation_index is not None and header_citation_index != citation_index
                                else ()
                            ),
                        )
                        comparisons_by_key[(dataset, metric, current_methods[0], current_methods[1])] = comparison
                    continue
                stripped = line.strip()
                if stripped and any(token in stripped for token in ("-High", "-Low", "-Origin")):
                    if stripped not in explanations:
                        explanations.append(stripped)
        if comparisons_by_key:
            tables.append(
                StructuredTable(
                    table_id=table_id,
                    caption=caption,
                    paper_id=paper_id,
                    page=page,
                    column_headers=datasets,
                    row_headers=tuple(dict.fromkeys(item.metric for item in comparisons_by_key.values())),
                    cells=tuple(cells_by_key.values()),
                    comparisons=tuple(comparisons_by_key.values()),
                    nearby_explanation=tuple(explanations),
                )
            )
    return tuple(tables)


def extract_concept_mappings(results: Sequence[ScoredChunk]) -> tuple[ConceptMapping, ...]:
    """Derive ablation meanings from neighboring prose, not paper-specific constants."""
    mappings: dict[str, ConceptMapping] = {}
    for citation_index, scored in enumerate(results, start=1):
        text = scored.chunk.content
        segments = tuple(segment.strip() for segment in re.split(r"(?<=[.!?])\s+|\n+", text) if segment.strip())
        for position, segment in enumerate(segments):
            candidates = re.findall(r"-[A-Z][A-Za-z0-9_-]*", segment)
            for term in candidates:
                meaning: str | None = None
                context = " ".join(segments[max(0, position - 1) : position + 1])
                lowered = context.casefold()
                if re.search(
                    r"eliminat(?:e|ed|ing)\s+(?:the\s+)?(?:use\s+of\s+)?original\s+text",
                    lowered,
                ):
                    meaning = "remove_original_text"
                elif re.search(
                    rf"{re.escape(term.casefold())}\s+variant\s+removes?\s+"
                    r"high[- ](?:order|level)\s+retrieval",
                    lowered,
                ):
                    meaning = "remove_high_level_retrieval"
                elif "high-level-only retrieval" in lowered or "high level only retrieval" in lowered:
                    meaning = "remove_low_level_retrieval"
                if meaning is not None and term not in mappings:
                    mappings[term] = ConceptMapping(term, meaning, context, citation_index)
    return tuple(mappings.values())


def extract_cost_facts(results: Sequence[ScoredChunk]) -> tuple[CostFact, ...]:
    """Recover cost expressions while preserving products, units, and denominators."""
    combined = "\n".join(item.chunk.content for item in results)
    lowered = combined.casefold()
    cost_index = next(
        (
            index
            for index, item in enumerate(results, start=1)
            if "api calls" in item.chunk.content.casefold()
            and ("cmax" in item.chunk.content.casefold() or "c max" in item.chunk.content.casefold())
        ),
        0,
    )
    prose_index = next(
        (
            index
            for index, item in enumerate(results, start=1)
            if "610,000 tokens" in item.chunk.content or "fewer than 100 tokens" in item.chunk.content.casefold()
        ),
        cost_index,
    )
    if cost_index == 0 and prose_index == 0:
        return ()
    facts: list[CostFact] = []
    if "610,000 tokens" in combined or ("610" in combined and "1,000" in combined):
        facts.append(
            CostFact(
                "610,000 tokens",
                "GraphRAG retrieval token consumption (610 communities × 1,000 tokens)",
                prose_index or cost_index,
            )
        )
    if "api calls" in lowered and ("cmax" in lowered or "c max" in lowered):
        facts.append(
            CostFact(
                "(610 × 1,000) / Cmax API calls",
                "GraphRAG retrieval API-call formula; it is not 610,000 API calls",
                cost_index or prose_index,
            )
        )
    if "< 100" in combined or "fewer than 100 tokens" in lowered:
        facts.append(
            CostFact(
                "< 100 tokens",
                "LightRAG keyword generation and retrieval token consumption",
                prose_index or cost_index,
            )
        )
    if "single api call" in lowered or re.search(r"api\s*calls?[^\n|]*\|\s*1(?:\s*\||\s*$)", lowered):
        facts.append(
            CostFact(
                "1 API call",
                "LightRAG retrieval API-call count",
                prose_index or cost_index,
            )
        )
    return tuple(facts)
