"""产品任务感知检索：在冻结检索内核上补充论文阅读任务的证据选择策略。

该模块只在产品运行时装配。E0--E6 的冻结实验继续直接使用 ``RetrievalService``，
因此这里的章节覆盖和术语扩展不会改变已经发布的实验结果。
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Sequence

from .evidence_planner import EvidenceRole, plan_multi_section_evidence
from .index import IndexHit, VectorIndexRepository
from .retrieval import RetrievalResult, RetrievalService
from .schemas import BlockType, QueryType, RetrievalTrace, ScoredChunk

_EXPERIMENT_SECTION_RE = re.compile(
    r"experiment|evaluation|empirical|dataset|benchmark|baseline|metric|"
    r"implementation|training|setup|实验|评估|数据集|基线|指标|实现|训练",
    re.IGNORECASE,
)
_EXPERIMENT_SETUP_SECTION_RE = re.compile(
    r"experimental\s+settings?|experiment(?:al)?\s+setup|实验设置|实验配置",
    re.IGNORECASE,
)
_EXPERIMENT_RESULTS_SECTION_RE = re.compile(
    r"(?:experimental|evaluation|main)\s+results?|performance|comparison|"
    r"ablation|case\s+stud(?:y|ies)|efficiency|cost(?:\s+\w+){0,4}\s+analysis|"
    r"adaptability\s+analysis|结果|性能|对比|消融|案例分析|效率|成本",
    re.IGNORECASE,
)
_APPENDIX_SECTION_RE = re.compile(r"appendix|supplementary|附录|补充材料", re.IGNORECASE)
_NON_PRIMARY_SECTION_RE = re.compile(
    r"appendix|supplementary|related\s+work|references?|附录|补充材料|相关工作|参考文献",
    re.IGNORECASE,
)
_MULTI_SECTION_ROLE_PATTERNS: dict[EvidenceRole, tuple[re.Pattern[str], re.Pattern[str]]] = {
    EvidenceRole.PROBLEM: (
        re.compile(r"abstract|introduction|background|motivation|摘要|引言|背景", re.IGNORECASE),
        re.compile(r"research problem|challenge|we aim|研究问题|挑战|目标", re.IGNORECASE),
    ),
    EvidenceRole.METHOD: (
        re.compile(r"method|approach|architecture|framework|index|retrieval|方法|架构|索引|检索", re.IGNORECASE),
        re.compile(r"we propose|our method|graph-based|dual-level|本文提出|双层", re.IGNORECASE),
    ),
    EvidenceRole.RESULTS: (
        _EXPERIMENT_RESULTS_SECTION_RE,
        re.compile(r"table\s+\d+|outperform|win\s+rate|ablation|结果|胜率|消融", re.IGNORECASE),
    ),
    EvidenceRole.CONCLUSION: (
        re.compile(r"conclusion|discussion|limitations?|结论|讨论|局限", re.IGNORECASE),
        re.compile(r"in conclusion|overall|limitation|总之|综上|局限", re.IGNORECASE),
    ),
}
_EXPERIMENT_BUCKETS: tuple[re.Pattern[str], ...] = (
    re.compile(r"evaluation\s+datasets?|experimental\s+data|数据集", re.IGNORECASE),
    re.compile(r"baselines?|基线", re.IGNORECASE),
    re.compile(
        r"implementation(?:\s+and\s+evaluation)?\s+details?|training\s+config|"
        r"hyperparameters?|chunk\s+size|实现细节|训练配置|超参数",
        re.IGNORECASE,
    ),
    re.compile(
        r"evaluation\s+(?:dimensions?|criteria)|metrics?|评价指标|评估指标",
        re.IGNORECASE,
    ),
)
_SUMMARY_BUCKETS: tuple[tuple[str, tuple[re.Pattern[str], ...], re.Pattern[str]], ...] = (
    (
        "overview",
        (
            re.compile(r"(?:^|>)\s*abstract\b|摘要", re.IGNORECASE),
            re.compile(r"introduction|引言|背景", re.IGNORECASE),
        ),
        re.compile(r"research problem|we propose|本文提出|研究问题", re.IGNORECASE),
    ),
    (
        "method",
        (re.compile(r"method|approach|architecture|framework|方法|架构|模型", re.IGNORECASE),),
        re.compile(r"our (?:method|approach|framework)|本文方法|核心方法", re.IGNORECASE),
    ),
    (
        "experiment",
        (
            _EXPERIMENT_SETUP_SECTION_RE,
            re.compile(r"evaluation|experiments?|results?|评估|实验|结果", re.IGNORECASE),
        ),
        _EXPERIMENT_SECTION_RE,
    ),
    (
        "results",
        (_EXPERIMENT_RESULTS_SECTION_RE,),
        re.compile(
            r"table\s+\d+|win\s+rates?|outperform|ablation|实验结果|胜率|消融",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion",
        (re.compile(r"conclusion|discussion|limitations?|结论|讨论|局限", re.IGNORECASE),),
        re.compile(r"in conclusion|overall,? (?:we|the)|总之|综上", re.IGNORECASE),
    ),
)
_EXPERIMENT_EXPANSION = (
    "experimental setup datasets baselines evaluation metrics implementation details "
    "training configuration 实验设置 数据集 基线 评估指标 实现细节 训练配置"
)
_EXPERIMENT_RESULTS_EXPANSION = (
    "experimental results evaluation results performance comparison ablation study "
    "efficiency cost win rate table 实验结果 评估结果 性能对比 消融实验 效率 成本"
)
_ABLATION_QUERY_RE = re.compile(r"ablation|消融|-high|-low|-origin|双层", re.IGNORECASE)
_ABLATION_MEANING_RE = re.compile(
    r"low-level-only|high-level-only|variant\s+removes?|eliminat(?:e|ed|ing).{0,40}original\s+text|"
    r"仅低层|仅高层|移除.{0,20}(?:检索|原始文本)",
    re.IGNORECASE,
)
_COST_QUERY_RE = re.compile(r"cost|efficien|token|api\s*calls?|成本|效率|开销", re.IGNORECASE)
_COST_DETAIL_RE = re.compile(
    r"token\s+consumption|tokens?\s+per\s+community|fewer\s+than\s+\d+\s+tokens?|"
    r"single\s+api\s+call|api\s+calls?.{0,80}(?:retrieval|overhead)|"
    r"令牌消耗|单次\s*api\s*调用|检索开销",
    re.IGNORECASE,
)


def _chunk_search_text(hit: IndexHit) -> str:
    section = " ".join(hit.chunk.section_path)
    return f"{section}\n{hit.chunk.content[:600]}"


def _unique_parent_ids(children: Sequence[IndexHit]) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            child.chunk.parent_id
            for child in children
            if isinstance(child.chunk.parent_id, str) and child.chunk.parent_id
        )
    )


def table_quality_score(text: str) -> float:
    """Estimate whether extracted table text is safe enough for numeric evidence.

    This is deliberately parser-agnostic and deterministic.  It rewards a
    stable row/column shape and numeric cells, while penalising replacement
    glyphs and OCR-fragmented operators.  The score is a selection signal, not
    a claim that every cell has been semantically verified.
    """
    normalized = text.strip()
    if not normalized:
        return 0.0
    rows = [line for line in normalized.splitlines() if line.count("|") >= 2]
    numeric_cells = re.findall(r"(?<!\w)\d[\d,]*(?:\.\d+)?\s*%?", normalized)
    replacement_count = normalized.count("�") + normalized.count("□")
    fragmented_operators = len(re.findall(r"\d\s*[��×]\s*\d|\d\s*,\s*\|", normalized))
    score = min(len(rows) / 6, 1.0) * 0.45 + min(len(numeric_cells) / 10, 1.0) * 0.45
    if re.search(r"(?:table|figure)\s+\d+", normalized, re.IGNORECASE):
        score += 0.1
    score -= min((replacement_count + fragmented_operators) * 0.12, 0.8)
    return max(0.0, min(score, 1.0))


def _result_evidence_priority(hit: IndexHit) -> tuple[int, int, str]:
    """让定量表格和含数字结果优先于章节标题或定性宣传语。"""
    table_quality = table_quality_score(hit.chunk.content)
    numeric_count = len(re.findall(r"(?<!\w)\d[\d,]*(?:\.\d+)?\s*%?", hit.chunk.content))
    if BlockType.TABLE in hit.chunk.block_types and table_quality >= 0.45:
        priority = 0
    elif BlockType.PARAGRAPH in hit.chunk.block_types and numeric_count >= 2:
        priority = 1
    elif BlockType.PARAGRAPH in hit.chunk.block_types:
        priority = 2
    elif BlockType.FIGURE_CAPTION in hit.chunk.block_types:
        priority = 3
    elif BlockType.TABLE in hit.chunk.block_types:
        priority = 4
    else:
        priority = 5
    return priority, hit.chunk.page_start, hit.chunk.chunk_id


def _select_balanced_result_children(
    children: Sequence[IndexHit],
    *,
    limit: int,
) -> tuple[IndexHit, ...]:
    """按结果子章节轮询取证，避免一张长表占满全部候选额度。"""
    groups: dict[tuple[str, ...], list[IndexHit]] = {}
    for child in children:
        parent_id = child.chunk.parent_id
        section_text = " > ".join(child.chunk.section_path)
        if (
            not parent_id
            or not _EXPERIMENT_RESULTS_SECTION_RE.search(section_text)
            or _APPENDIX_SECTION_RE.search(section_text)
        ):
            continue
        groups.setdefault(child.chunk.section_path, []).append(child)

    deduplicated_groups: list[list[IndexHit]] = []
    for group in groups.values():
        deduplicated: list[IndexHit] = []
        seen_parents: set[str] = set()
        for hit in sorted(group, key=_result_evidence_priority):
            parent_id = hit.chunk.parent_id
            if parent_id and parent_id not in seen_parents:
                deduplicated.append(hit)
                seen_parents.add(parent_id)
        if deduplicated:
            deduplicated_groups.append(deduplicated)

    ordered_groups = sorted(
        deduplicated_groups,
        key=lambda group: (
            group[0].chunk.page_start,
            " > ".join(group[0].chunk.section_path),
        ),
    )
    selected: list[IndexHit] = []
    round_index = 0
    while len(selected) < limit:
        added = False
        for group in ordered_groups:
            if round_index < len(group):
                selected.append(group[round_index])
                added = True
                if len(selected) >= limit:
                    break
        if not added:
            break
        round_index += 1
    return tuple(selected)


def _hydrate(
    *,
    index: VectorIndexRepository,
    children: Sequence[IndexHit],
    user_id: str,
    session_id: str,
    paper_id: str | None,
) -> tuple[ScoredChunk, ...]:
    parent_ids = _unique_parent_ids(children)
    parents = index.get_parent_chunks(
        parent_chunk_ids=parent_ids,
        user_id=user_id,
        session_id=session_id,
        paper_id=paper_id,
    )
    results: list[ScoredChunk] = []
    for rank, parent_id in enumerate(parent_ids, start=1):
        parent = parents.get(parent_id)
        if parent is None:
            continue
        results.append(
            ScoredChunk(
                chunk=parent.chunk,
                source_name=parent.source_name,
                fusion_score=1.0 / (60 + rank),
            )
        )
    return tuple(results)


class ProductTaskRetrievalService(RetrievalService):
    """为元信息、全文总结和实验设置提供确定性的产品检索策略。"""

    def __init__(
        self,
        retriever: object,
        *,
        index: VectorIndexRepository,
        result_limit: int = 10,
    ) -> None:
        super().__init__(retriever)  # type: ignore[arg-type]
        if not isinstance(result_limit, int) or result_limit < 1:
            raise ValueError("result_limit 必须是正整数。")
        self._index = index
        self._result_limit = result_limit

    def retrieve(
        self,
        *,
        query: str,
        user_id: str,
        session_id: str,
        search_query: str | None = None,
        query_type: QueryType | None = None,
        recent_history: str = "",
        paper_id: str | None = None,
    ) -> RetrievalResult:
        resolved_query = (search_query or query).strip()
        if query_type is QueryType.METADATA:
            return self._front_matter(
                query=query,
                search_query=resolved_query,
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
            )
        if query_type is QueryType.DOCUMENT_SUMMARY:
            return self._section_balanced_summary(
                query=query,
                search_query=resolved_query,
                user_id=user_id,
                session_id=session_id,
                recent_history=recent_history,
                paper_id=paper_id,
            )
        if query_type is QueryType.EXPERIMENTAL_SETUP:
            return self._experimental_setup(
                query=query,
                search_query=resolved_query,
                user_id=user_id,
                session_id=session_id,
                recent_history=recent_history,
                paper_id=paper_id,
            )
        if query_type is QueryType.EXPERIMENTAL_RESULTS:
            return self._experimental_results(
                query=query,
                search_query=resolved_query,
                user_id=user_id,
                session_id=session_id,
                recent_history=recent_history,
                paper_id=paper_id,
            )
        if query_type is QueryType.MULTI_SECTION:
            return self._multi_section(
                query=query,
                search_query=resolved_query,
                user_id=user_id,
                session_id=session_id,
                recent_history=recent_history,
                paper_id=paper_id,
            )
        return super().retrieve(
            query=query,
            user_id=user_id,
            session_id=session_id,
            search_query=search_query,
            query_type=query_type,
            recent_history=recent_history,
            paper_id=paper_id,
        )

    def _trace(
        self,
        *,
        query: str,
        search_query: str,
        query_type: QueryType,
        results: Sequence[ScoredChunk],
        candidate_count: int,
        started_at: float,
        stages: tuple[str, ...],
        expanded_queries: tuple[str, ...],
        index_version: str | None = None,
    ) -> RetrievalResult:
        resolved_results = tuple(results[: self._result_limit])
        trace = RetrievalTrace(
            trace_id=f"trace_{uuid.uuid4().hex}",
            query=query.strip(),
            search_query=search_query,
            query_type=query_type,
            candidate_count=max(candidate_count, len(resolved_results)),
            results=resolved_results,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            stages=stages,
            expanded_queries=expanded_queries,
            index_version=index_version,
        )
        return RetrievalResult(results=resolved_results, trace=trace)

    def _front_matter(
        self,
        *,
        query: str,
        search_query: str,
        user_id: str,
        session_id: str,
        paper_id: str | None,
    ) -> RetrievalResult:
        started_at = time.perf_counter()
        children = tuple(
            sorted(
                self._index.list_children(
                    user_id=user_id,
                    session_id=session_id,
                    paper_id=paper_id,
                ),
                key=lambda item: (item.chunk.page_start, item.chunk.chunk_id),
            )
        )
        # 作者、标题、会议通常位于首页；第二页作为双栏首页或超长作者列表的兼容范围。
        front_children = tuple(child for child in children if child.chunk.page_start <= 2)
        results = _hydrate(
            index=self._index,
            children=front_children,
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )
        return self._trace(
            query=query,
            search_query=search_query,
            query_type=QueryType.METADATA,
            results=results,
            candidate_count=len(front_children),
            started_at=started_at,
            stages=("front_matter_selection", "parent_hydration"),
            expanded_queries=(search_query,),
        )

    def _section_balanced_summary(
        self,
        *,
        query: str,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str,
        paper_id: str | None,
    ) -> RetrievalResult:
        started_at = time.perf_counter()
        children = tuple(
            sorted(
                self._index.list_children(
                    user_id=user_id,
                    session_id=session_id,
                    paper_id=paper_id,
                ),
                key=lambda item: (item.chunk.page_start, item.chunk.chunk_id),
            )
        )
        selected: list[IndexHit] = []
        selected_parents: set[str] = set()
        for bucket_name, section_patterns, content_fallback in _SUMMARY_BUCKETS:
            if bucket_name == "results":
                for result_child in _select_balanced_result_children(children, limit=3):
                    parent_id = result_child.chunk.parent_id
                    if parent_id and parent_id not in selected_parents:
                        selected.append(result_child)
                        selected_parents.add(parent_id)
                continue
            matched: IndexHit | None = None
            for pattern in section_patterns:
                matched = next(
                    (
                        child
                        for child in children
                        if child.chunk.parent_id
                        and child.chunk.parent_id not in selected_parents
                        and pattern.search(" > ".join(child.chunk.section_path))
                    ),
                    None,
                )
                if matched is not None:
                    break
            if matched is None:
                matched = next(
                    (
                        child
                        for child in children
                        if child.chunk.parent_id
                        and child.chunk.parent_id not in selected_parents
                        and content_fallback.search(child.chunk.content[:800])
                    ),
                    None,
                )
            if matched is not None and matched.chunk.parent_id is not None:
                selected.append(matched)
                selected_parents.add(matched.chunk.parent_id)
        section_results = list(
            _hydrate(
                index=self._index,
                children=selected,
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
            )
        )
        # 结构标题缺失或论文命名特殊时，用冻结的 Hybrid 结果补足，而不是返回空摘要。
        base = super().retrieve(
            query=query,
            user_id=user_id,
            session_id=session_id,
            search_query=search_query,
            query_type=QueryType.DOCUMENT_SUMMARY,
            recent_history=recent_history,
            paper_id=paper_id,
        )
        by_chunk_id = {item.chunk.chunk_id: item for item in section_results}
        for item in base.results:
            by_chunk_id.setdefault(item.chunk.chunk_id, item)
        return self._trace(
            query=query,
            search_query=search_query,
            query_type=QueryType.DOCUMENT_SUMMARY,
            results=tuple(by_chunk_id.values()),
            candidate_count=len(children) + base.trace.candidate_count,
            started_at=started_at,
            stages=("section_balanced_selection", *base.trace.stages, "summary_context_merge"),
            expanded_queries=base.trace.expanded_queries,
            index_version=base.trace.index_version,
        )

    def _experimental_setup(
        self,
        *,
        query: str,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str,
        paper_id: str | None,
    ) -> RetrievalResult:
        started_at = time.perf_counter()
        children = tuple(
            sorted(
                self._index.list_children(
                    user_id=user_id,
                    session_id=session_id,
                    paper_id=paper_id,
                ),
                key=lambda item: (item.chunk.page_start, item.chunk.chunk_id),
            )
        )
        structured_children: list[IndexHit] = []
        structured_parent_ids: set[str] = set()

        # 先保留 Experimental Settings 章节中的多个不同父块。数据集、基线和
        # 实现配置分落在相邻父块时，不会再因全局 Top-K 竞争而互相挤掉。
        for child in children:
            parent_id = child.chunk.parent_id
            section = " ".join(child.chunk.section_path)
            if parent_id and parent_id not in structured_parent_ids and _EXPERIMENT_SETUP_SECTION_RE.search(section):
                structured_children.append(child)
                structured_parent_ids.add(parent_id)
                if len(structured_children) >= 6:
                    break

        # 标题识别不稳定时，按实验子主题各补一个父块。
        for pattern in _EXPERIMENT_BUCKETS:
            for child in children:
                parent_id = child.chunk.parent_id
                if parent_id and parent_id not in structured_parent_ids and pattern.search(_chunk_search_text(child)):
                    structured_children.append(child)
                    structured_parent_ids.add(parent_id)
                    break

        structured_results = _hydrate(
            index=self._index,
            children=structured_children,
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )
        query_variants = (search_query, _EXPERIMENT_EXPANSION)
        executions = tuple(
            super(ProductTaskRetrievalService, self).retrieve(
                query=query,
                user_id=user_id,
                session_id=session_id,
                search_query=variant,
                query_type=QueryType.EXPERIMENTAL_SETUP,
                recent_history=recent_history,
                paper_id=paper_id,
            )
            for variant in query_variants
        )
        merged: dict[str, tuple[ScoredChunk, int]] = {}
        for execution in executions:
            for rank, item in enumerate(execution.results, start=1):
                current = merged.get(item.chunk.chunk_id)
                if current is None or rank < current[1]:
                    merged[item.chunk.chunk_id] = (item, rank)
        hybrid_ranked = sorted(
            merged.values(),
            key=lambda pair: (
                0
                if _EXPERIMENT_SETUP_SECTION_RE.search(" ".join(pair[0].chunk.section_path))
                else 1
                if _EXPERIMENT_SECTION_RE.search(
                    " ".join(pair[0].chunk.section_path) + "\n" + pair[0].chunk.content[:600]
                )
                else 2,
                pair[1],
                pair[0].chunk.page_start,
                pair[0].chunk.chunk_id,
            ),
        )
        merged_results = {item.chunk.chunk_id: item for item in structured_results}
        for item, _ in hybrid_ranked:
            merged_results.setdefault(item.chunk.chunk_id, item)
        stages = tuple(dict.fromkeys(stage for execution in executions for stage in execution.trace.stages)) + (
            "experimental_structure_selection",
            "experimental_query_merge",
            "experimental_section_priority",
        )
        return self._trace(
            query=query,
            search_query=search_query,
            query_type=QueryType.EXPERIMENTAL_SETUP,
            results=tuple(merged_results.values()),
            candidate_count=len(children) + sum(item.trace.candidate_count for item in executions),
            started_at=started_at,
            stages=stages,
            expanded_queries=query_variants,
            index_version=executions[0].trace.index_version,
        )

    def _experimental_results(
        self,
        *,
        query: str,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str,
        paper_id: str | None,
    ) -> RetrievalResult:
        """优先覆盖结果、消融和效率章节，并用 Hybrid 结果补足具体数字。"""
        started_at = time.perf_counter()
        children = tuple(
            sorted(
                self._index.list_children(
                    user_id=user_id,
                    session_id=session_id,
                    paper_id=paper_id,
                ),
                key=lambda item: (item.chunk.page_start, item.chunk.chunk_id),
            )
        )
        # 第一阶段按结果子章节轮询配额并优先选择表格/数字证据，避免摘要、引言
        # 中的 performance 宣传语或同一张超长表格提前耗尽全部候选额度。
        is_ablation_query = bool(_ABLATION_QUERY_RE.search(query))
        structured_limit = 6 if is_ablation_query else 8
        structured_children = list(_select_balanced_result_children(children, limit=structured_limit))
        selected_parent_ids = {child.chunk.parent_id for child in structured_children if child.chunk.parent_id}

        # A table label such as ``-High`` is not self-explanatory.  For
        # ablation questions, reserve context for the neighboring prose that
        # states which component was removed.  This gives the deterministic
        # table layer evidence for the mapping instead of hard-coding one
        # paper's variant names in product code.
        if is_ablation_query:
            for child in children:
                parent_id = child.chunk.parent_id
                section = " > ".join(child.chunk.section_path)
                if (
                    parent_id
                    and parent_id not in selected_parent_ids
                    and _EXPERIMENT_RESULTS_SECTION_RE.search(section)
                    and _ABLATION_MEANING_RE.search(child.chunk.content)
                ):
                    structured_children.append(child)
                    selected_parent_ids.add(parent_id)
                    if len(structured_children) >= 9:
                        break

        # 图表中的公式可能因 PDF 字形解析而残缺；成本问题必须同时保留邻近
        # 正文，利用其完整单位和自然语言说明校验表格单元格。该规则只识别
        # 通用的成本/调用语义，不硬编码某篇论文的数字。
        if _COST_QUERY_RE.search(query):
            for child in children:
                parent_id = child.chunk.parent_id
                section = " > ".join(child.chunk.section_path)
                if (
                    parent_id
                    and parent_id not in selected_parent_ids
                    and _EXPERIMENT_RESULTS_SECTION_RE.search(section)
                    and _COST_DETAIL_RE.search(child.chunk.content)
                ):
                    structured_children.append(child)
                    selected_parent_ids.add(parent_id)
                    if len(structured_children) >= 9:
                        break

        # 结构标题不完整时，才在 Evaluation 范围内按正文与表格内容补足。不能把
        # Introduction 中的定性宣传语当作主结果证据。
        if len(structured_children) < 8:
            for child in children:
                parent_id = child.chunk.parent_id
                section = " > ".join(child.chunk.section_path)
                if (
                    parent_id
                    and parent_id not in selected_parent_ids
                    and _EXPERIMENT_SECTION_RE.search(section)
                    and _EXPERIMENT_RESULTS_SECTION_RE.search(_chunk_search_text(child))
                ):
                    structured_children.append(child)
                    selected_parent_ids.add(parent_id)
                    if len(structured_children) >= 8:
                        break

        structured_results = _hydrate(
            index=self._index,
            children=structured_children,
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )
        query_variants = (search_query, _EXPERIMENT_RESULTS_EXPANSION)
        executions = tuple(
            super(ProductTaskRetrievalService, self).retrieve(
                query=query,
                user_id=user_id,
                session_id=session_id,
                search_query=variant,
                query_type=QueryType.EXPERIMENTAL_RESULTS,
                recent_history=recent_history,
                paper_id=paper_id,
            )
            for variant in query_variants
        )
        merged_results = {item.chunk.chunk_id: item for item in structured_results}
        for execution in executions:
            for item in execution.results:
                merged_results.setdefault(item.chunk.chunk_id, item)
        stages = tuple(dict.fromkeys(stage for execution in executions for stage in execution.trace.stages)) + (
            "experimental_results_structure_selection",
            "experimental_results_query_merge",
        )
        return self._trace(
            query=query,
            search_query=search_query,
            query_type=QueryType.EXPERIMENTAL_RESULTS,
            results=tuple(merged_results.values()),
            candidate_count=len(children) + sum(item.trace.candidate_count for item in executions),
            started_at=started_at,
            stages=stages,
            expanded_queries=query_variants,
            index_version=executions[0].trace.index_version,
        )

    def _multi_section(
        self,
        *,
        query: str,
        search_query: str,
        user_id: str,
        session_id: str,
        recent_history: str,
        paper_id: str | None,
    ) -> RetrievalResult:
        """Reserve evidence for each requested section role, then merge Hybrid hits."""
        started_at = time.perf_counter()
        plan = plan_multi_section_evidence(query)
        children = tuple(
            sorted(
                self._index.list_children(
                    user_id=user_id,
                    session_id=session_id,
                    paper_id=paper_id,
                ),
                key=lambda item: (item.chunk.page_start, item.chunk.chunk_id),
            )
        )
        selected_children: list[IndexHit] = []
        selected_parent_ids: set[str] = set()
        covered_roles: list[str] = []
        executions: list[RetrievalResult] = []

        for requirement in plan.requirements:
            section_pattern, content_pattern = _MULTI_SECTION_ROLE_PATTERNS[requirement.role]
            role_children = [
                child
                for child in children
                if child.chunk.parent_id
                and child.chunk.parent_id not in selected_parent_ids
                and not _NON_PRIMARY_SECTION_RE.search(" > ".join(child.chunk.section_path))
                and (
                    section_pattern.search(" > ".join(child.chunk.section_path))
                    or content_pattern.search(child.chunk.content[:900])
                )
            ]
            for child in role_children[: requirement.minimum_parents]:
                selected_children.append(child)
                if child.chunk.parent_id:
                    selected_parent_ids.add(child.chunk.parent_id)
            if role_children:
                covered_roles.append(requirement.role.value)

            execution = super().retrieve(
                query=query,
                user_id=user_id,
                session_id=session_id,
                search_query=requirement.search_query,
                query_type=QueryType.MULTI_SECTION,
                recent_history=recent_history,
                paper_id=paper_id,
            )
            executions.append(execution)

        merged_results: dict[str, ScoredChunk] = {
            item.chunk.chunk_id: item
            for item in _hydrate(
                index=self._index,
                children=selected_children,
                user_id=user_id,
                session_id=session_id,
                paper_id=paper_id,
            )
        }
        # One candidate per role is admitted before additional hits so a single
        # semantic neighborhood cannot consume the whole context budget.
        for execution in executions:
            if execution.results:
                item = execution.results[0]
                merged_results.setdefault(item.chunk.chunk_id, item)
        for execution in executions:
            for item in execution.results[1:]:
                section = " > ".join(item.chunk.section_path)
                if not _NON_PRIMARY_SECTION_RE.search(section):
                    merged_results.setdefault(item.chunk.chunk_id, item)

        base_stages = tuple(dict.fromkeys(stage for execution in executions for stage in execution.trace.stages))
        return self._trace(
            query=query,
            search_query=search_query,
            query_type=QueryType.MULTI_SECTION,
            results=tuple(merged_results.values()),
            candidate_count=len(children) + sum(item.trace.candidate_count for item in executions),
            started_at=started_at,
            stages=(
                f"evidence_plan:{plan.planner_version}",
                *(f"evidence_role_covered:{role}" for role in covered_roles),
                *(f"role_retrieved:{role}" for role in covered_roles),
                *(f"role_selected:{role}" for role in covered_roles),
                *base_stages,
                "multi_section_role_quota_merge",
            ),
            expanded_queries=tuple(item.search_query for item in plan.requirements),
            index_version=executions[0].trace.index_version if executions else None,
        )
