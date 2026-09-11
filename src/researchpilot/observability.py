"""本地可观测性边界：OpenTelemetry Span + 不含正文的 JSONL 审计轨迹。

默认不把问题、回答、PDF 正文或 API Key 写入 Trace。这样本地演示时仍能
定位一次请求经历了什么，而不会把论文内容悄悄复制到日志中。
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry import trace


def new_trace_id() -> str:
    return f"api_{uuid.uuid4().hex}"


def _scope_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class ApiTraceEvent:
    trace_id: str
    operation: str
    status: str
    elapsed_ms: float
    user_scope_hash: str
    session_scope_hash: str
    retrieval_trace_id: str | None = None
    candidate_count: int | None = None
    citation_count: int | None = None
    refused: bool | None = None
    mode: str | None = None
    route: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    rewrite_model_calls: int = 0
    answer_model_calls: int = 0
    metadata_model_calls: int = 0
    cache_hit: bool = False
    estimated_cost: float | None = None
    error_code: str | None = None
    recorded_at: str = ""


class LocalTraceStore:
    """Append-only JSONL trace store for local development and Docker demo."""

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._tracer = trace.get_tracer("researchpilot.api")

    @property
    def path(self) -> Path:
        return self._directory / "api_traces.jsonl"

    def record(
        self,
        *,
        trace_id: str,
        operation: str,
        started_at: float,
        user_id: str,
        session_id: str,
        status: str,
        retrieval_trace_id: str | None = None,
        candidate_count: int | None = None,
        citation_count: int | None = None,
        refused: bool | None = None,
        mode: str | None = None,
        route: str | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        rewrite_model_calls: int = 0,
        answer_model_calls: int = 0,
        metadata_model_calls: int = 0,
        cache_hit: bool = False,
        estimated_cost: float | None = None,
        error_code: str | None = None,
    ) -> None:
        event = ApiTraceEvent(
            trace_id=trace_id,
            operation=operation,
            status=status,
            elapsed_ms=(time.perf_counter() - started_at) * 1000,
            user_scope_hash=_scope_digest(user_id),
            session_scope_hash=_scope_digest(session_id),
            retrieval_trace_id=retrieval_trace_id,
            candidate_count=candidate_count,
            citation_count=citation_count,
            refused=refused,
            mode=mode,
            route=route,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            rewrite_model_calls=rewrite_model_calls,
            answer_model_calls=answer_model_calls,
            metadata_model_calls=metadata_model_calls,
            cache_hit=cache_hit,
            estimated_cost=estimated_cost,
            error_code=error_code,
            recorded_at=datetime.now(UTC).isoformat(),
        )
        self._directory.mkdir(parents=True, exist_ok=True)
        with self._tracer.start_as_current_span(operation) as span:
            span.set_attribute("researchpilot.trace_id", trace_id)
            span.set_attribute("researchpilot.status", status)
            span.set_attribute("researchpilot.elapsed_ms", event.elapsed_ms)
            if error_code:
                span.set_attribute("researchpilot.error_code", error_code)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event), ensure_ascii=False, sort_keys=True) + "\n")
