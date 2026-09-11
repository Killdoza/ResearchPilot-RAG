import base64
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.config import (  # noqa: E402
    ConfigurationError,
    build_settings,
    ensure_runtime_directories,
    require_ascii_chroma_path,
)
from researchpilot.evidence_alignment import format_citation_markers_with_pages  # noqa: E402
from researchpilot.ingestion import PdfIngestionError  # noqa: E402
from researchpilot.legacy_adapter import LegacySessionAdapter  # noqa: E402
from researchpilot.memory_service import (  # noqa: E402
    format_recent_history,
    messages_in_scope,
    scope_key_for_papers,
)
from researchpilot.orchestration import RuntimeOrchestrator  # noqa: E402
from researchpilot.research_workflow import (  # noqa: E402
    derive_session_title,
    persist_pdf_copy,
    render_pdf_page_png,
    trim_logo_image,
)
from researchpilot.routing import (  # noqa: E402
    RuntimeEvidenceAnchor,
    RuntimePaper,
    RuntimeRequest,
    RuntimeRouter,
)
from researchpilot.runtime_factory import build_product_runtime_collection  # noqa: E402
from researchpilot.schemas import Citation, QueryType  # noqa: E402
from researchpilot.session import generate_local_user_id, generate_session_id  # noqa: E402
from researchpilot.storage import (  # noqa: E402
    SessionDataCorrupted,
    SessionFileRepository,
    empty_working_memory,
)

EVIDENCE_PREVIEW_LIMIT = 800


def build_paper_evidence_payload(answer, retrieval_trace, *, route=None) -> dict:
    """Convert a service answer into JSON-safe data stored with the chat turn.

    Streamlit reruns after every completed turn.  Keeping only the answer text
    would make page-level citations and retrieval diagnostics disappear on the
    next run, so this compact payload is saved alongside the assistant message.
    """
    results_by_chunk_id = {result.chunk.chunk_id: result for result in retrieval_trace.results}
    citations = []
    for citation in answer.citations:
        scored_chunk = results_by_chunk_id.get(citation.chunk_id)
        scores = {}
        if scored_chunk is not None:
            scores = {
                "dense_score": scored_chunk.dense_score,
                "sparse_score": scored_chunk.sparse_score,
                "fusion_score": scored_chunk.fusion_score,
                "rerank_score": scored_chunk.rerank_score,
                "source_name": scored_chunk.source_name,
            }
        citations.append(
            {
                "paper_id": citation.paper_id,
                "chunk_id": citation.chunk_id,
                "page_start": citation.page_start,
                "page_end": citation.page_end,
                "section_path": list(citation.section_path),
                "evidence": citation.evidence[:EVIDENCE_PREVIEW_LIMIT],
                "evidence_truncated": len(citation.evidence) > EVIDENCE_PREVIEW_LIMIT,
                "local_pdf_path": next(
                    (
                        item.get("local_pdf_path")
                        for item in st.session_state.get("active_documents", [])
                        if isinstance(item, dict) and item.get("paper_id") == citation.paper_id
                    ),
                    None,
                ),
                **scores,
            }
        )

    return {
        "trace_id": answer.trace_id,
        "refused": answer.refused,
        "refusal_reason": answer.refusal_reason,
        "latency_ms": answer.latency_ms,
        "candidate_count": retrieval_trace.candidate_count,
        "search_query": retrieval_trace.search_query,
        "expanded_queries": list(retrieval_trace.expanded_queries),
        "stages": list(retrieval_trace.stages),
        "stage_elapsed_ms": [list(item) for item in retrieval_trace.stage_elapsed_ms],
        "index_version": retrieval_trace.index_version,
        "product_profile": active_product_profile().name,
        "product_profile_label": active_product_profile().label,
        "product_profile_hash": active_product_profile().config_hash,
        "citations": citations,
        "claims": [
            {
                "claim_id": claim.claim_id,
                "text": claim.text,
                "citation_ids": list(claim.citation_ids),
                "support_status": claim.support_status,
                "numeric_values": list(claim.numeric_values),
            }
            for claim in getattr(answer, "claims", ())
        ],
        "answer_contract_version": getattr(answer, "contract_version", "legacy"),
        "numeric_fidelity_passed": getattr(answer, "numeric_fidelity_passed", None),
        "usage": {
            "rewrite_model_calls": getattr(answer, "rewrite_model_calls", 0),
            "answer_model_calls": getattr(answer, "answer_model_calls", 0),
            "metadata_model_calls": getattr(answer, "metadata_model_calls", 0),
            "prompt_tokens": getattr(answer, "prompt_tokens", None),
            "completion_tokens": getattr(answer, "completion_tokens", None),
            "cache_hit": getattr(answer, "cache_hit", False),
            "estimated_cost": getattr(answer, "estimated_cost", None),
        },
        "runtime_route": getattr(route, "value", None),
    }


def render_paper_evidence(message: dict) -> None:
    """Render persisted evidence defensively for both new and older sessions."""
    payload = message.get("paper_evidence")
    if not isinstance(payload, dict):
        return

    trace_id = str(payload.get("trace_id") or "unknown")
    citation_items = payload.get("citations")
    citation_count = len(citation_items) if isinstance(citation_items, list) else 0
    with st.expander(f"📎 论文证据 · {citation_count} 条", expanded=False):
        if payload.get("refused"):
            st.warning(payload.get("refusal_reason") or "未检索到可用论文证据。")
            return

        if isinstance(citation_items, list):
            for index, citation in enumerate(citation_items, start=1):
                if not isinstance(citation, dict):
                    continue
                page_start = citation.get("page_start")
                page_end = citation.get("page_end")
                page_label = f"第 {page_start} 页" if page_start == page_end else f"第 {page_start}-{page_end} 页"
                source_name = citation.get("source_name") or citation.get("paper_id") or "未知论文"
                section_path = citation.get("section_path")
                section_label = " / ".join(str(item) for item in section_path) if isinstance(section_path, list) else ""
                st.markdown(f"**证据 {index}** · {source_name}")
                st.caption(f"{page_label}{' · ' + section_label if section_label else ''}")
                st.markdown(f"> {str(citation.get('evidence') or '未保存证据文本。')}")
                if citation.get("evidence_truncated"):
                    st.caption("（证据预览已截断；完整文本仍可由 Chunk ID 在索引中定位。）")
                local_pdf_path = citation.get("local_pdf_path")
                if isinstance(local_pdf_path, str) and Path(local_pdf_path).is_file():
                    if st.button(
                        f"查看第 {page_start} 页",
                        key=f"open_pdf_{trace_id}_{index}",
                    ):
                        try:
                            page_image = render_pdf_page_png(Path(local_pdf_path), int(page_start))
                            st.image(page_image, caption=f"{source_name} · 第 {page_start} 页")
                        except Exception as error:
                            st.warning(f"暂时无法渲染该页：{error}")

        claim_items = payload.get("claims")
        if isinstance(claim_items, list) and claim_items:
            unsupported = [
                item
                for item in claim_items
                if isinstance(item, dict) and item.get("support_status") in {"unsupported", "numeric_unsupported"}
            ]
            if unsupported:
                st.warning(
                    f"回答契约发现 {len(unsupported)} 条未完全绑定证据的 Claim。"
                    "这不等于内容一定错误，而是该句没有完成明确的证据编号绑定。"
                )
                with st.expander("查看未完全绑定的 Claim", expanded=False):
                    for item in unsupported:
                        claim_id = item.get("claim_id") or "未知 Claim"
                        status = item.get("support_status") or "unsupported"
                        st.markdown(f"- `{claim_id}` · `{status}`：{item.get('text') or '未保存文本'}")
            else:
                st.caption("Claim-Citation 契约：本轮已完成引用范围与数字忠实度检查。")

        if st.checkbox("显示技术详情", value=False, key=f"trace_details_{trace_id}"):
            latency_ms = payload.get("latency_ms")
            candidate_count = payload.get("candidate_count")
            latency_label = f"{float(latency_ms):.0f} ms" if isinstance(latency_ms, int | float) else "未知"
            st.caption(f"Trace：{trace_id}；检索耗时：{latency_label}；候选父块：{candidate_count}")
            runtime_route = payload.get("runtime_route")
            if isinstance(runtime_route, str) and runtime_route:
                st.caption(f"任务路由：{runtime_route}")
            search_query = payload.get("search_query")
            if isinstance(search_query, str) and search_query.strip():
                st.caption(f"最终检索问题：{search_query}")
            stages = payload.get("stages")
            if isinstance(stages, list) and stages:
                st.caption(f"检索阶段：{' → '.join(str(stage) for stage in stages)}")
            stage_elapsed_ms = payload.get("stage_elapsed_ms")
            if isinstance(stage_elapsed_ms, list) and stage_elapsed_ms:
                formatted_timings = []
                for item in stage_elapsed_ms:
                    if isinstance(item, list) and len(item) == 2 and isinstance(item[1], int | float):
                        formatted_timings.append(f"{item[0]}: {float(item[1]):.0f} ms")
                if formatted_timings:
                    st.caption("阶段耗时：" + " | ".join(formatted_timings))
            if isinstance(payload.get("index_version"), str):
                st.caption(f"BM25 索引版本：{payload['index_version']}")
            usage = payload.get("usage")
            if isinstance(usage, dict):
                token_text = (
                    f"Prompt {usage.get('prompt_tokens')} / Completion {usage.get('completion_tokens')}"
                    if usage.get("prompt_tokens") is not None
                    else "Token：模型供应商未返回"
                )
                st.caption(
                    "模型调用："
                    f"重写 {usage.get('rewrite_model_calls', 0)} / "
                    f"回答 {usage.get('answer_model_calls', 0)} / "
                    f"元信息 {usage.get('metadata_model_calls', 0)}；"
                    f"{token_text}；缓存命中：{'是' if usage.get('cache_hit') else '否'}"
                )
            profile_label = payload.get("product_profile_label")
            profile_hash = payload.get("product_profile_hash")
            if isinstance(profile_label, str) and profile_label:
                suffix = f" · 配置 {profile_hash}" if isinstance(profile_hash, str) else ""
                st.caption(f"运行模式：{profile_label}{suffix}")
            if isinstance(citation_items, list):
                for index, citation in enumerate(citation_items, start=1):
                    if not isinstance(citation, dict):
                        continue
                    score_items = [
                        ("Dense", citation.get("dense_score")),
                        ("BM25", citation.get("sparse_score")),
                        ("RRF", citation.get("fusion_score")),
                        ("Rerank", citation.get("rerank_score")),
                    ]
                    score_text = "；".join(
                        f"{label}: {float(score):.4f}" for label, score in score_items if isinstance(score, int | float)
                    )
                    chunk_id = citation.get("chunk_id") or "未知块"
                    st.caption(f"证据 {index} · Chunk：{chunk_id}{'；' + score_text if score_text else ''}")


def _citations_from_payload(message: dict) -> tuple[Citation, ...]:
    payload = message.get("paper_evidence")
    rows = payload.get("citations") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return ()
    citations: list[Citation] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            citations.append(
                Citation(
                    paper_id=str(row["paper_id"]),
                    chunk_id=str(row["chunk_id"]),
                    page_start=int(row["page_start"]),
                    page_end=int(row["page_end"]),
                    section_path=tuple(str(item) for item in row.get("section_path", [])),
                    evidence=str(row.get("evidence") or "证据预览未保存"),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return tuple(citations)


def render_save_note_action(message: dict, message_index: int) -> None:
    paper_id = message.get("paper_id")
    question = message.get("question")
    citations = _citations_from_payload(message)
    if not isinstance(paper_id, str) or not isinstance(question, str) or not citations:
        return
    if st.button("保存为研究笔记", key=f"save_note_{message_index}", icon="📝"):
        research_note_service.save_answer(
            user_id=st.session_state.local_user_id,
            session_id=st.session_state.current_session,
            paper_id=paper_id,
            question=question,
            answer_text=str(message.get("content") or ""),
            citations=citations,
        )
        st.toast("已保存到当前论文的研究笔记。")


settings = build_settings()
require_ascii_chroma_path(settings.global_chroma_dir, "RESEARCHPILOT_GLOBAL_CHROMA_DIR")
require_ascii_chroma_path(settings.session_chroma_dir, "RESEARCHPILOT_SESSION_CHROMA_DIR")
ensure_runtime_directories(settings)
DEFAULT_PRODUCT_PROFILE_NAME = "high_precision"


def _render_brand_header(logo_path: Path) -> None:
    """展示裁切视窗中的横向品牌图，不改写用户提供的原始 Logo 文件。"""
    if not logo_path.is_file():
        st.title("ResearchPilot-RAG")
        return
    encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <section class="rp-hero" aria-label="ResearchPilot-RAG">
          <div class="rp-brand">
            <svg viewBox="20 350 1215 490" role="img" aria-label="ResearchPilot-RAG Logo">
              <image href="data:image/png;base64,{encoded}" x="0" y="0" width="1254" height="1254" />
            </svg>
          </div>
          <p class="rp-tagline">面向 AI/CS 论文的可追溯阅读、问答与跨论文研究助手</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


# 页面布局与品牌视觉
full_logo_path = settings.resources_dir / "full_logo.png"
brand_icon = trim_logo_image(settings.logo_path)
brand_wordmark = trim_logo_image(full_logo_path, padding_ratio=0.025)
st.set_page_config(
    page_title="ResearchPilot-RAG",
    page_icon=brand_icon or "📚",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={"About": "### ResearchPilot\n面向 AI/CS 论文的可追溯阅读与研究助手。"},
)
if brand_wordmark is not None:
    st.logo(brand_wordmark, size="large", icon_image=brand_icon or brand_wordmark)
st.markdown(
    """
    <style>
      :root { --rp-blue: #1769d2; --rp-cyan: #10bfc1; }
      .stApp { background: linear-gradient(145deg, #ffffff 0%, #f7fbff 55%, #f4fffe 100%); }
      [data-testid="stSidebar"] { border-right: 1px solid rgba(23, 105, 210, 0.12); }
      [data-testid="stSidebar"] [data-testid="stLogo"] { padding-top: .75rem; }
      [data-testid="stSidebar"] [data-testid="stLogo"] img {
        max-height: 4.6rem; width: auto; object-fit: contain;
      }
      .rp-hero { max-width: 720px; margin: .35rem 0 1.35rem; }
      .rp-brand { width: min(400px, 78vw); overflow: hidden; line-height: 0; }
      .rp-brand svg { display: block; width: 100%; height: auto; }
      .rp-brand image { mix-blend-mode: multiply; }
      .rp-tagline { margin: .55rem 0 0; color: #6b7280; font-size: .9rem; line-height: 1.55; }
      @media (max-width: 640px) {
        .main .block-container { padding-left: .8rem; padding-right: .8rem; padding-top: .8rem; }
        .rp-hero { max-width: 92vw; margin: .1rem 0 .8rem; }
        .rp-brand { width: min(310px, 90vw); }
        .rp-tagline { font-size: .84rem; }
        [data-testid="stSidebar"] { max-width: 88vw; }
        [data-testid="stChatInput"] { max-width: calc(100vw - 1rem); }
        [data-testid="stChatMessage"] { padding-left: .55rem; padding-right: .55rem; }
        [data-testid="column"] { min-width: 0 !important; }
      }
      .stButton > button { border-radius: .7rem; }
      [data-testid="stFileUploaderDropzone"] { border-color: rgba(23, 105, 210, .28); }
      [data-testid="stChatMessage"] { border-radius: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)
_render_brand_header(full_logo_path)


@st.cache_resource
def load_product_runtime():
    return build_product_runtime_collection(settings=settings, streaming=True)


product_runtime = load_product_runtime()
product_profiles = product_runtime.profiles
session_rag_services = product_runtime.services
session_file_repository = SessionFileRepository(settings)
scoped_memory_service = product_runtime.memory_service
annotation_service = product_runtime.annotation_service
research_note_service = product_runtime.research_note_service
conversation_service = product_runtime.conversation_service
product_data_maintenance = product_runtime.data_maintenance


def resolve_local_user_scope() -> str:
    """Keep an anonymous browser scope stable across a page refresh.

    A configured local owner ID is reserved for explicit local development or
    migration. Otherwise the UUID is stored in the current browser URL, so a
    reload preserves the same session namespace while a fresh/incognito visit
    to the base URL remains isolated.
    """
    if settings.local_user_id:
        return settings.local_user_id

    query_user_id = st.query_params.get("rp_user")
    if isinstance(query_user_id, str):
        try:
            settings.user_sessions_dir(query_user_id)
        except ConfigurationError:
            pass
        else:
            return query_user_id

    generated_user_id = generate_local_user_id()
    st.query_params["rp_user"] = generated_user_id
    return generated_user_id


def active_product_profile():
    """返回当前页面选择的真实 Profile；未知值安全回退到高精度模式。"""
    name = st.session_state.get("product_mode", DEFAULT_PRODUCT_PROFILE_NAME)
    return product_profiles.get(name, product_profiles[DEFAULT_PRODUCT_PROFILE_NAME])


def active_rag_service():
    """让用户选择的模式实际决定 Service，而不是仅改变页面标签。"""
    return session_rag_services[active_product_profile().name]


def selected_document() -> dict | None:
    paper_id = st.session_state.get("selected_paper_id")
    if not isinstance(paper_id, str) or not paper_id:
        return None
    for document in st.session_state.get("active_documents", []):
        if isinstance(document, dict) and document.get("paper_id") == paper_id:
            return document
    return None


def active_scope_label() -> str:
    """Return a user-facing description of the actual paper retrieval scope."""
    document = selected_document()
    if document is not None:
        return str(document.get("source_name") or "当前论文")
    if st.session_state.get("active_pdf_names"):
        return "当前会话全部论文"
    return "尚未选择论文"


def active_memory_paper_id() -> str | None:
    """仅在论文范围唯一时返回可用于跨会话记忆匹配的 paper_id。"""
    selected_paper_id = st.session_state.get("selected_paper_id")
    if isinstance(selected_paper_id, str) and selected_paper_id:
        return selected_paper_id
    paper_ids = tuple(
        dict.fromkeys(
            str(item["paper_id"])
            for item in st.session_state.get("active_documents", [])
            if isinstance(item, dict) and isinstance(item.get("paper_id"), str) and item["paper_id"]
        )
    )
    return paper_ids[0] if len(paper_ids) == 1 else None


def active_scope_paper_ids() -> tuple[str, ...]:
    selected = active_memory_paper_id()
    if selected:
        return (selected,)
    return tuple(
        str(item["paper_id"])
        for item in st.session_state.get("active_documents", [])
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str)
    )


def active_working_memory_scope_key() -> str:
    return scope_key_for_papers(active_scope_paper_ids())


def switch_working_memory_scope(new_scope_key: str) -> None:
    """Persist the previous scope before loading another paper's WM."""
    previous = st.session_state.get("working_memory_scope_key", "general")
    if previous == new_scope_key:
        return
    scoped_memory_service.save(
        user_id=st.session_state.local_user_id,
        session_id=st.session_state.current_session,
        scope_key=previous,
        memory=st.session_state.get("working_memory", empty_working_memory()),
    )
    st.session_state.working_memory = scoped_memory_service.load(
        user_id=st.session_state.local_user_id,
        session_id=st.session_state.current_session,
        scope_key=new_scope_key,
    )
    st.session_state.working_memory_scope_key = new_scope_key
    st.session_state.wm_last_update_msg_count = sum(
        1 for message in messages_in_scope(st.session_state.messages, new_scope_key) if message.get("role") == "user"
    )


# 生成会话标识函数
def generate_session_name():
    return generate_session_id()


# 状态保存与加载逻辑 (包含活跃文档名单)
def save_session():
    if st.session_state.current_session:
        user_id = st.session_state.local_user_id
        session_id = st.session_state.current_session
        session_file_repository.save(
            user_id,
            session_id,
            st.session_state.messages,
            st.session_state.active_pdf_names,
            st.session_state.active_documents,
            st.session_state.selected_paper_id,
            st.session_state.get("session_title"),
        )
        scoped_memory_service.save(
            user_id=user_id,
            session_id=session_id,
            scope_key=st.session_state.get("working_memory_scope_key", "general"),
            memory=st.session_state.working_memory,
        )


def load_sessions():
    return session_file_repository.list(st.session_state.local_user_id, st.session_state.current_session)


def load_session(session_name):
    try:
        user_id = st.session_state.local_user_id
        snapshot = session_file_repository.load(user_id, session_name)
        if snapshot is None:
            st.warning("该会话文件不存在，可能已被删除。")
            return
        st.session_state.current_session = session_name
        st.session_state.messages = snapshot.messages
        st.session_state.active_pdf_names = snapshot.active_pdf_names
        st.session_state.active_documents = snapshot.active_documents or [
            {"source_name": source_name, "legacy": True} for source_name in snapshot.active_pdf_names
        ]
        restored_paper_id = snapshot.selected_paper_id
        st.session_state.selected_paper_id = (
            restored_paper_id
            if any(item.get("paper_id") == restored_paper_id for item in st.session_state.active_documents)
            else None
        )
        st.session_state.session_title = snapshot.session_title
        scope_key = active_working_memory_scope_key()
        st.session_state.working_memory_scope_key = scope_key
        st.session_state.working_memory = scoped_memory_service.load(
            user_id=user_id,
            session_id=session_name,
            scope_key=scope_key,
        )
    except SessionDataCorrupted as error:
        st.error("会话文件已损坏，未加载该会话；原文件已保留以便审计。")
        print(f"会话 JSON 损坏 [{session_name}]：{error}")
    except Exception as error:
        st.error(f"加载会话失败：{error}")
        print(f"会话加载报错 [{session_name}]：{error}")


# 删除会话
def delete_session(session_name):
    try:
        user_id = st.session_state.local_user_id
        cleanup = product_data_maintenance.delete_session_data(
            user_id=user_id,
            session_id=session_name,
        )
        cleanup["session_json_deleted"] = session_file_repository.delete(user_id, session_name)

        print(f"会话清理完成 [{user_id}/{session_name}]：{cleanup}")

        # 5. UI 状态重置
        if session_name == st.session_state.current_session:
            st.session_state.messages = []
            st.session_state.current_session = generate_session_name()
            st.session_state.active_pdf_names = []
            st.session_state.active_documents = []
            st.session_state.selected_paper_id = None
            st.session_state.session_title = None
            st.session_state.working_memory = empty_working_memory()
            st.session_state.working_memory_scope_key = "general"

    except Exception as error:
        st.error(f"删除会话发生异常：{error}")


def remove_document_from_current_session(fname: str, document_details: dict | None) -> None:
    """Delete one confirmed paper and all same-scope derived artifacts."""
    st.session_state.active_pdf_names.remove(fname)
    st.session_state.active_documents = [
        item for item in st.session_state.active_documents if item.get("source_name") != fname
    ]
    st.session_state.selected_paper_id = None
    paper_id = (
        str(document_details["paper_id"])
        if isinstance(document_details, dict) and isinstance(document_details.get("paper_id"), str)
        else None
    )
    if paper_id is not None:
        active_rag_service().delete_paper(
            user_id=st.session_state.local_user_id,
            session_id=st.session_state.current_session,
            paper_id=paper_id,
        )
        local_pdf_path = document_details.get("local_pdf_path")
        if isinstance(local_pdf_path, str):
            resolved_pdf = Path(local_pdf_path).resolve()
            uploads_root = settings.uploads_dir.resolve()
            if uploads_root in resolved_pdf.parents:
                resolved_pdf.unlink(missing_ok=True)
    else:
        product_data_maintenance.delete_legacy_paper(
            user_id=st.session_state.local_user_id,
            session_id=st.session_state.current_session,
            source_name=fname,
        )
    switch_working_memory_scope(active_working_memory_scope_key())
    st.session_state.pending_delete_paper = None
    save_session()


# 新增：中期工作记忆 (WM) 动态更新引擎
def update_working_memory():
    try:
        scope_key = st.session_state.get("working_memory_scope_key", "general")
        result = scoped_memory_service.maybe_update_working_memory(
            user_id=st.session_state.local_user_id,
            session_id=st.session_state.current_session,
            scope_key=scope_key,
            messages=st.session_state.messages,
            current_memory=st.session_state.working_memory,
            last_update_count=st.session_state.wm_last_update_msg_count,
            conversation=conversation_service,
        )
        if result.updated:
            st.session_state.working_memory = result.memory
            st.session_state.wm_last_update_msg_count = result.user_message_count
            print(f"🧠 [WM更新成功] 核心主题: {result.memory.get('current_topic', '未知')}")
            save_session()
    except Exception as error:
        print("更新中期工作记忆失败:", error)


# 新增：长期记忆 (LTM) 跨会话结算与冲突消解
def settle_to_long_term_memory():
    if len(st.session_state.messages) < 2:
        st.warning("⚠️ 对话太少，暂无足够的信息可供结算。")
        return

    with st.spinner("🧠 正在进行跨会话记忆比对与冲突消解..."):
        try:
            scoped_memory_service.settle_long_term_memory(
                user_id=st.session_state.local_user_id,
                session_id=st.session_state.current_session,
                paper_ids=active_scope_paper_ids(),
                working_memory=st.session_state.working_memory,
                conversation=conversation_service,
            )
            st.toast("✅ 长期记忆结算成功！已完成同一论文作用域内的去重。")
        except Exception as error:
            st.error(f"结算长期记忆失败: {error}")
            print("LTM结算报错:", error)


# 新增：会话级动态总结引擎 (Session LTM Extractor)
def extract_dynamic_session_conclusions():
    scope_key = st.session_state.get("working_memory_scope_key", "general")
    last_turns = st.session_state.session_summary_last_turn_by_scope
    last_turn = int(last_turns.get(scope_key, 0))
    try:
        updated_turn = scoped_memory_service.maybe_extract_incremental_conclusions(
            user_id=st.session_state.local_user_id,
            session_id=st.session_state.current_session,
            paper_ids=active_scope_paper_ids(),
            scope_key=scope_key,
            messages=st.session_state.messages,
            last_turn=last_turn,
            conversation=conversation_service,
        )
        if updated_turn != last_turn:
            last_turns[scope_key] = updated_turn
            print("✅ 阶段性结论已存入同一论文作用域的 LTM 数据库！")
    except Exception as error:
        print("动态会话总结失败:", error)


# 状态初始化区 (修复：补充缺失变量，删除冗余)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "local_user_id" not in st.session_state:
    st.session_state.local_user_id = resolve_local_user_scope()
if "current_session" not in st.session_state:
    st.session_state.current_session = generate_session_name()
if "active_pdf_names" not in st.session_state:
    st.session_state.active_pdf_names = []
if "active_documents" not in st.session_state:
    st.session_state.active_documents = []
if "failed_pdf_names" not in st.session_state:
    st.session_state.failed_pdf_names = []
if "product_mode" not in st.session_state:
    st.session_state.product_mode = DEFAULT_PRODUCT_PROFILE_NAME
if "annotation_mode" not in st.session_state:
    st.session_state.annotation_mode = "paper_only"
if "selected_paper_id" not in st.session_state:
    st.session_state.selected_paper_id = None
if "session_title" not in st.session_state:
    st.session_state.session_title = None

# 中期工作记忆 (Working Memory) 状态初始化
if "working_memory" not in st.session_state:
    st.session_state.working_memory = empty_working_memory()
if "working_memory_scope_key" not in st.session_state:
    st.session_state.working_memory_scope_key = "general"

# 触发器锚点记录
if "wm_last_update_msg_count" not in st.session_state:
    st.session_state.wm_last_update_msg_count = 0
if "session_summary_last_turn" not in st.session_state:
    st.session_state.session_summary_last_turn = 0
if "session_summary_last_turn_by_scope" not in st.session_state:
    st.session_state.session_summary_last_turn_by_scope = {}
if "last_hit_parent_ids" not in st.session_state:
    st.session_state.last_hit_parent_ids = []
if "last_evidence_anchors" not in st.session_state:
    st.session_state.last_evidence_anchors = []


# 展示聊天信息。内部 session_id 保持稳定；页面只暴露最少的技术标识。
st.caption(f"当前会话 · {st.session_state.session_title or st.session_state.current_session[:8]}")
for message_index, messages in enumerate(st.session_state.messages):
    if messages["role"] == "system":
        st.chat_message("system").write(messages["content"])
    elif messages.get("status") == "failed":
        with st.chat_message("assistant"):
            st.error(messages["content"])
    else:
        st.chat_message(messages["role"]).write(messages["content"])
    if messages["role"] == "assistant":
        render_paper_evidence(messages)
        render_save_note_action(messages, message_index)

pending_retry_question = st.session_state.get("pending_retry_question")
if isinstance(pending_retry_question, str) and pending_retry_question.strip():
    retry_col, dismiss_col = st.columns([1, 4])
    if retry_col.button("重试上一问题", type="primary"):
        st.session_state.pending_question = st.session_state.pop("pending_retry_question")
        st.rerun()
    if dismiss_col.button("暂不重试"):
        st.session_state.pop("pending_retry_question", None)
        st.rerun()

# 侧边栏设置角色
with st.sidebar:
    st.subheader("研究工作区")
    mode_labels = {
        "high_precision": "精确问答 · 方法、实验与页码证据",
        "standard": "快速问答 · 较低检索开销",
    }
    selected_mode = st.radio(
        "回答模式",
        options=("high_precision", "standard"),
        format_func=lambda value: mode_labels[value],
        index=(0 if st.session_state.product_mode == "high_precision" else 1),
        help="模式会真实改变后端是否运行 BGE Reranker；全文精读和严格跨论文模式仍处于实验阶段。",
    )
    st.session_state.product_mode = selected_mode
    current_profile = active_product_profile()
    st.caption(f"当前模式：{current_profile.label}")
    with st.expander("当前运行配置", expanded=False):
        st.caption(f"解析：{current_profile.ingestion.parser_name} {current_profile.ingestion.parser_version}")
        st.caption(f"分块：{current_profile.ingestion.chunker_version}")
        st.caption(f"检索：Dense + BM25 + {current_profile.retrieval.fusion.upper()}")
        st.caption("重排：BGE Reranker" if current_profile.retrieval.reranker_enabled else "重排：未启用")
        st.caption(f"配置版本：{current_profile.config_hash}")
    st.session_state.annotation_mode = st.radio(
        "用户笔记",
        options=("paper_only", "with_user_notes"),
        format_func=lambda value: ("仅按论文原文回答" if value == "paper_only" else "并列参考我的批注"),
        horizontal=True,
        help="批注始终标记为用户观点；即使启用，也不会静默覆盖论文原文。",
    )
    # 会话信息
    if st.button("新建会话", width="stretch", icon="✏️"):
        if st.session_state.messages:
            with st.spinner("正在后台为您自动保存全局记忆..."):
                settle_to_long_term_memory()
            # 优化：只有旧会话有内容时，才需要保存旧会话状态
            save_session()

            # 创建新的干净会话
            st.session_state.messages = []
            st.session_state.current_session = generate_session_name()
            st.session_state.active_pdf_names = []
            st.session_state.active_documents = []
            st.session_state.selected_paper_id = None
            st.session_state.session_title = None
            st.session_state.working_memory = empty_working_memory()
            st.session_state.working_memory_scope_key = "general"
            # 删除原有的第二次 save_session()，新会话不需要立刻存盘，等用户发消息后再存即可
            st.rerun()

    # 历史会话列表
    st.text("历史会话")
    session_list = load_sessions()
    for session in session_list:
        col1, col2 = st.columns([4, 1])
        with col1:
            snapshot = session_file_repository.load(st.session_state.local_user_id, session)
            display_title = snapshot.session_title if snapshot and snapshot.session_title else session
            button_type = "primary" if session == st.session_state.current_session else "secondary"
            if col1.button(
                display_title,
                width="stretch",
                icon="💾",
                key=f"load{session}",
                type=button_type,
            ):
                # 【新增】：在切走之前，先自动保存【当前会话】的长期记忆
                if session != st.session_state.current_session and st.session_state.messages:
                    with st.spinner("正在后台同步全局记忆..."):
                        settle_to_long_term_memory()

                load_session(session)
                st.rerun()
        with col2:
            if col2.button("", width="stretch", icon="❌️", key=f"delete_{session}"):
                delete_session(session)
                st.rerun()

    # 侧边栏：文档与记忆控制台
    st.text("📚 当前研究论文")
    scoped_documents = [
        item
        for item in st.session_state.active_documents
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str)
    ]
    if scoped_documents:
        scope_labels = {"全部当前会话论文": None}
        scope_labels.update({str(item["source_name"]): str(item["paper_id"]) for item in scoped_documents})
        scope_options = tuple(scope_labels)
        restored_scope_index = next(
            (
                index
                for index, value in enumerate(scope_options)
                if scope_labels[value] == st.session_state.selected_paper_id
            ),
            0,
        )
        selected_scope_label = st.selectbox(
            "回答范围",
            options=scope_options,
            index=restored_scope_index,
            help="正文检索仅在选定论文内进行；未选择时会检索当前会话全部论文。",
            key=f"paper_scope_{st.session_state.current_session}",
        )
        previous_scope_key = st.session_state.get("working_memory_scope_key", "general")
        st.session_state.selected_paper_id = scope_labels[selected_scope_label]
        new_scope_key = active_working_memory_scope_key()
        if previous_scope_key != new_scope_key:
            switch_working_memory_scope(new_scope_key)
        export_paper_id = active_memory_paper_id()
        if export_paper_id:
            paper_document = next(
                (item for item in scoped_documents if str(item.get("paper_id")) == export_paper_id),
                None,
            )
            notes = research_note_service.list_notes(
                user_id=st.session_state.local_user_id,
                session_id=st.session_state.current_session,
                paper_id=export_paper_id,
            )
            with st.expander(f"📝 研究笔记（{len(notes)}）", expanded=False):
                if not notes:
                    st.caption("尚未保存笔记。可在带 Citation 的回答下点击“保存为研究笔记”。")
                for note_index, note in enumerate(notes, start=1):
                    st.markdown(f"**{note_index}. {note.question}**")
                    answer_preview = note.answer_text.strip()
                    if len(answer_preview) > 240:
                        answer_preview = answer_preview[:240].rstrip() + "…"
                    st.caption(answer_preview)
                    for citation_index, citation in enumerate(note.citations, start=1):
                        page_label = (
                            f"第 {citation.page_start} 页"
                            if citation.page_start == citation.page_end
                            else f"第 {citation.page_start}-{citation.page_end} 页"
                        )
                        st.caption(f"证据 {citation_index} · {page_label} · {citation.chunk_id}")
                        local_pdf_path = (
                            paper_document.get("local_pdf_path") if isinstance(paper_document, dict) else None
                        )
                        if isinstance(local_pdf_path, str) and Path(local_pdf_path).is_file():
                            if st.button(
                                f"查看笔记证据第 {citation.page_start} 页",
                                key=f"note_page_{note.note_id}_{citation_index}",
                            ):
                                try:
                                    page_image = render_pdf_page_png(Path(local_pdf_path), citation.page_start)
                                    st.image(page_image, caption=page_label)
                                except Exception as error:
                                    st.warning(f"暂时无法渲染该页：{error}")
                    if st.button("删除这条笔记", key=f"delete_note_{note.note_id}"):
                        research_note_service.delete_note(
                            user_id=st.session_state.local_user_id,
                            session_id=st.session_state.current_session,
                            paper_id=export_paper_id,
                            note_id=note.note_id,
                        )
                        st.toast("研究笔记已删除。")
                        st.rerun()
                    if note_index != len(notes):
                        st.divider()
            exported_notes = research_note_service.export_markdown(
                user_id=st.session_state.local_user_id,
                session_id=st.session_state.current_session,
                paper_id=export_paper_id,
            )
            st.download_button(
                "导出当前论文阅读记录",
                data=exported_notes.encode("utf-8"),
                file_name=f"researchpilot_{export_paper_id}.md",
                mime="text/markdown",
                width="stretch",
                icon="📤",
            )
    elif st.session_state.active_pdf_names:
        st.caption("当前会话含旧版索引；请重新上传论文后再使用单论文范围筛选。")
        st.session_state.selected_paper_id = None
    uploader_key = f"pdf_uploader_{st.session_state.current_session}"
    uploaded_files = st.file_uploader("拖拽上传新PDF", type=["pdf"], accept_multiple_files=True, key=uploader_key)

    # 1. 处理新上传的文档
    if uploaded_files:
        # 先筛选出真正“新上传”的文件
        # 👇 先筛选出真正“新上传”的文件（排除已解析的白名单，和解析失败的黑名单）
        new_files = [
            file
            for file in uploaded_files
            if file.name not in st.session_state.active_pdf_names and file.name not in st.session_state.failed_pdf_names
        ]

        if new_files:
            upload_status = st.status("正在处理上传论文…", expanded=True)
            for file in new_files:
                try:
                    upload_status.write(f"✓ 已接收 {file.name}")
                    upload_status.write("● 正在进行结构解析、生成分块并写入向量索引")
                    pdf_bytes = file.getvalue()
                    ingestion_result = active_rag_service().ingest_pdf(
                        pdf_bytes,
                        source_name=file.name,
                        user_id=st.session_state.local_user_id,
                        session_id=st.session_state.current_session,
                    )
                except PdfIngestionError as exc:
                    upload_status.write(f"✗ {file.name} 未建立索引：文档无法提取足够正文")
                    st.error(
                        f"未能解析《{file.name}》，本次没有写入空索引。可能是扫描件、加密 PDF 或文本层损坏。详情：{exc}"
                    )
                    if file.name in st.session_state.active_pdf_names:
                        st.session_state.active_pdf_names.remove(file.name)
                    # 解析失败文件进入黑名单，避免 Streamlit rerun 后无限重复解析。
                    if file.name not in st.session_state.failed_pdf_names:
                        st.session_state.failed_pdf_names.append(file.name)
                    continue

                quality = ingestion_result.parse_quality
                local_pdf_path = persist_pdf_copy(
                    settings=settings,
                    user_id=st.session_state.local_user_id,
                    session_id=st.session_state.current_session,
                    paper_id=ingestion_result.paper.paper_id,
                    pdf_bytes=pdf_bytes,
                )
                actual_parser = quality.parser_name if quality else "未知解析器"
                upload_status.write(
                    f"✓ 结构解析完成：{actual_parser}；"
                    f"Parent {len(ingestion_result.parent_chunks)} 个，"
                    f"Child {len(ingestion_result.child_chunks)} 个"
                )

                if file.name not in st.session_state.active_pdf_names:
                    st.session_state.active_pdf_names.append(file.name)
                st.session_state.active_documents = [
                    item for item in st.session_state.active_documents if item.get("source_name") != file.name
                ]
                st.session_state.active_documents.append(
                    {
                        "source_name": file.name,
                        "paper_id": ingestion_result.paper.paper_id,
                        "page_count": ingestion_result.paper.page_count,
                        "parser_name": quality.parser_name if quality else None,
                        "parser_version": quality.parser_version if quality else None,
                        "requested_parser_name": ingestion_result.requested_parser_name,
                        "chunker_version": active_product_profile().ingestion.chunker_version,
                        "product_profile": active_product_profile().name,
                        "product_profile_hash": active_product_profile().config_hash,
                        "fallback_used": ingestion_result.fallback_used,
                        "extraction_quality": (quality.extraction_quality.value if quality else "unknown"),
                        "enrichment_status": "on_demand",
                        "local_pdf_path": str(local_pdf_path),
                    }
                )
                switch_working_memory_scope(active_working_memory_scope_key())
                # 上传只负责确定性的解析与索引。作者和全文摘要在用户首次提问时，
                # 由可追溯的首页/章节证据按需生成；DeepSeek 暂时断网不会再破坏上传。
                save_session()
                upload_status.write("✓ 论文已可检索；元信息与全文摘要将按需生成并附页码证据")

            upload_status.update(label="论文处理完成", state="complete", expanded=False)
            save_session()
            st.rerun()

    # 2. 渲染当前会话的文档菜单列表 (解决问题2 和 问题3)
    if st.session_state.active_pdf_names:
        st.caption("已关联至当前对话的文档：")
        for fname in st.session_state.active_pdf_names.copy():
            # 使用 Streamlit 容器打造优雅菜单
            container = st.container(border=True)
            col1, col2 = container.columns([7, 3])
            col1.markdown(f"📄 **{fname[:15]}...**")
            document_details = next(
                (item for item in st.session_state.active_documents if item.get("source_name") == fname),
                None,
            )
            if isinstance(document_details, dict):
                parser_name = document_details.get("parser_name") or "未知"
                chunker_name = document_details.get("chunker_version") or "旧版索引"
                quality = document_details.get("extraction_quality") or "未知"
                col1.caption(f"{parser_name} · {chunker_name} · 解析质量：{quality}")
                if document_details.get("fallback_used"):
                    col1.caption("已从 Docling 回退到 pdfplumber 兼容解析。")
            else:
                col1.caption("旧版索引：建议重新上传以启用结构化解析与单论文范围筛选。")

            # 按钮区
            sub_col1, sub_col2 = container.columns(2)
            if sub_col1.button("📥 归档", key=f"arc_{fname}", help="按需生成带证据摘要并归档"):
                paper_id = (
                    str(document_details["paper_id"])
                    if isinstance(document_details, dict) and isinstance(document_details.get("paper_id"), str)
                    else None
                )
                if paper_id is None:
                    st.warning("旧版论文缺少 paper_id；请重新上传后再归档。")
                else:
                    with st.spinner("正在基于全文章节证据生成归档摘要..."):
                        try:
                            summary_execution = active_rag_service().answer_with_retrieval(
                                question=(
                                    "请总结这篇论文的研究问题、核心方法、实验与主要结果、"
                                    "结论和局限，并为每部分标明证据编号。"
                                ),
                                user_id=st.session_state.local_user_id,
                                session_id=st.session_state.current_session,
                                query_type=QueryType.DOCUMENT_SUMMARY,
                                paper_id=paper_id,
                            )
                            if summary_execution.answer.refused:
                                st.warning("当前论文证据不足，暂未归档；你可以稍后重试。")
                            else:
                                summary_text = summary_execution.answer.text
                                product_data_maintenance.archive_summary(
                                    user_id=st.session_state.local_user_id,
                                    session_id=st.session_state.current_session,
                                    paper_id=paper_id,
                                    source_name=fname,
                                    summary_text=summary_text,
                                )
                                st.toast(f"✅ {fname} 已生成章节覆盖摘要并完成归档！")
                        except Exception as error:
                            print("论文摘要归档失败：", error)
                            st.error("论文仍可正常检索，但摘要生成暂时失败。请检查 DeepSeek 网络连接后再次点击“归档”。")

            if sub_col2.button("❌ 移除", key=f"del_{fname}", help="移除前会显示关联数据并二次确认"):
                current_uploaded_names = [f.name for f in uploaded_files] if uploaded_files else []
                if fname in current_uploaded_names:
                    st.warning(f"请先在上传框中点击 '{fname}' 旁边的 ×，再移除论文。")
                else:
                    st.session_state.pending_delete_paper = fname

            if st.session_state.get("pending_delete_paper") == fname:
                paper_id = document_details.get("paper_id") if isinstance(document_details, dict) else None
                counts = (
                    product_data_maintenance.paper_related_counts(
                        user_id=st.session_state.local_user_id,
                        session_id=st.session_state.current_session,
                        paper_id=str(paper_id),
                    )
                    if isinstance(paper_id, str)
                    else {"annotations": 0, "research_notes": 0, "long_term_memories": 0}
                )
                st.warning(
                    "确认移除？将同步删除该论文索引、PDF 副本、"
                    f"{counts['research_notes']} 条笔记、{counts['annotations']} 条批注和"
                    f" {counts['long_term_memories']} 条当前会话论文记忆。"
                )
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("确认移除", key=f"confirm_del_{fname}", type="primary"):
                    remove_document_from_current_session(fname, document_details)
                    st.rerun()
                if cancel_col.button("取消", key=f"cancel_del_{fname}"):
                    st.session_state.pending_delete_paper = None
                    st.rerun()

    # 3. 全局长期记忆管理台 (解决问题2)
    # 【核心修复】：注意这里的缩进！将它移出上面的 if 判断，保证始终可见！

    st.markdown("---")
    with st.expander("🗄️ 全局长期记忆管理台"):
        try:
            # 动态查询 Chroma 库里有哪些归档摘要
            archived_files = product_data_maintenance.list_archived_sources(user_id=st.session_state.local_user_id)
            if archived_files:
                for src in archived_files:
                    colA, colB = st.columns([7, 3])
                    colA.markdown(f"💾 {src[:12]}...")
                    if colB.button("删除", key=f"gdel_{src}"):
                        product_data_maintenance.delete_archive(
                            user_id=st.session_state.local_user_id,
                            source_name=src,
                        )
                        st.rerun()
            else:
                st.caption("暂无归档记忆。")
        except Exception as e:
            st.caption("暂无归档记忆或读取为空。")
            print("全局记忆台底层日志:", e)  # 将真实报错打印到终端，方便后续排查
# 论文就绪后的第一轮建议问题不调用模型。点击后会作为下一条用户问题提交，
# 避免把技术模板或额外 API 成本藏在上传流程中。
if st.session_state.active_pdf_names and not st.session_state.messages:
    st.markdown("#### 从论文阅读开始")
    st.caption(f"当前回答范围：{active_scope_label()} · {active_product_profile().label}")
    suggested_questions = (
        "这篇论文解决了什么问题？",
        "核心方法由哪些模块组成？",
        "使用了哪些数据集和 Baseline？",
        "主要实验结果是什么？",
        "作者提到了哪些局限性？",
    )
    suggestion_columns = st.columns(2)
    for index, question in enumerate(suggested_questions):
        if suggestion_columns[index % 2].button(question, key=f"suggestion_{index}"):
            st.session_state.pending_question = question
            st.rerun()

if st.session_state.active_pdf_names:
    st.caption(f"回答范围：{active_scope_label()} · 回答模式：{active_product_profile().label}")
    legacy_status = LegacySessionAdapter.inspect(st.session_state.active_documents)
    if legacy_status.has_legacy_documents:
        st.warning(legacy_status.banner_text)
else:
    st.info("从左侧上传一篇 AI/CS 论文；论文完成解析后即可获得带页码证据的问答。")


def handle_product_prompt(user_prompt: str) -> None:
    """Thin UI adapter around the shared product RuntimeOrchestrator."""
    st.chat_message("user").write(user_prompt)
    print("->提示词：", user_prompt)
    scope_key = active_working_memory_scope_key()

    user_message = {
        "role": "user",
        "content": user_prompt,
        "paper_scope_key": scope_key,
        "status": "pending",
        "eligible_for_memory": False,
    }
    st.session_state.messages.append(user_message)
    recent_history = format_recent_history(st.session_state.messages[:-1], scope_key, limit=6, max_chars=300)
    runtime_papers = tuple(
        RuntimePaper(
            paper_id=str(item["paper_id"]),
            source_name=str(item["source_name"]),
            page_count=int(item["page_count"]) if isinstance(item.get("page_count"), int) else None,
            parser_name=str(item["parser_name"]) if isinstance(item.get("parser_name"), str) else None,
            chunker_version=(str(item["chunker_version"]) if isinstance(item.get("chunker_version"), str) else None),
            extraction_quality=(
                str(item["extraction_quality"]) if isinstance(item.get("extraction_quality"), str) else None
            ),
        )
        for item in st.session_state.active_documents
        if isinstance(item, dict) and isinstance(item.get("paper_id"), str) and isinstance(item.get("source_name"), str)
    )
    legacy_status = LegacySessionAdapter.inspect(st.session_state.active_documents)
    stream_state = {"text": "", "placeholder": None}

    def render_delta(delta: str) -> None:
        if not delta:
            return
        stream_state["text"] += delta
        if stream_state["placeholder"] is None:
            with st.chat_message("assistant"):
                stream_state["placeholder"] = st.empty()
        stream_state["placeholder"].markdown(stream_state["text"] + "▌")

    runtime_result = None
    if not runtime_papers and legacy_status.requires_reupload:
        response_text = legacy_status.answer_text
        with st.chat_message("assistant"):
            st.markdown(response_text)
        route = None
    else:
        request = RuntimeRequest(
            question=user_prompt,
            user_id=st.session_state.local_user_id,
            session_id=st.session_state.current_session,
            available_papers=runtime_papers,
            selected_paper_ids=((st.session_state.selected_paper_id,) if st.session_state.selected_paper_id else ()),
            recent_history=recent_history,
            product_mode=active_product_profile().name,
            last_evidence_chunk_ids=tuple(st.session_state.last_hit_parent_ids),
            last_evidence_anchors=tuple(
                RuntimeEvidenceAnchor(
                    chunk_id=str(item["chunk_id"]),
                    page=(int(item["page"]) if item.get("page") is not None else None),
                    section_path=tuple(item.get("section_path") or ()),
                )
                for item in st.session_state.last_evidence_anchors
            ),
        )
        orchestrator = RuntimeOrchestrator(
            service=active_rag_service(),
            router=RuntimeRouter(),
            conversation_service=conversation_service,
            memory_service=scoped_memory_service,
            annotation_service=annotation_service,
            annotation_mode=st.session_state.annotation_mode,
            strict_crossdoc_enabled=False,
        )
        try:
            with st.spinner("正在理解问题、检索证据并生成回答…"):
                runtime_result = orchestrator.handle(request, on_text_delta=render_delta)
            response_text = format_citation_markers_with_pages(
                runtime_result.answer.text,
                runtime_result.answer.citations,
            )
            route = runtime_result.decision.route
            if runtime_result.annotation_saved:
                st.toast("已保存为用户批注；它不会替代论文原文。")
            if stream_state["placeholder"] is None:
                with st.chat_message("assistant"):
                    st.markdown(response_text)
            else:
                # Claim/数字契约可能对流式草稿做最终收口，因此以最终文本校准。
                stream_state["placeholder"].markdown(response_text)
        except Exception as error:
            if stream_state["placeholder"] is not None:
                stream_state["placeholder"].empty()
            response_text = (
                "当前模型或检索服务暂时不可用。论文索引仍然保留，请检查网络与 DeepSeek 服务后重试，无需重新上传论文。"
            )
            route = None
            with st.chat_message("assistant"):
                st.error(response_text)
            print("RuntimeOrchestrator 请求失败：", error)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response_text,
                    "paper_scope_key": scope_key,
                    "question": user_prompt,
                    "status": "failed",
                    "eligible_for_memory": False,
                    "error_type": type(error).__name__,
                }
            )
            st.session_state.pending_retry_question = user_prompt
            save_session()
            st.rerun()
            return

    paper_evidence_payload = None
    answer_paper_id = active_memory_paper_id()
    if runtime_result is not None and runtime_result.retrieval is not None:
        paper_evidence_payload = build_paper_evidence_payload(
            runtime_result.answer,
            runtime_result.retrieval.trace,
            route=runtime_result.decision.route,
        )
        render_paper_evidence({"paper_evidence": paper_evidence_payload})
        st.session_state.last_hit_parent_ids = [citation.chunk_id for citation in runtime_result.answer.citations]
        st.session_state.last_evidence_anchors = [
            {
                "chunk_id": citation.chunk_id,
                "page": citation.page_start,
                "section_path": list(citation.section_path),
            }
            for citation in runtime_result.answer.citations
        ]
    else:
        st.session_state.last_hit_parent_ids = []
        st.session_state.last_evidence_anchors = []

    assistant_message = {
        "role": "assistant",
        "content": response_text,
        "paper_scope_key": scope_key,
        "question": user_prompt,
        "status": "completed",
        "eligible_for_memory": bool(runtime_result is None or not runtime_result.answer.refused),
    }
    if answer_paper_id:
        assistant_message["paper_id"] = answer_paper_id
    if paper_evidence_payload is not None:
        assistant_message["paper_evidence"] = paper_evidence_payload
    st.session_state.messages.append(assistant_message)
    user_message["status"] = "completed"
    user_message["eligible_for_memory"] = True
    st.session_state.pop("pending_retry_question", None)

    if not st.session_state.session_title:
        source_name = (
            str(selected_document().get("source_name"))
            if selected_document() is not None
            else (runtime_papers[0].source_name if len(runtime_papers) == 1 else None)
        )
        st.session_state.session_title = derive_session_title(
            source_name=source_name,
            route=route,
        )

    update_working_memory()
    extract_dynamic_session_conclusions()
    save_session()
    st.rerun()


# 消息输入框。建议问题优先于空输入框，保证点击建议后立即走同一条问答链路。
prompt = st.session_state.pop("pending_question", None) or st.chat_input("请输入你的问题")
if prompt:
    handle_product_prompt(prompt)
