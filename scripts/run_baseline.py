r"""从命令行运行 ResearchPilot Baseline，并写入可评测 JSONL 记录。

示例：
    .venv\Scripts\python.exe scripts\run_baseline.py `
      --user-id <浏览器地址中的 rp_user> `
      --session-id <session_id> `
      --question "这篇论文的核心方法是什么？"

脚本不会修改论文、会话或索引；仅查询当前用户、当前会话的既有索引，并向忽略目录
``data/runtime/runs/`` 追加一条 JSONL 运行记录。
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.config import (  # noqa: E402
    ConfigurationError,
    Settings,
    build_settings,
    ensure_runtime_directories,
    require_deepseek_api_key,
    validate_storage_id,
)
from researchpilot.generation import (  # noqa: E402
    AnswerGenerator,
    GenerationContext,
    LangChainTextGenerationClient,
)
from researchpilot.index import ChromaVectorIndexRepository  # noqa: E402
from researchpilot.retrieval import (  # noqa: E402
    HybridRetriever,
    LangChainQueryExpander,
    RetrievalService,
    SentenceTransformerReranker,
)
from researchpilot.schemas import to_jsonable  # noqa: E402
from researchpilot.services import AnswerExecution, ResearchPilotService  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 ResearchPilot Hybrid RAG Baseline。")
    parser.add_argument("--question", required=True, help="待回答的论文问题。")
    parser.add_argument("--session-id", required=True, help="要读取的已存在会话 ID。")
    parser.add_argument(
        "--user-id",
        help="浏览器地址 rp_user 参数中的用户范围；未提供时尝试 RESEARCHPILOT_LOCAL_USER_ID。",
    )
    parser.add_argument(
        "--search-query",
        help="可选：已完成指代消解的独立检索问题；默认与 --question 相同。",
    )
    parser.add_argument(
        "--recent-history",
        default="",
        help="可选：少量近期对话，用于查询扩展中的指代消解。",
    )
    return parser.parse_args(argv)


def resolve_user_id(*, requested_user_id: str | None, settings: Settings) -> str:
    user_id = (requested_user_id or settings.local_user_id or "").strip()
    if not user_id:
        raise ConfigurationError(
            "缺少 user_id。请传入 --user-id（浏览器地址中的 rp_user），"
            "或在本地 .env 中设置 RESEARCHPILOT_LOCAL_USER_ID。"
        )
    validate_storage_id(user_id, "user_id")
    return user_id


def build_service(settings: Settings) -> ResearchPilotService:
    """构建与 Streamlit 相同的 Hybrid RAG 组件；仅在实际执行脚本时调用。"""
    # 大模型和 CrossEncoder 仅在用户实际执行 CLI 时导入，单元测试不会因此加载模型。
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_openai import ChatOpenAI
    from sentence_transformers import CrossEncoder

    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    session_vdb = Chroma(
        collection_name=settings.session_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.session_chroma_dir),
    )
    llm = ChatOpenAI(
        api_key=require_deepseek_api_key(settings),
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        streaming=False,
    )
    reranker_model = CrossEncoder("BAAI/bge-reranker-base")
    index_repository = ChromaVectorIndexRepository(session_vdb)
    return ResearchPilotService(
        index_repository=index_repository,
        answer_generator=AnswerGenerator(LangChainTextGenerationClient(llm)),
        retrieval_service=RetrievalService(
            HybridRetriever(
                index_repository,
                query_expander=LangChainQueryExpander(llm),
                reranker=SentenceTransformerReranker(reranker_model),
            )
        ),
    )


def build_run_record(
    *,
    execution: AnswerExecution,
    question: str,
    search_query: str,
    user_id: str,
    session_id: str,
) -> dict[str, Any]:
    """构造可由后续 evaluation.py 读取的单条 JSONL 记录。"""
    trace = execution.retrieval.trace
    return {
        "schema_version": "baseline_run_v1",
        "run_id": f"run_{uuid.uuid4().hex}",
        "created_at": datetime.now(UTC).isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "question": question,
        "search_query": search_query,
        "retrieval": to_jsonable(trace),
        "answer": to_jsonable(execution.answer),
    }


def write_jsonl_record(*, runs_dir: Path, record: dict[str, Any]) -> Path:
    """向按日期命名的运行文件追加一条 UTF-8 JSONL 记录。"""
    runs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"baseline_{datetime.now().strftime('%Y%m%d')}.jsonl"
    output_path = runs_dir / filename
    with output_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return output_path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = build_settings()
    ensure_runtime_directories(settings)
    user_id = resolve_user_id(requested_user_id=args.user_id, settings=settings)
    validate_storage_id(args.session_id, "session_id")
    service = build_service(settings)
    search_query = (args.search_query or args.question).strip()
    execution = service.answer_with_retrieval(
        question=args.question,
        search_query=search_query,
        recent_history=args.recent_history,
        user_id=user_id,
        session_id=args.session_id,
        context=GenerationContext(chat_history=args.recent_history),
    )
    record = build_run_record(
        execution=execution,
        question=args.question,
        search_query=search_query,
        user_id=user_id,
        session_id=args.session_id,
    )
    output_path = write_jsonl_record(runs_dir=settings.runs_dir, record=record)
    print(f"已写入 Baseline 记录：{output_path}")
    print(f"trace_id：{execution.answer.trace_id}；引用数：{len(execution.answer.citations)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
