"""产品运行时的唯一装配入口。

本模块只负责把配置、模型适配器和 Repository 组合为 ``ResearchPilotService``；
它不包含 Streamlit 状态、FastAPI 请求或实验 Gold 标注。两个产品入口通过它使用
相同的 E5 Parser/Chunker 与 RRF 检索配置。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .annotation_service import AnnotationService
from .config import Settings, ensure_runtime_directories, require_ascii_chroma_path, require_deepseek_api_key
from .conversation_service import ConversationService
from .generation import AnswerGenerator, LangChainTextGenerationClient, PaperMetadataExtractor
from .index import ChromaVectorIndexRepository, VectorIndexRepository
from .memory_service import ScopedMemoryService
from .product_config import ProductProfile, load_product_profile
from .product_retrieval import ProductTaskRetrievalService
from .product_storage import ProductDataMaintenance, build_product_vectorstores
from .research_workflow import ResearchNoteService
from .retrieval import (
    JiebaTokenizer,
    LangChainQueryExpander,
    PersistentHybridRetriever,
    SentenceTransformerReranker,
)
from .services import ResearchPilotService
from .sparse import PersistentBM25Repository
from .storage import SQLiteMemoryRepository


@dataclass(frozen=True)
class ProductRuntime:
    """产品入口需要共享的服务和其已冻结的 Profile。"""

    profile: ProductProfile
    service: ResearchPilotService
    conversation_service: ConversationService
    memory_service: ScopedMemoryService
    annotation_service: AnnotationService


@dataclass(frozen=True)
class ProductRuntimeCollection:
    """Shared product dependencies for a UI that offers multiple profiles."""

    profiles: Mapping[str, ProductProfile]
    services: Mapping[str, ResearchPilotService]
    conversation_service: ConversationService
    memory_service: ScopedMemoryService
    annotation_service: AnnotationService
    research_note_service: ResearchNoteService
    data_maintenance: ProductDataMaintenance


def build_service_from_components(
    *,
    profile: ProductProfile,
    index_repository: VectorIndexRepository,
    sparse_repository: PersistentBM25Repository,
    llm: object,
    reranker_model: object | None = None,
    artifact_repository: SQLiteMemoryRepository | None = None,
    artifact_model_name: str = "unknown",
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
) -> ResearchPilotService:
    """用已创建的依赖装配服务，便于 Streamlit 缓存资源和单元测试复用。"""
    if profile.retrieval.reranker_enabled and reranker_model is None:
        raise ValueError(f"产品模式 {profile.name} 需要 BGE Reranker，但没有提供 reranker_model。")
    query_expander = LangChainQueryExpander(llm) if profile.retrieval.query_expansion else None
    reranker = (
        SentenceTransformerReranker(reranker_model)
        if profile.retrieval.reranker_enabled and reranker_model is not None
        else None
    )
    retriever = PersistentHybridRetriever(
        index_repository,
        sparse_repository,
        fusion=profile.retrieval.fusion,
        query_expander=query_expander,
        reranker=reranker,
        dense_limit_per_query=profile.retrieval.dense_limit_per_query,
        bm25_limit_per_query=profile.retrieval.bm25_limit_per_query,
        result_limit=profile.retrieval.result_limit,
        rrf_k=profile.retrieval.rrf_k,
    )
    retrieval = ProductTaskRetrievalService(
        retriever,
        index=index_repository,
        result_limit=profile.retrieval.result_limit,
    )
    generation_client = LangChainTextGenerationClient(
        llm,
        input_cost_per_million=input_cost_per_million,
        output_cost_per_million=output_cost_per_million,
    )
    return ResearchPilotService(
        index_repository=index_repository,
        answer_generator=AnswerGenerator(generation_client),
        ingestion_config=profile.ingestion,
        retrieval_service=retrieval,
        artifact_repository=artifact_repository,
        metadata_extractor=PaperMetadataExtractor(generation_client),
        artifact_model_name=artifact_model_name,
        annotation_service=(AnnotationService(artifact_repository) if artifact_repository is not None else None),
    )


def build_product_runtime(
    *,
    settings: Settings,
    profile_name: str = "high_precision",
    llm: object | None = None,
    vectorstore: object | None = None,
    reranker_model: object | None = None,
) -> ProductRuntime:
    """延迟加载真实依赖，构建 Streamlit/API 都可使用的产品运行时。"""
    profile = load_product_profile(project_root=settings.project_root, profile_name=profile_name)
    ensure_runtime_directories(settings)
    require_ascii_chroma_path(settings.session_chroma_dir, "RESEARCHPILOT_SESSION_CHROMA_DIR")

    if llm is None or vectorstore is None:
        from langchain_chroma import Chroma
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI

        if llm is None:
            llm = ChatOpenAI(
                api_key=require_deepseek_api_key(settings),
                base_url=settings.deepseek_base_url,
                model=settings.deepseek_model,
                streaming=False,
            )
        if vectorstore is None:
            embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
            vectorstore = Chroma(
                collection_name=settings.session_collection_name,
                embedding_function=embeddings,
                persist_directory=str(settings.session_chroma_dir),
            )

    if profile.retrieval.reranker_enabled and reranker_model is None:
        from sentence_transformers import CrossEncoder

        reranker_model = CrossEncoder("BAAI/bge-reranker-base")

    index_repository = ChromaVectorIndexRepository(vectorstore)  # type: ignore[arg-type]
    artifact_repository = SQLiteMemoryRepository(settings)
    artifact_repository.initialize_schema()
    sparse_repository = PersistentBM25Repository(
        index_repository,
        root_dir=settings.sparse_index_dir,
        tokenizer=JiebaTokenizer(),
    )
    generation_client = LangChainTextGenerationClient(
        llm,
        input_cost_per_million=settings.deepseek_input_cost_per_million,
        output_cost_per_million=settings.deepseek_output_cost_per_million,
    )
    annotation_service = AnnotationService(artifact_repository)
    return ProductRuntime(
        profile=profile,
        service=build_service_from_components(
            profile=profile,
            index_repository=index_repository,
            sparse_repository=sparse_repository,
            llm=llm,
            reranker_model=reranker_model,
            artifact_repository=artifact_repository,
            artifact_model_name=settings.deepseek_model,
            input_cost_per_million=settings.deepseek_input_cost_per_million,
            output_cost_per_million=settings.deepseek_output_cost_per_million,
        ),
        conversation_service=ConversationService(generation_client),
        memory_service=ScopedMemoryService(artifact_repository),
        annotation_service=annotation_service,
    )


def build_product_runtime_collection(
    *,
    settings: Settings,
    profile_names: tuple[str, ...] = ("standard", "high_precision"),
    streaming: bool = True,
) -> ProductRuntimeCollection:
    """Build all shared Streamlit dependencies outside the presentation layer."""
    if not profile_names:
        raise ValueError("profile_names 不能为空。")

    from langchain_openai import ChatOpenAI
    from sentence_transformers import CrossEncoder

    ensure_runtime_directories(settings)
    require_ascii_chroma_path(
        settings.global_chroma_dir,
        "RESEARCHPILOT_GLOBAL_CHROMA_DIR",
    )
    require_ascii_chroma_path(
        settings.session_chroma_dir,
        "RESEARCHPILOT_SESSION_CHROMA_DIR",
    )
    profiles = {
        name: load_product_profile(project_root=settings.project_root, profile_name=name) for name in profile_names
    }
    llm = ChatOpenAI(
        api_key=require_deepseek_api_key(settings),
        base_url=settings.deepseek_base_url,
        model=settings.deepseek_model,
        streaming=streaming,
    )
    stores = build_product_vectorstores(settings)
    index_repository = ChromaVectorIndexRepository(stores.session_store)
    memory_repository = SQLiteMemoryRepository(settings)
    memory_repository.initialize_schema()
    sparse_repository = PersistentBM25Repository(
        index_repository,
        root_dir=settings.sparse_index_dir,
        tokenizer=JiebaTokenizer(),
    )
    reranker_model = (
        CrossEncoder("BAAI/bge-reranker-base")
        if any(profile.retrieval.reranker_enabled for profile in profiles.values())
        else None
    )
    services = {
        name: build_service_from_components(
            profile=profile,
            index_repository=index_repository,
            sparse_repository=sparse_repository,
            llm=llm,
            reranker_model=(reranker_model if profile.retrieval.reranker_enabled else None),
            artifact_repository=memory_repository,
            artifact_model_name=settings.deepseek_model,
            input_cost_per_million=settings.deepseek_input_cost_per_million,
            output_cost_per_million=settings.deepseek_output_cost_per_million,
        )
        for name, profile in profiles.items()
    }
    generation_client = LangChainTextGenerationClient(
        llm,
        input_cost_per_million=settings.deepseek_input_cost_per_million,
        output_cost_per_million=settings.deepseek_output_cost_per_million,
    )
    return ProductRuntimeCollection(
        profiles=MappingProxyType(profiles),
        services=MappingProxyType(services),
        conversation_service=ConversationService(generation_client),
        memory_service=ScopedMemoryService(memory_repository),
        annotation_service=AnnotationService(memory_repository),
        research_note_service=ResearchNoteService(memory_repository),
        data_maintenance=ProductDataMaintenance(
            global_store=stores.global_store,
            session_store=stores.session_store,
            sparse_repository=sparse_repository,
            memory_repository=memory_repository,
        ),
    )
