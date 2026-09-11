"""Vector-store maintenance kept outside the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from .config import Settings
from .sparse import PersistentBM25Repository
from .storage import SQLiteMemoryRepository


@dataclass(frozen=True)
class ProductVectorStores:
    global_store: Chroma
    session_store: Chroma


def build_product_vectorstores(settings: Settings) -> ProductVectorStores:
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    return ProductVectorStores(
        global_store=Chroma(
            collection_name=settings.global_collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.global_chroma_dir),
        ),
        session_store=Chroma(
            collection_name=settings.session_collection_name,
            embedding_function=embeddings,
            persist_directory=str(settings.session_chroma_dir),
        ),
    )


@dataclass
class ProductDataMaintenance:
    global_store: Chroma
    session_store: Chroma
    sparse_repository: PersistentBM25Repository
    memory_repository: SQLiteMemoryRepository

    def delete_session_data(self, *, user_id: str, session_id: str) -> dict[str, object]:
        """Delete the non-JSON runtime data owned by one session."""
        cleanup: dict[str, object] = {
            "sqlite": self.memory_repository.delete_session(user_id, session_id),
        }
        cleanup.update(self.delete_session_vectors(user_id=user_id, session_id=session_id))
        return cleanup

    def paper_related_counts(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
    ) -> dict[str, int]:
        """Expose delete-impact counts without leaking the SQL repository into the UI."""
        return self.memory_repository.paper_related_counts(
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )

    def delete_session_vectors(self, *, user_id: str, session_id: str) -> dict[str, int]:
        session_rows = self.session_store.get(where={"$and": [{"user_id": user_id}, {"session_id": session_id}]})
        session_ids = session_rows.get("ids", []) if session_rows else []
        if session_ids:
            self.session_store.delete(ids=session_ids)
        self.sparse_repository.delete_scope(user_id=user_id, session_id=session_id)

        chat_rows = self.global_store.get(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"session_id": session_id},
                    {"type": "chat_history"},
                ]
            }
        )
        chat_ids = chat_rows.get("ids", []) if chat_rows else []
        if chat_ids:
            self.global_store.delete(ids=chat_ids)
        return {"session_vector_records_deleted": len(session_ids), "chat_vector_records_deleted": len(chat_ids)}

    def delete_legacy_paper(self, *, user_id: str, session_id: str, source_name: str) -> int:
        rows = self.session_store.get(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"session_id": session_id},
                    {"source": source_name},
                ]
            }
        )
        ids = rows.get("ids", []) if rows else []
        if ids:
            self.session_store.delete(ids=ids)
        return len(ids)

    def archive_summary(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        source_name: str,
        summary_text: str,
    ) -> None:
        self.global_store.add_documents(
            [
                Document(
                    page_content=f"【全局归档文档摘要】\n文档名：{source_name}\n{summary_text}",
                    metadata={
                        "user_id": user_id,
                        "source": source_name,
                        "type": "long_term_summary",
                        "session_id": "global_archive",
                    },
                )
            ],
            ids=[f"summary_{user_id}_{paper_id}"],
        )
        self.memory_repository.add_long_term_memory(
            user_id,
            session_id,
            "document_summary",
            f"【{source_name} 全局摘要】: {summary_text}",
            source_doc=paper_id,
            priority=6,
        )

    def list_archived_sources(self, *, user_id: str) -> tuple[str, ...]:
        rows = self.global_store.get(where={"$and": [{"user_id": user_id}, {"type": "long_term_summary"}]})
        metadata = rows.get("metadatas", []) if rows else []
        return tuple(dict.fromkeys(str(item["source"]) for item in metadata if item.get("source")))

    def delete_archive(self, *, user_id: str, source_name: str) -> int:
        rows = self.global_store.get(
            where={
                "$and": [
                    {"user_id": user_id},
                    {"source": source_name},
                    {"type": "long_term_summary"},
                ]
            }
        )
        ids = rows.get("ids", []) if rows else []
        if ids:
            self.global_store.delete(ids=ids)
        return len(ids)
