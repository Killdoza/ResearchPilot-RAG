"""构建或校验指定 Scope 的持久化 BM25 索引。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from researchpilot.config import build_settings, ensure_runtime_directories, require_ascii_chroma_path  # noqa: E402
from researchpilot.index import ChromaVectorIndexRepository  # noqa: E402
from researchpilot.retrieval import JiebaTokenizer  # noqa: E402
from researchpilot.sparse import PersistentBM25Repository  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建 ResearchPilot 持久化 BM25 索引")
    parser.add_argument("--user-id", default="eval_ai_cs_v01")
    parser.add_argument("--session-id", default="corpus_v01")
    parser.add_argument("--force", action="store_true", help="忽略已有同指纹索引并重建")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        from langchain_chroma import Chroma

        settings = build_settings()
        require_ascii_chroma_path(settings.evaluation_chroma_dir, "RESEARCHPILOT_EVALUATION_CHROMA_DIR")
        ensure_runtime_directories(settings)
        vectorstore = Chroma(
            collection_name=settings.evaluation_collection_name,
            persist_directory=str(settings.evaluation_chroma_dir),
        )
        repository = PersistentBM25Repository(
            ChromaVectorIndexRepository(vectorstore),
            root_dir=settings.sparse_index_dir,
            tokenizer=JiebaTokenizer(),
        )
        manifest = repository.build(user_id=args.user_id, session_id=args.session_id, force=args.force)
    except Exception as error:
        print(f"BM25 索引构建失败：{error}", file=sys.stderr)
        return 1
    print(
        f"BM25 索引就绪：{manifest.chunk_count} 个子 Chunk，"
        f"指纹 {manifest.corpus_fingerprint[:16]}，创建于 {manifest.created_at}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
