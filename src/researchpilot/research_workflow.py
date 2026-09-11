"""Small, deterministic research-workflow helpers used by the UI."""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops

from .config import Settings, validate_storage_id
from .routing import RuntimeRoute
from .schemas import Citation
from .storage import ResearchNote, SQLiteMemoryRepository


def trim_logo_image(path: Path, *, padding_ratio: float = 0.04) -> Image.Image | None:
    """Crop near-white margins in memory without rewriting the source asset."""
    if not path.is_file():
        return None
    with Image.open(path) as source:
        image = source.convert("RGB")
    white = Image.new("RGB", image.size, "white")
    difference = ImageChops.difference(image, white).convert("L")
    foreground = difference.point(lambda value: 255 if value > 12 else 0)
    box = foreground.getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    padding = max(4, int(max(right - left, bottom - top) * padding_ratio))
    return image.crop(
        (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width, right + padding),
            min(image.height, bottom + padding),
        )
    )


def derive_session_title(*, source_name: str | None, route: RuntimeRoute | None) -> str:
    paper = Path(source_name).stem if source_name else "通用研究"
    paper = re.sub(r"[_-]+", " ", paper).strip()
    if len(paper) > 24:
        paper = paper[:24].rstrip() + "…"
    task = {
        RuntimeRoute.METADATA: "元信息",
        RuntimeRoute.DOCUMENT_SUMMARY: "全文总结",
        RuntimeRoute.EXPERIMENTAL_SETUP: "实验设置",
        RuntimeRoute.EXPERIMENTAL_RESULTS: "实验结果",
        RuntimeRoute.MULTI_SECTION: "综合分析",
        RuntimeRoute.DOCUMENT_DETAIL: "论文问答",
        RuntimeRoute.CROSS_DOCUMENT: "跨论文比较",
        RuntimeRoute.GENERAL_CHAT: "通用对话",
    }.get(route, "论文研究")
    return f"{paper} · {task}"


def persist_pdf_copy(*, settings: Settings, user_id: str, session_id: str, paper_id: str, pdf_bytes: bytes) -> Path:
    """Persist an ignored local PDF copy for page-level evidence inspection."""
    for value, name in ((user_id, "user_id"), (session_id, "session_id"), (paper_id, "paper_id")):
        validate_storage_id(value, name)
    if not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("uploaded content is not a PDF")
    directory = settings.uploads_dir / user_id / session_id
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{paper_id}.pdf"
    temporary = directory / f".{paper_id}.pdf.tmp"
    temporary.write_bytes(pdf_bytes)
    temporary.replace(target)
    return target


def render_pdf_page_png(pdf_path: Path, page: int, *, scale: float = 1.5) -> bytes:
    """Render one 1-based PDF page for in-product citation verification."""
    if page < 1:
        raise ValueError("page must be positive")
    import pypdfium2 as pdfium

    document = pdfium.PdfDocument(str(pdf_path))
    try:
        if page > len(document):
            raise ValueError("page exceeds PDF page count")
        bitmap = document[page - 1].render(scale=scale)
        output = BytesIO()
        bitmap.to_pil().convert("RGB").save(output, format="PNG")
        return output.getvalue()
    finally:
        document.close()


@dataclass
class ResearchNoteService:
    repository: SQLiteMemoryRepository

    def save_answer(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        question: str,
        answer_text: str,
        citations: tuple[Citation, ...],
    ) -> ResearchNote:
        citation_fingerprint = "\n".join(
            f"{item.paper_id}:{item.chunk_id}:{item.page_start}:{item.page_end}" for item in citations
        )
        fingerprint = sha256(
            "\n".join(
                (
                    user_id,
                    session_id,
                    paper_id,
                    question.strip(),
                    answer_text.strip(),
                    citation_fingerprint,
                )
            ).encode()
        ).hexdigest()
        note = ResearchNote(
            note_id=f"note_{fingerprint}",
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
            question=question,
            answer_text=answer_text,
            citations=citations,
        )
        self.repository.save_research_note(note)
        return note

    def list_notes(self, *, user_id: str, session_id: str, paper_id: str) -> list[ResearchNote]:
        return self.repository.list_research_notes(
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
        )

    def delete_note(
        self,
        *,
        user_id: str,
        session_id: str,
        paper_id: str,
        note_id: str,
    ) -> bool:
        return self.repository.delete_research_note(
            user_id=user_id,
            session_id=session_id,
            paper_id=paper_id,
            note_id=note_id,
        )

    def export_markdown(self, *, user_id: str, session_id: str, paper_id: str) -> str:
        notes = self.repository.list_research_notes(user_id=user_id, session_id=session_id, paper_id=paper_id)
        rows = ["# ResearchPilot 论文阅读记录", "", f"- Paper ID：`{paper_id}`", ""]
        for index, note in enumerate(notes, start=1):
            rows.extend(
                [
                    f"## {index}. {note.question}",
                    "",
                    note.answer_text,
                    "",
                    "### 证据",
                    "",
                ]
            )
            rows.extend(
                f"- 第 {citation.page_start}"
                f"{'-' + str(citation.page_end) if citation.page_end != citation.page_start else ''} 页"
                f" · `{citation.chunk_id}`"
                for citation in note.citations
            )
            rows.append("")
        if not notes:
            rows.extend(["当前论文尚未保存研究笔记。", ""])
        return "\n".join(rows)
