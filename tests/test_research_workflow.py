from __future__ import annotations

import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from researchpilot.config import build_settings  # noqa: E402
from researchpilot.research_workflow import (  # noqa: E402
    ResearchNoteService,
    derive_session_title,
    persist_pdf_copy,
    render_pdf_page_png,
    trim_logo_image,
)
from researchpilot.routing import RuntimeRoute  # noqa: E402
from researchpilot.schemas import Citation  # noqa: E402
from researchpilot.storage import SQLiteMemoryRepository  # noqa: E402


class ResearchWorkflowTests(unittest.TestCase):
    def test_session_title_is_deterministic_and_readable(self) -> None:
        title = derive_session_title(source_name="2024_LightRAG.pdf", route=RuntimeRoute.EXPERIMENTAL_RESULTS)
        self.assertEqual(title, "2024 LightRAG · 实验结果")

    def test_note_can_be_exported_with_page_citation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(project_root=Path(directory), environ={}, load_dotenv_file=False)
            repo = SQLiteMemoryRepository(settings)
            repo.initialize_schema()
            service = ResearchNoteService(repo)
            citation = Citation("paper_a", "chunk_a", 7, 8, "result")
            service.save_answer(
                user_id="user_a",
                session_id="session_a",
                paper_id="paper_a",
                question="结果？",
                answer_text="结果如下。",
                citations=(citation,),
            )
            duplicate = service.save_answer(
                user_id="user_a",
                session_id="session_a",
                paper_id="paper_a",
                question="结果？",
                answer_text="结果如下。",
                citations=(citation,),
            )
            notes = service.list_notes(user_id="user_a", session_id="session_a", paper_id="paper_a")
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].note_id, duplicate.note_id)
            exported = service.export_markdown(user_id="user_a", session_id="session_a", paper_id="paper_a")
            self.assertIn("结果？", exported)
            self.assertIn("第 7-8 页", exported)
            self.assertTrue(
                service.delete_note(
                    user_id="user_a",
                    session_id="session_a",
                    paper_id="paper_a",
                    note_id=duplicate.note_id,
                )
            )
            self.assertEqual(
                service.list_notes(user_id="user_a", session_id="session_a", paper_id="paper_a"),
                [],
            )

    def test_persisted_pdf_page_can_be_rendered_for_citation_review(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = build_settings(project_root=Path(directory), environ={}, load_dotenv_file=False)
            document = pdfium.PdfDocument.new()
            document.new_page(300, 400)
            buffer = BytesIO()
            document.save(buffer)
            document.close()

            path = persist_pdf_copy(
                settings=settings,
                user_id="user_a",
                session_id="session_a",
                paper_id="paper_a",
                pdf_bytes=buffer.getvalue(),
            )
            png = render_pdf_page_png(path, 1)

            self.assertTrue(path.is_file())
            self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_logo_margin_is_cropped_in_memory_without_rewriting_source(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "logo.png"
            image = Image.new("RGB", (200, 200), "white")
            for x in range(80, 120):
                for y in range(70, 130):
                    image.putpixel((x, y), (0, 80, 200))
            image.save(path)
            original = path.read_bytes()

            cropped = trim_logo_image(path, padding_ratio=0.05)

            self.assertIsNotNone(cropped)
            assert cropped is not None
            self.assertLess(cropped.width, 100)
            self.assertLess(cropped.height, 100)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
