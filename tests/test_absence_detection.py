from researchpilot.absence_detection import (
    DocumentField,
    infer_requested_field,
    scan_document_fields,
)
from researchpilot.index import IndexHit
from researchpilot.schemas import BlockType, Chunk


def _hit(text: str, *, page: int = 2) -> IndexHit:
    chunk = Chunk.create(
        paper_id="paper_demo",
        parent_id="parent_demo",
        page_start=page,
        page_end=page,
        section_path=("Resources",),
        content=text,
        block_types=(BlockType.PARAGRAPH,),
        chunker_version="test_v1",
    )
    return IndexHit(chunk=chunk, source_name="demo.pdf", score=0.0)


def test_requested_field_is_conservative():
    assert infer_requested_field("论文有没有公开代码地址？") is DocumentField.CODE_URL
    assert infer_requested_field("这篇论文的 DOI 是什么？") is DocumentField.DOI
    assert infer_requested_field("代码模块是怎么实现的？") is None


def test_full_document_scan_finds_code_url_and_reports_counts():
    scan = scan_document_fields(
        (_hit("The code is available at https://github.com/example/researchpilot."),),
        field_type=DocumentField.CODE_URL,
    )
    assert scan.scanned_block_count == 1
    assert scan.matched_block_count == 1
    assert scan.matches[0].value == "https://github.com/example/researchpilot"


def test_full_document_scan_does_not_treat_arbitrary_url_as_code():
    scan = scan_document_fields(
        (_hit("Project page: https://example.org/paper"),),
        field_type=DocumentField.CODE_URL,
    )
    assert not scan.matches


def test_full_document_scan_finds_declared_bare_code_url():
    scan = scan_document_fields(
        (_hit("The code is available in aka.ms/efficientrag ."),),
        field_type=DocumentField.CODE_URL,
    )
    assert scan.matches[0].value == "aka.ms/efficientrag"
