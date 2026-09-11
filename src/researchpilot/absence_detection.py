"""Deterministic full-document scans for fields that must not be guessed."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from .index import IndexHit


class DocumentField(str, Enum):
    CODE_URL = "code_url"
    DOI = "doi"
    URL = "url"


@dataclass(frozen=True)
class FieldMatch:
    field_type: DocumentField
    value: str
    hit: IndexHit


@dataclass(frozen=True)
class FullDocumentScan:
    field_type: DocumentField
    scanned_block_count: int
    matches: tuple[FieldMatch, ...]

    @property
    def matched_block_count(self) -> int:
        return len({item.hit.chunk.chunk_id for item in self.matches})


_DOI_RE = re.compile(r"(?i)(?:https?://(?:dx\.)?doi\.org/)?10\.\d{4,9}/[-._;()/:a-z0-9]+")
_URL_RE = re.compile(
    r"(?i)(?:https?://[^\s<>\[\]{}\"']+|"
    r"(?<![@\w])(?:www\.)?[a-z0-9][a-z0-9.-]*\."
    r"(?:com|org|net|io|ai|ms|dev|edu|gov)/[^\s<>\[\]{}\"']+)"
)
_CODE_HOST_RE = re.compile(r"(?i)(?:github\.com|gitlab\.com|bitbucket\.org)")


def infer_requested_field(question: str) -> DocumentField | None:
    lowered = question.casefold()
    if re.search(r"\bdoi\b", lowered):
        return DocumentField.DOI
    if re.search(
        r"(?:代码|仓库|源码|实现|code|repository|repo|github|gitlab).{0,12}"
        r"(?:地址|链接|网址|url|开源|公开|link)|"
        r"(?:地址|链接|网址|url|link).{0,12}(?:代码|仓库|源码|code|repo|github|gitlab)",
        lowered,
    ):
        return DocumentField.CODE_URL
    if re.search(r"(?:项目|数据集|主页|网站|supplementary).{0,12}(?:地址|链接|网址|url|link)", lowered):
        return DocumentField.URL
    return None


def _clean_match(value: str) -> str:
    return value.rstrip(".,;:!?，。；：！？)]}>'\"")


def scan_document_fields(hits: Sequence[IndexHit], *, field_type: DocumentField) -> FullDocumentScan:
    matches: list[FieldMatch] = []
    seen: set[tuple[str, str]] = set()
    for hit in hits:
        pattern = _DOI_RE if field_type is DocumentField.DOI else _URL_RE
        for matched in pattern.finditer(hit.chunk.content):
            value = _clean_match(matched.group(0))
            if field_type is DocumentField.CODE_URL:
                context = hit.chunk.content[max(0, matched.start() - 120) : matched.end() + 120]
                if not _CODE_HOST_RE.search(value) and not re.search(
                    r"(?i)code|repository|repo|source|代码|仓库|源码", context
                ):
                    continue
            key = (hit.chunk.chunk_id, value.casefold())
            if not value or key in seen:
                continue
            seen.add(key)
            matches.append(FieldMatch(field_type=field_type, value=value, hit=hit))
    return FullDocumentScan(
        field_type=field_type,
        scanned_block_count=len(hits),
        matches=tuple(matches),
    )
