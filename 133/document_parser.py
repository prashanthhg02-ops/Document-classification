"""Extract plain text from common document formats."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path


def extract_text(filename: str, content: bytes) -> str:
    """Extract text from TXT, PDF, or DOCX bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix in {".txt", ".md", ".csv"}:
        return content.decode("utf-8", errors="replace").strip()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if suffix == ".docx":
        from docx import Document

        document = Document(BytesIO(content))
        return "\n".join(paragraph.text for paragraph in document.paragraphs).strip()
    raise ValueError("Unsupported file type. Upload a TXT, PDF, or DOCX document.")
