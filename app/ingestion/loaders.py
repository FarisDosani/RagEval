from pathlib import Path
from uuid import uuid4

from docx import Document
from pypdf import PdfReader

from app.schemas.document import ExtractedUnit


def load_document(path: str | Path) -> list[ExtractedUnit]:
    """Load text with filename metadata and a shared, new ID for each call.

    PDF page numbers are one-based. Whitespace-only units are omitted;
    nonempty text is preserved as extracted. Empty documents return [].
    File access and parser errors propagate to the caller.
    """
    path = Path(path)
    extension = path.suffix.lower()
    if extension not in {".pdf", ".txt", ".docx"}:
        raise ValueError(
            f"Unsupported document extension {extension!r}; expected .pdf, .txt, or .docx"
        )

    document_id = str(uuid4())
    units: list[ExtractedUnit] = []

    def append(text: str, page_number: int | None = None) -> None:
        if text.strip():
            units.append(
                ExtractedUnit(
                    document_id=document_id,
                    source=path.name,
                    page_number=page_number,
                    text=text,
                )
            )

    if extension == ".txt":
        append(path.read_text(encoding="utf-8"))
    elif extension == ".pdf":
        with path.open("rb") as stream:
            reader = PdfReader(stream)
            for page_number, page in enumerate(reader.pages, start=1):
                append(page.extract_text() or "", page_number)
    else:
        for paragraph in Document(path).paragraphs:
            append(paragraph.text)

    return units
