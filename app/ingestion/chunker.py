from collections.abc import Iterable

from app.schemas.chunk import DocumentChunk
from app.schemas.document import ExtractedUnit


def chunk_document(
    document: ExtractedUnit | Iterable[ExtractedUnit],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[DocumentChunk]:
    """Split one unit or ordered units into overlapping word windows.

    Whitespace becomes single spaces. Windows never cross unit boundaries,
    preserving page metadata. Indices start at zero per document ID and
    continue across its units within this call. IDs are deterministic for
    the same input IDs, unit order, and settings: {document_id}_chunk_{index}.
    Pass all units together to keep chunk IDs unique within a document.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int):
        raise ValueError("chunk_size must be an integer greater than 0")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int):
        raise ValueError("chunk_overlap must be an integer")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    units = [document] if isinstance(document, ExtractedUnit) else document
    chunks: list[DocumentChunk] = []
    next_indices: dict[str, int] = {}
    step = chunk_size - chunk_overlap

    for unit in units:
        words = unit.text.split()
        for start in range(0, len(words), step):
            index = next_indices.get(unit.document_id, 0)
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{unit.document_id}_chunk_{index}",
                    document_id=unit.document_id,
                    source=unit.source,
                    page_number=unit.page_number,
                    text=" ".join(words[start : start + chunk_size]),
                    chunk_index=index,
                )
            )
            next_indices[unit.document_id] = index + 1
            if start + chunk_size >= len(words):
                break

    return chunks
