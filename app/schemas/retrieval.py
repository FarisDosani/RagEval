from app.schemas.chunk import DocumentChunk


class RetrievalResult(DocumentChunk):
    """Chunk metadata and retriever-specific score (higher ranks first)."""

    score: float
