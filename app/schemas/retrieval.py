from app.schemas.chunk import DocumentChunk


class RetrievalResult(DocumentChunk):
    """Chunk metadata and cosine similarity (higher is more similar)."""

    score: float
