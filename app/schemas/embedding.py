from pydantic import Field

from app.schemas.chunk import DocumentChunk


class EmbeddedChunk(DocumentChunk):
    """An existing chunk with its local model embedding."""

    embedding: list[float] = Field(min_length=1)
