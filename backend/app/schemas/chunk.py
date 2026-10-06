from pydantic import Field

from app.schemas.document import ExtractedUnit


class DocumentChunk(ExtractedUnit):
    """A word window with a zero-based index within its document."""

    chunk_id: str
    chunk_index: int = Field(ge=0)
