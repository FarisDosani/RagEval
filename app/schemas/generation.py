from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievalResult


class Citation(BaseModel):
    """Metadata for a source supplied as context, not verified claim attribution."""

    chunk_id: str
    source: str
    page_number: int | None


class RAGResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]
    model: str
    latency_ms: float = Field(ge=0)
