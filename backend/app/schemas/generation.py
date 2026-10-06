from pydantic import BaseModel, Field

from app.schemas.retrieval import RetrievalResult


class Citation(BaseModel):
    """Metadata for a source supplied as context, not verified claim attribution."""

    chunk_id: str
    source: str
    page_number: int | None


class GenerationUsage(BaseModel):
    """One provider call; cost uses provider-reported units, never estimated pricing.

    None means unavailable. A skipped call uses zero latency, tokens, and cost.
    """

    latency_ms: float = Field(default=0.0, ge=0, allow_inf_nan=False)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    cost: float | None = Field(default=None, ge=0, allow_inf_nan=False)


class LLMGenerationResult(GenerationUsage):
    text: str
    model: str


class RAGResponse(BaseModel):
    """latency_ms includes retrieval/local processing; usage.latency_ms is the LLM call only."""

    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievalResult]
    model: str
    latency_ms: float = Field(ge=0)
    usage: GenerationUsage = Field(default_factory=GenerationUsage)
