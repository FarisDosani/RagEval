from time import perf_counter

from app.embeddings.embedding_service import EmbeddingService
from app.generation.llm_service import LLMService
from app.generation.prompts import INSUFFICIENT_INFORMATION, SYSTEM_PROMPT, build_user_prompt
from app.retrieval import VectorStore
from app.schemas.generation import Citation, RAGResponse


class RAGPipeline:
    """Answer questions over an already-built vector store.

    Supply a matching embedding service for stores built with a custom model.
    Citations list context sources; they do not assert claim-level support.
    The model field records the configured model even when no LLM call occurs.
    """

    def __init__(
        self, vector_store: VectorStore, llm_service: LLMService | None = None,
        embedding_service: EmbeddingService | None = None,
    ):
        self.vector_store = vector_store
        self.llm = llm_service if llm_service is not None else LLMService()
        self.embedding_service = embedding_service

    def ask(self, question: str, top_k: int = 5) -> RAGResponse:
        started = perf_counter()
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be an integer greater than 0")
        chunks = self.vector_store.search_text(
            question, top_k=top_k, service=self.embedding_service
        )
        answer = INSUFFICIENT_INFORMATION
        if chunks:
            answer = self.llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks))
        citations = [] if answer == INSUFFICIENT_INFORMATION else [
            Citation(chunk_id=chunk.chunk_id, source=chunk.source, page_number=chunk.page_number)
            for chunk in chunks
        ]
        return RAGResponse(
            question=question, answer=answer, citations=citations,
            retrieved_chunks=chunks, model=self.llm.model,
            latency_ms=(perf_counter() - started) * 1000,
        )
