"""No-retrieval baseline shared by query and experiment entry points."""
from time import perf_counter

from app.generation.prompts import INSUFFICIENT_INFORMATION
from app.schemas.generation import GenerationUsage, RAGResponse

LLM_ONLY_PROMPT = f"""Answer the question using your own knowledge.
Keep the answer concise and factual. No retrieved sources are supplied.
If you cannot answer reliably, return exactly:
{INSUFFICIENT_INFORMATION}
"""


def generate_llm_only(llm, question: str) -> RAGResponse:
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    started = perf_counter()
    result = llm.generate_result(LLM_ONLY_PROMPT, question)
    return RAGResponse(
        question=question, answer=result.text, model=result.model,
        retrieved_chunks=[], citations=[], latency_ms=(perf_counter() - started) * 1000,
        usage=GenerationUsage(**result.model_dump(exclude={"text", "model"})),
    )
