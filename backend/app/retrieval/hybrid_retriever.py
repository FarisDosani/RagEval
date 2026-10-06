from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.retrieval import RetrievalResult


def _positive_integer(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be an integer greater than 0")


class HybridRetriever:
    """Fuse dense and BM25 rankings, ignoring their raw scores.

    Each chunk contributes once per retriever at its first original 1-based
    rank; duplicates do not shift later ranks. Metadata comes from the first
    occurrence, preferring dense results on conflicts. Ties use chunk_id in
    ascending order. Underlying retrieval errors propagate to the caller.
    """

    def __init__(
        self, dense_retriever: VectorStore, bm25_retriever: BM25Retriever,
        rrf_k: int = 60,
    ):
        _positive_integer(rrf_k, "rrf_k")
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.rrf_k = rrf_k

    def search(
        self, query: str, candidate_k: int = 20, final_top_k: int = 5
    ) -> list[RetrievalResult]:
        """Retrieve up to candidate_k from each source, then return fused top results."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        _positive_integer(candidate_k, "candidate_k")
        _positive_integer(final_top_k, "final_top_k")
        _positive_integer(self.rrf_k, "rrf_k")
        dense = self.dense_retriever.search_text(query, top_k=candidate_k)
        lexical = self.bm25_retriever.search(query, top_k=candidate_k)

        scores: dict[str, float] = {}
        originals: dict[str, RetrievalResult] = {}
        for ranking in (dense, lexical):
            seen: set[str] = set()
            for rank, result in enumerate(ranking[:candidate_k], start=1):
                chunk_id = result.chunk_id
                if chunk_id in seen:
                    continue
                seen.add(chunk_id)
                originals.setdefault(chunk_id, result)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)

        ordered_ids = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
        return [
            originals[chunk_id].model_copy(update={"score": scores[chunk_id]}, deep=True)
            for chunk_id in ordered_ids[:final_top_k]
        ]
