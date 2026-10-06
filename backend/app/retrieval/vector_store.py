from collections.abc import Iterable

import faiss
import numpy as np

from app.embeddings import embedding_service
from app.embeddings.embedding_service import EmbeddingService
from app.schemas.embedding import EmbeddedChunk
from app.schemas.retrieval import RetrievalResult


def _normalized_vector(embedding: list[float]) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError("Embedding must be a non-empty one-dimensional vector")
    if not np.isfinite(vector).all():
        raise ValueError("Embedding values must be finite")
    scale = np.max(np.abs(vector))
    if scale == 0:
        raise ValueError("Embedding must have non-zero magnitude for cosine similarity")
    # Scale before float32 conversion to avoid overflow/underflow of valid vectors.
    normalized = np.ascontiguousarray((vector / scale)[None, :], dtype=np.float32)
    faiss.normalize_L2(normalized)
    return normalized[0]


def _validate_top_k(top_k: int) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be an integer greater than 0")


class VectorStore:
    """In-memory exact cosine search over copies of embedded chunks.

    Rebuild replaces the index and metadata together after validation. Empty
    input clears the store. Query vectors must come from the same embedding
    model as the indexed chunks; dimension checks cannot detect model mismatch.
    """

    def __init__(self, chunks: Iterable[EmbeddedChunk] = ()):
        self._index: faiss.IndexFlatIP | None = None
        self._chunks: list[EmbeddedChunk] = []
        self.build_index(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def build_index(self, chunks: Iterable[EmbeddedChunk]) -> None:
        """Build or rebuild without modifying the input chunks or embeddings."""
        stored = [chunk.model_copy(deep=True) for chunk in chunks]
        index = None
        if stored:
            vectors = [_normalized_vector(chunk.embedding) for chunk in stored]
            dimension = len(vectors[0])
            if any(len(vector) != dimension for vector in vectors):
                raise ValueError("All embeddings must have the same dimension")
            matrix = np.ascontiguousarray(np.stack(vectors), dtype=np.float32)
            index = faiss.IndexFlatIP(dimension)
            index.add(matrix)
        self._index = index
        self._chunks = stored

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[RetrievalResult]:
        """Return highest scores first; a valid query on an empty store returns []."""
        _validate_top_k(top_k)
        query = _normalized_vector(query_embedding)
        if self._index is None:
            return []
        if len(query) != self._index.d:
            raise ValueError("Query embedding dimension must match the index dimension")
        scores, positions = self._index.search(query[None, :], min(top_k, len(self)))
        return [
            RetrievalResult(
                **self._chunks[int(position)].model_dump(exclude={"embedding"}),
                score=float(score),
            )
            for score, position in zip(scores[0], positions[0], strict=True)
        ]

    def search_text(
        self,
        query: str,
        top_k: int = 5,
        *,
        service: EmbeddingService | None = None,
    ) -> list[RetrievalResult]:
        """Embed using the existing shared service, or a supplied matching service.

        Empty stores return [] without loading an embedding model. Blank queries
        and invalid top_k values are rejected even when the store is empty.
        """
        _validate_top_k(top_k)
        if not query.strip():
            raise ValueError("Query text must not be empty or whitespace-only")
        if not self._chunks:
            return []
        embed = service.embed_text if service is not None else embedding_service.embed_text
        return self.search(embed(query), top_k)
