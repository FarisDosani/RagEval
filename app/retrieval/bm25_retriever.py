import re
from collections.abc import Iterable

from rank_bm25 import BM25Okapi

from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import RetrievalResult


def _tokenize(text: str) -> list[str]:
    """Lowercase Unicode alphanumeric tokens; punctuation/underscores separate words."""
    return re.findall(r"[^\W_]+", text.lower())


class BM25Retriever:
    """In-memory BM25Okapi with original-order tie breaking.

    Scores are raw BM25 scores (not cosine scores); zero/negative scores are
    retained. No-match queries return available chunks with zero scores.
    Empty/tokenless chunks retain their corpus positions. An entirely tokenless
    corpus returns zero scores without constructing an invalid BM25 index.
    """

    def __init__(self, chunks: Iterable[DocumentChunk] = ()):
        self._chunks: list[DocumentChunk] = []
        self._index: BM25Okapi | None = None
        self.build_index(chunks)

    def __len__(self) -> int:
        return len(self._chunks)

    def build_index(self, chunks: Iterable[DocumentChunk]) -> None:
        """Replace the corpus after building successfully; [] clears it."""
        stored = [chunk.model_copy(deep=True) for chunk in chunks]
        corpus = [_tokenize(chunk.text) for chunk in stored]
        index = BM25Okapi(corpus) if any(corpus) else None
        self._chunks = stored
        self._index = index

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return highest BM25 scores first, capped at corpus size.

        Blank or punctuation-only queries raise ValueError. A valid query on
        an empty corpus returns []. Neither input chunks nor metadata are changed.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be an integer greater than 0")
        tokens = _tokenize(query)
        if not tokens:
            raise ValueError("query must contain at least one alphanumeric token")
        if not self._chunks:
            return []
        scores = self._index.get_scores(tokens) if self._index is not None else [0.0] * len(self)
        positions = sorted(range(len(self)), key=lambda position: -scores[position])[:top_k]
        return [
            RetrievalResult(
                **self._chunks[position].model_dump(include={
                    "chunk_id", "document_id", "source", "page_number", "chunk_index", "text"
                }),
                score=float(scores[position]),
            )
            for position in positions
        ]
