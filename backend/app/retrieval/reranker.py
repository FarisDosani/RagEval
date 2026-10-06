from __future__ import annotations

from collections.abc import Iterable
from threading import Lock
from typing import TYPE_CHECKING

import numpy as np

from app.schemas.retrieval import RetrievalResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class CrossEncoderReranker:
    """Batch rerank any retrieval results using one lazy CPU cross-encoder.

    Scores replace retrieval scores in new result objects; inputs are untouched.
    Equal scores retain candidate order. The first nonempty call may download
    weights. Custom models must return one scalar relevance score per pair.
    Model-loading and prediction errors propagate to the caller.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        self.model_name = model_name
        self._model: CrossEncoder | None = None
        self._model_lock = Lock()

    def _get_model(self) -> CrossEncoder:
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, device="cpu")
            return self._model

    def rerank(
        self,
        query: str,
        candidates: Iterable[RetrievalResult],
        top_k: int | None = None,
    ) -> list[RetrievalResult]:
        """Score all candidates before selecting top_k; [] never loads the model."""
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        if top_k is not None and (
            isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0
        ):
            raise ValueError("top_k must be an integer greater than 0")
        candidates = list(candidates)
        if not candidates:
            return []

        pairs = [(query, candidate.text) for candidate in candidates]
        scores = np.asarray(self._get_model().predict(
            pairs, batch_size=32, show_progress_bar=False, convert_to_numpy=True,
        ), dtype=np.float64)
        if scores.shape == (len(candidates), 1):
            scores = scores[:, 0]
        if scores.shape != (len(candidates),) or not np.isfinite(scores).all():
            raise ValueError("CrossEncoder must return one finite score per candidate")

        positions = sorted(range(len(candidates)), key=lambda position: -scores[position])
        return [
            candidates[position].model_copy(update={"score": float(scores[position])}, deep=True)
            for position in positions[:top_k]
        ]
