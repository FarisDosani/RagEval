from collections.abc import Iterable
from math import isfinite
from threading import Lock
from typing import TYPE_CHECKING

from app.schemas.chunk import DocumentChunk
from app.schemas.embedding import EmbeddedChunk

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def _load_model(model_name: str) -> "SentenceTransformer":
    # Import and download weights only when embedding is first requested.
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name, device="cpu")
    model.eval()
    return model


class EmbeddingService:
    """Reuse one lazily loaded CPU model per service instance.

    The first nonempty request may download model weights. Encoding uses
    inference mode via SentenceTransformer.encode (no dropout). The same
    text and model produce repeatable vectors within the same runtime.
    Long text is subject to the selected model's input-length limit.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model: SentenceTransformer | None = None
        self._model_lock = Lock()

    def _get_model(self) -> "SentenceTransformer":
        with self._model_lock:
            if self._model is None:
                self._model = _load_model(self.model_name)
            return self._model

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if any(not text.strip() for text in texts):
            raise ValueError("Embedding text must not be empty or whitespace-only")
        if not texts:
            return []

        model = self._get_model()
        encoded = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        vectors = [[float(value) for value in row] for row in encoded]
        dimension = model.get_sentence_embedding_dimension()
        if (
            len(vectors) != len(texts)
            or not dimension
            or any(len(vector) != dimension for vector in vectors)
            or any(not isfinite(value) for vector in vectors for value in vector)
        ):
            raise ValueError("Model returned invalid embedding dimensions or values")
        return vectors

    def embed_text(self, text: str) -> list[float]:
        """Embed nonblank text as a plain Python float list."""
        return self._embed([text])[0]

    def embed_chunks(self, chunks: Iterable[DocumentChunk]) -> list[EmbeddedChunk]:
        """Batch embed chunks in input order, preserving all chunk fields.

        An empty iterable returns []; any blank chunk raises ValueError before
        loading the model or encoding any of the batch.
        """
        chunks = list(chunks)
        vectors = self._embed([chunk.text for chunk in chunks])
        return [
            EmbeddedChunk(**chunk.model_dump(), embedding=vector)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]


_default_service = EmbeddingService()


def embed_text(text: str) -> list[float]:
    return _default_service.embed_text(text)


def embed_chunks(chunks: Iterable[DocumentChunk]) -> list[EmbeddedChunk]:
    return _default_service.embed_chunks(chunks)
