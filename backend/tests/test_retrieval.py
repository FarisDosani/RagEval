from unittest.mock import Mock

import pytest

from app.embeddings import embedding_service
from app.retrieval import VectorStore
from app.schemas.embedding import EmbeddedChunk
from app.schemas.retrieval import RetrievalResult


def chunk(embedding, index=0):
    return EmbeddedChunk(
        chunk_id=f"doc_chunk_{index}", document_id="doc", source="report.pdf",
        page_number=3 if index == 0 else None, chunk_index=index,
        text=f"Text {index}", embedding=embedding,
    )


@pytest.fixture(autouse=True)
def forbid_model_download(monkeypatch):
    monkeypatch.setattr(
        embedding_service, "_load_model",
        Mock(side_effect=AssertionError("Retrieval tests must not load a real model")),
    )


def test_index_creation_order_scores_and_metadata():
    chunks = [chunk([0, 10], 1), chunk([2, 0]), chunk([-1, 0], 2), chunk([3, 4], 3)]
    original = [item.model_dump() for item in chunks]
    store = VectorStore(iter(chunks))
    assert len(store) == 4

    results = store.search([7, 0])

    assert [result.chunk_id for result in results] == [
        "doc_chunk_0", "doc_chunk_3", "doc_chunk_1", "doc_chunk_2"
    ]
    assert [result.score for result in results] == pytest.approx([1, 0.6, 0, -1])
    for result, source in zip(results, [chunks[1], chunks[3], chunks[0], chunks[2]]):
        assert isinstance(result, RetrievalResult)
        assert type(result.score) is float
        assert result.model_dump(exclude={"score"}) == source.model_dump(exclude={"embedding"})
    assert [item.model_dump() for item in chunks] == original


def test_input_mutation_does_not_corrupt_mapping():
    item = chunk([2, 0])
    store = VectorStore([item])
    item.text = "changed"
    item.embedding[:] = [0, 2]
    result = store.search([1, 0])[0]
    assert result.text == "Text 0"
    assert result.score == pytest.approx(1)


@pytest.mark.parametrize("top_k, expected", [(1, 1), (2, 2), (20, 2)])
def test_top_k(top_k, expected):
    store = VectorStore([chunk([1, 0]), chunk([0, 1], 1)])
    assert len(store.search([1, 0], top_k)) == expected


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True, "2"])
def test_invalid_top_k(top_k):
    for store in [VectorStore(), VectorStore([chunk([1, 0])])]:
        with pytest.raises(ValueError, match="top_k"):
            store.search([1, 0], top_k)
        with pytest.raises(ValueError, match="top_k"):
            store.search_text("query", top_k)


def test_empty_index_skips_embedding(monkeypatch):
    embed = Mock()
    monkeypatch.setattr(embedding_service, "embed_text", embed)
    store = VectorStore([])
    assert len(store) == 0
    assert store.search([1, 0]) == []
    assert store.search_text("query") == []
    embed.assert_not_called()


def test_dimension_mismatch():
    with pytest.raises(ValueError, match="same dimension"):
        VectorStore([chunk([1, 0]), chunk([1, 0, 0], 1)])
    with pytest.raises(ValueError, match="dimension must match"):
        VectorStore([chunk([1, 0])]).search([1, 0, 0])


@pytest.mark.parametrize("vector", [[], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_invalid_vectors(vector):
    # Assignment can bypass Pydantic validation; the store must still validate.
    item = chunk([1, 0])
    item.embedding = vector
    with pytest.raises(ValueError):
        VectorStore([item])
    with pytest.raises(ValueError):
        VectorStore([chunk([1, 0])]).search(vector)


def test_large_and_small_vectors_normalize_safely():
    store = VectorStore([chunk([1e300, 0]), chunk([0, 1e-300], 1)])
    assert store.search([1e-300, 0])[0].score == pytest.approx(1)
    assert store.search([0, 1e300])[0].chunk_id == "doc_chunk_1"


def test_rebuild_replaces_index_and_can_clear_it():
    store = VectorStore([chunk([1, 0])])
    store.build_index([chunk([0, 1, 0], 7)])
    assert len(store) == 1
    assert store.search([0, 1, 0])[0].chunk_id == "doc_chunk_7"
    store.build_index([])
    assert len(store) == 0
    assert store.search([1, 0]) == []


def test_failed_rebuild_preserves_existing_index():
    store = VectorStore([chunk([1, 0])])
    with pytest.raises(ValueError):
        store.build_index([chunk([1, 0]), chunk([1, 0, 0], 1)])
    assert store.search([1, 0])[0].chunk_id == "doc_chunk_0"
    assert len(store) == 1


def test_text_search_uses_existing_embedding_function(monkeypatch):
    embed = Mock(return_value=[0, 1])
    monkeypatch.setattr(embedding_service, "embed_text", embed)
    store = VectorStore([chunk([1, 0]), chunk([0, 1], 1)])
    results = store.search_text("find this", top_k=1)
    assert [result.chunk_id for result in results] == ["doc_chunk_1"]
    embed.assert_called_once_with("find this")


def test_text_search_accepts_configured_service():
    service = Mock()
    service.embed_text.return_value = [1, 0]
    assert VectorStore([chunk([1, 0])]).search_text("query", service=service)[0].score == pytest.approx(1)
    service.embed_text.assert_called_once_with("query")


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_blank_query_rejected(text):
    with pytest.raises(ValueError, match="empty or whitespace"):
        VectorStore().search_text(text)
