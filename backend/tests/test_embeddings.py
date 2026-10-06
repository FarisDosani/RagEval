import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.embeddings import DEFAULT_MODEL_NAME, EmbeddingService, embed_chunks, embed_text
from app.embeddings import embedding_service
from app.schemas.chunk import DocumentChunk
from app.schemas.embedding import EmbeddedChunk


@pytest.fixture
def model_setup(monkeypatch):
    model = Mock()
    model.get_sentence_embedding_dimension.return_value = 3
    model.encode.side_effect = lambda texts, **kwargs: [
        [len(text), sum(map(ord, text)), 0] for text in texts
    ]
    loader = Mock(return_value=model)
    monkeypatch.setattr(embedding_service, "_load_model", loader)
    service = EmbeddingService()
    monkeypatch.setattr(embedding_service, "_default_service", service)
    return service, model, loader


def chunk(text, index=0, page=None):
    return DocumentChunk(
        chunk_id=f"doc_chunk_{index}",
        document_id="doc",
        source="research.pdf",
        page_number=page,
        text=text,
        chunk_index=index,
    )


def test_single_text_embedding(model_setup):
    _, model, loader = model_setup
    vector = embed_text("hello")
    assert vector == [5.0, 532.0, 0.0]
    assert type(vector) is list
    assert all(type(value) is float for value in vector)
    loader.assert_called_once_with(DEFAULT_MODEL_NAME)
    model.encode.assert_called_once_with(
        ["hello"], batch_size=32, show_progress_bar=False, convert_to_numpy=True
    )


def test_batch_preserves_metadata_order_and_input(model_setup):
    _, model, loader = model_setup
    chunks = [chunk("longer text", 7, 3), chunk("short", 2)]
    originals = [item.model_dump() for item in chunks]

    embedded = embed_chunks(iter(chunks))

    assert len(embedded) == 2
    for result, original in zip(embedded, originals, strict=True):
        assert isinstance(result, EmbeddedChunk)
        assert result.model_dump(exclude={"embedding"}) == original
        assert result.embedding[0] == float(len(original["text"]))
        assert len(result.embedding) == 3
        assert all(type(value) is float for value in result.embedding)
    assert [item.model_dump() for item in chunks] == originals
    assert model.encode.call_count == 1
    assert model.encode.call_args.args[0] == ["longer text", "short"]
    loader.assert_called_once()


def test_model_reused_and_outputs_repeatable(model_setup):
    _, _, loader = model_setup
    first = embed_text("same text")
    assert first == embed_text("same text")
    batch = embed_chunks([chunk("same text"), chunk("different", 1)])
    assert batch[0].embedding == first
    assert len(batch[1].embedding) == len(first)
    loader.assert_called_once()


def test_empty_chunk_list_does_not_load_model(model_setup):
    _, model, loader = model_setup
    assert embed_chunks([]) == []
    loader.assert_not_called()
    model.encode.assert_not_called()


@pytest.mark.parametrize("text", ["", " \n\t"])
def test_blank_text_rejected_before_loading(model_setup, text):
    _, model, loader = model_setup
    with pytest.raises(ValueError, match="empty or whitespace"):
        embed_text(text)
    with pytest.raises(ValueError, match="empty or whitespace"):
        embed_chunks([chunk("valid"), chunk(text, 1)])
    loader.assert_not_called()
    model.encode.assert_not_called()


def test_custom_model_name(model_setup):
    _, _, loader = model_setup
    service = EmbeddingService("custom-local-model")
    service.embed_text("hello")
    service.embed_text("again")
    loader.assert_called_once_with("custom-local-model")


def test_model_factory_uses_cpu_and_eval(monkeypatch):
    model = Mock()
    constructor = Mock(return_value=model)
    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=constructor),
    )
    service = EmbeddingService()
    constructor.assert_not_called()
    assert service._get_model() is model
    assert service._get_model() is model
    constructor.assert_called_once_with(DEFAULT_MODEL_NAME, device="cpu")
    model.eval.assert_called_once()


def test_numpy_output_becomes_plain_float_lists(model_setup):
    import numpy as np

    _, model, _ = model_setup
    model.encode.side_effect = None
    model.encode.return_value = np.array([[1.25, 2.5, 3.75]], dtype=np.float32)
    vector = embed_text("hello")
    assert vector == [1.25, 2.5, 3.75]
    assert type(vector) is list
    assert all(type(value) is float for value in vector)


@pytest.mark.parametrize("vectors", [
    [[1, 2, 3], [4, 5]],
    [[1, 2, 3]],
    [[1, 2], [3, 4]],
    [[], []],
    [[1, 2, float("nan")], [1, 2, 3]],
])
def test_invalid_model_output(model_setup, vectors):
    _, model, _ = model_setup
    model.encode.side_effect = None
    model.encode.return_value = vectors
    with pytest.raises(ValueError, match="invalid embedding"):
        embed_chunks([chunk("first"), chunk("second", 1)])
