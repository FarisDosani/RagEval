import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.retrieval import CrossEncoderReranker
from app.retrieval.reranker import DEFAULT_RERANKER_MODEL
from app.schemas.retrieval import RetrievalResult


def candidate(index, score=100.0):
    return RetrievalResult(
        chunk_id=f"chunk_{index}", document_id=f"doc_{index}", source=f"source_{index}.txt",
        page_number=2 if index == 0 else None, chunk_index=index,
        text=f"Candidate text {index}", score=score,
    )


@pytest.fixture(autouse=True)
def cross_encoder(monkeypatch):
    model = Mock()
    model.predict.return_value = np.array([0.1, 0.9, -0.2], dtype=np.float32)
    constructor = Mock(return_value=model)
    # Replace the external import so even an accidental construction is mocked.
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(CrossEncoder=constructor))
    return constructor, model


def test_ranking_metadata_batch_scoring_and_unmodified_inputs(cross_encoder):
    constructor, model = cross_encoder
    candidates = [candidate(i) for i in range(3)]
    original = [item.model_dump() for item in candidates]
    results = CrossEncoderReranker().rerank("query", iter(candidates))
    assert [item.chunk_id for item in results] == ["chunk_1", "chunk_0", "chunk_2"]
    assert [item.score for item in results] == pytest.approx([0.9, 0.1, -0.2])
    for item in results:
        assert isinstance(item, RetrievalResult)
        assert type(item.score) is float
        assert item.model_dump(exclude={"score"}) == candidates[item.chunk_index].model_dump(exclude={"score"})
    assert [item.model_dump() for item in candidates] == original
    results[0].text = "Changed"
    assert candidates[1].text == "Candidate text 1"
    constructor.assert_called_once_with(DEFAULT_RERANKER_MODEL, device="cpu")
    model.predict.assert_called_once_with(
        [("query", item.text) for item in candidates],
        batch_size=32, show_progress_bar=False, convert_to_numpy=True,
    )


@pytest.mark.parametrize("top_k, count", [(None, 3), (1, 1), (2, 2), (10, 3)])
def test_top_k(top_k, count, cross_encoder):
    results = CrossEncoderReranker().rerank("query", [candidate(i) for i in range(3)], top_k)
    assert len(results) == count
    assert results[0].chunk_id == "chunk_1"
    assert len(cross_encoder[1].predict.call_args.args[0]) == 3


def test_lazy_loading_and_empty_candidates(cross_encoder):
    constructor, model = cross_encoder
    reranker = CrossEncoderReranker()
    constructor.assert_not_called()
    assert reranker.rerank("query", []) == []
    constructor.assert_not_called()
    model.predict.assert_not_called()


def test_model_reuse_and_custom_name(cross_encoder):
    constructor, model = cross_encoder
    reranker = CrossEncoderReranker("custom-cross-encoder")
    candidates = [candidate(i) for i in range(3)]
    reranker.rerank("first query", candidates)
    reranker.rerank("second query", candidates)
    constructor.assert_called_once_with("custom-cross-encoder", device="cpu")
    assert model.predict.call_count == 2


def test_ties_preserve_candidate_order(cross_encoder):
    cross_encoder[1].predict.return_value = [0.5, 0.5, 0.5]
    reranker = CrossEncoderReranker()
    candidates = [candidate(2), candidate(0), candidate(1)]
    assert [item.chunk_id for item in reranker.rerank("query", candidates)] == ["chunk_2", "chunk_0", "chunk_1"]
    assert reranker.rerank("query", candidates) == reranker.rerank("query", candidates)


@pytest.mark.parametrize("query", ["", " \n\t", None, 123])
def test_invalid_query(query, cross_encoder):
    with pytest.raises(ValueError, match="query"):
        CrossEncoderReranker().rerank(query, [])
    cross_encoder[0].assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_invalid_top_k(top_k, cross_encoder):
    with pytest.raises(ValueError, match="top_k"):
        CrossEncoderReranker().rerank("query", [], top_k)
    cross_encoder[0].assert_not_called()


@pytest.mark.parametrize("scores", [[0.2], [[0.2]]])
def test_single_candidate_output_shapes(scores, cross_encoder):
    cross_encoder[1].predict.return_value = scores
    assert CrossEncoderReranker().rerank("query", [candidate(0)])[0].score == pytest.approx(0.2)


@pytest.mark.parametrize("scores", [[], [1, 2], [[1, 2]], [float("nan")], [float("inf")]])
def test_invalid_scores(scores, cross_encoder):
    cross_encoder[1].predict.return_value = scores
    with pytest.raises(ValueError, match="one finite score"):
        CrossEncoderReranker().rerank("query", [candidate(0)])


@pytest.mark.parametrize("original_score", [0.85, 7.5, 0.032])
def test_generic_results_ignore_original_score(original_score, cross_encoder):
    cross_encoder[1].predict.return_value = [0.25]
    result = CrossEncoderReranker().rerank("query", [candidate(0, original_score)])[0]
    assert result.score == 0.25


def test_prediction_error_propagates(cross_encoder):
    cross_encoder[1].predict.side_effect = RuntimeError("prediction failed")
    with pytest.raises(RuntimeError, match="prediction failed"):
        CrossEncoderReranker().rerank("query", [candidate(0)])
