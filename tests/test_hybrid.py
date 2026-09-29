from unittest.mock import Mock

import pytest

from app.retrieval import BM25Retriever, HybridRetriever, VectorStore
from app.schemas.retrieval import RetrievalResult


def result(chunk_id, score=0.5, text="Example text"):
    return RetrievalResult(
        chunk_id=chunk_id, document_id="doc", source="notes.pdf",
        page_number=2, chunk_index=7, text=text, score=score,
    )


def hybrid(dense_results=(), bm25_results=(), rrf_k=60):
    dense = Mock(spec=VectorStore)
    dense.search_text.return_value = list(dense_results)
    bm25 = Mock(spec=BM25Retriever)
    bm25.search.return_value = list(bm25_results)
    return HybridRetriever(dense, bm25, rrf_k), dense, bm25


def test_fusion_ranking_and_source_only_chunks():
    retriever, dense, bm25 = hybrid(
        [result("shared", -100), result("dense-only", 99999)],
        [result("lexical-only", 1e9), result("shared", -999)],
    )
    results = retriever.search("question")
    assert [item.chunk_id for item in results] == ["shared", "lexical-only", "dense-only"]
    assert [item.score for item in results] == pytest.approx([1 / 61 + 1 / 62, 1 / 61, 1 / 62])
    dense.search_text.assert_called_once_with("question", top_k=20)
    bm25.search.assert_called_once_with("question", top_k=20)


def test_highly_ranked_in_both():
    retriever, _, _ = hybrid([result("shared"), result("a")], [result("shared"), result("b")])
    results = retriever.search("query")
    assert results[0].chunk_id == "shared"
    assert results[0].score == pytest.approx(2 / 61)


def test_custom_parameters_and_final_limit():
    retriever, dense, bm25 = hybrid([result("a"), result("b")], [result("b")], rrf_k=10)
    results = retriever.search("query", candidate_k=2, final_top_k=1)
    assert len(results) == 1
    assert results[0].chunk_id == "b"
    assert results[0].score == pytest.approx(1 / 12 + 1 / 11)
    dense.search_text.assert_called_once_with("query", top_k=2)
    bm25.search.assert_called_once_with("query", top_k=2)


def test_duplicate_elimination_and_original_ranks():
    retriever, _, _ = hybrid(
        [result("a"), result("a"), result("b")],
        [result("b"), result("b"), result("a")],
    )
    results = retriever.search("query")
    assert [item.chunk_id for item in results] == ["a", "b"]
    assert [item.score for item in results] == pytest.approx([1 / 61 + 1 / 63] * 2)


def test_metadata_preserved_without_mutating_inputs():
    original = result("a", score=87)
    before = original.model_dump()
    retriever, _, _ = hybrid([original], [result("a", text="conflicting metadata")])
    fused = retriever.search("query")[0]
    assert isinstance(fused, RetrievalResult)
    assert fused.model_dump(exclude={"score"}) == original.model_dump(exclude={"score"})
    assert original.model_dump() == before
    fused.text = "mutated result"
    assert original.text == "Example text"


@pytest.mark.parametrize("dense_ids, lexical_ids", [([], ["b"]), (["a"], []), ([], [])])
def test_empty_rankings_and_large_final_k(dense_ids, lexical_ids):
    retriever, _, _ = hybrid([result(i) for i in dense_ids], [result(i) for i in lexical_ids])
    results = retriever.search("query", final_top_k=100)
    assert [item.chunk_id for item in results] == dense_ids + lexical_ids
    assert all(item.score == pytest.approx(1 / 61) for item in results)


def test_ties_use_chunk_id_not_retriever_order():
    first, _, _ = hybrid([result("z")], [result("a")])
    second, _, _ = hybrid([result("a")], [result("z")])
    assert first.search("query") == second.search("query")
    assert [item.chunk_id for item in first.search("query")] == ["a", "z"]


@pytest.mark.parametrize("name", ["candidate_k", "final_top_k", "rrf_k"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5, "2", None])
def test_invalid_parameters(name, value):
    if name == "rrf_k":
        with pytest.raises(ValueError, match=name):
            hybrid(rrf_k=value)
    else:
        retriever, dense, bm25 = hybrid()
        with pytest.raises(ValueError, match=name):
            retriever.search("query", **{name: value})
        dense.search_text.assert_not_called()
        bm25.search.assert_not_called()


@pytest.mark.parametrize("query", ["", " \n\t", None, 42])
def test_invalid_query(query):
    retriever, dense, bm25 = hybrid()
    with pytest.raises(ValueError, match="query"):
        retriever.search(query)
    dense.search_text.assert_not_called()
    bm25.search.assert_not_called()


def test_candidate_limit_is_respected():
    retriever, _, _ = hybrid([result("a"), result("b")], [result("c"), result("b")])
    assert {item.chunk_id for item in retriever.search("query", candidate_k=1)} == {"a", "c"}


def test_retrieval_failure_propagates():
    retriever, dense, _ = hybrid()
    dense.search_text.side_effect = RuntimeError("retrieval failed")
    with pytest.raises(RuntimeError, match="retrieval failed"):
        retriever.search("query")
