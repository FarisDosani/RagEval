import math

import pytest

from app.retrieval import BM25Retriever
from app.schemas.chunk import DocumentChunk
from app.schemas.embedding import EmbeddedChunk
from app.schemas.retrieval import RetrievalResult


def chunk(text, index=0):
    return DocumentChunk(
        chunk_id=f"doc_chunk_{index}", document_id="doc", source="example.pdf",
        page_number=2 if index == 0 else None, chunk_index=index, text=text,
    )


def corpus():
    return [chunk(text, index) for index, text in enumerate([
        "quartz quartz quartz mineral", "quartz mineral stone sample",
        "ocean water waves blue", "forest trees leaves green", "city road bus car",
    ])]


def test_index_creation_lexical_ranking_and_metadata():
    chunks = corpus()
    originals = [item.model_dump() for item in chunks]
    retriever = BM25Retriever(iter(chunks))
    assert len(retriever) == 5
    results = retriever.search("quartz")
    assert [item.chunk_id for item in results[:2]] == ["doc_chunk_0", "doc_chunk_1"]
    assert results[0].score > results[1].score > 0
    assert [item.score for item in results] == sorted([item.score for item in results], reverse=True)
    for result in results:
        assert isinstance(result, RetrievalResult)
        assert type(result.score) is float
        assert result.model_dump(exclude={"score"}) == originals[result.chunk_index]
    assert [item.model_dump() for item in chunks] == originals


def test_exact_keyword_and_normalization():
    retriever = BM25Retriever(corpus())
    assert retriever.search("WAVES!!!", 1)[0].chunk_id == "doc_chunk_2"
    assert retriever.search("waves") == retriever.search(" (WAVES) ")
    # No stemming/substrings: 'wave' must not match 'waves'.
    assert all(item.score == 0 for item in retriever.search("wave"))


def test_unicode_and_punctuation_splitting():
    retriever = BM25Retriever([
        chunk("CAFÉ, blue-green_code", 0), chunk("road", 1), chunk("water", 2),
    ])
    for query in ["café", "green", "code"]:
        assert retriever.search(query, 1)[0].chunk_id == "doc_chunk_0"


@pytest.mark.parametrize("top_k, count", [(1, 1), (3, 3), (50, 5)])
def test_top_k(top_k, count):
    assert len(BM25Retriever(corpus()).search("quartz", top_k)) == count


def test_empty_corpus():
    retriever = BM25Retriever()
    assert len(retriever) == 0
    assert retriever.search("valid query") == []


@pytest.mark.parametrize("query", ["", " \n\t", None, 12, "!!!___"])
def test_invalid_query(query):
    with pytest.raises(ValueError, match="query"):
        BM25Retriever().search(query)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        BM25Retriever().search("query", top_k)


def test_rebuild_replaces_and_clears_corpus():
    retriever = BM25Retriever(corpus())
    retriever.build_index([chunk("new text", 20)])
    assert len(retriever) == 1
    assert retriever.search("new")[0].chunk_id == "doc_chunk_20"
    retriever.build_index([])
    assert retriever.search("new") == []


def test_tokenless_corpus_and_empty_documents():
    retriever = BM25Retriever([chunk("", 0), chunk("!!!", 1)])
    assert [item.score for item in retriever.search("query")] == [0.0, 0.0]
    retriever.build_index([chunk("", 0), chunk("quartz", 1), chunk("water", 2)])
    results = retriever.search("quartz")
    assert results[0].chunk_id == "doc_chunk_1"
    assert all(math.isfinite(item.score) for item in results)


def test_no_match_ties_preserve_corpus_order():
    retriever = BM25Retriever(corpus())
    assert [item.chunk_index for item in retriever.search("absent")] == list(range(5))
    assert all(item.score == 0 for item in retriever.search("absent"))


def test_copies_input_and_supports_embedded_chunks():
    original = EmbeddedChunk(**chunk("quartz").model_dump(), embedding=[1.0, 0.0])
    retriever = BM25Retriever([original])
    original.text = "mutated"
    original.embedding[0] = 9
    result = retriever.search("quartz")[0]
    assert result.text == "quartz"
    assert "embedding" not in result.model_dump()
