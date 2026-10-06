import pytest

from app.ingestion import chunk_document, load_document
from app.schemas.document import ExtractedUnit


def unit(text, page_number=None, document_id="doc-1", source="notes.txt"):
    return ExtractedUnit(
        document_id=document_id, source=source, page_number=page_number, text=text
    )


def test_normal_chunking():
    chunks = chunk_document(unit("one two three four five"), 2, 0)
    assert [chunk.text for chunk in chunks] == ["one two", "three four", "five"]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]


def test_overlap():
    words = [str(number) for number in range(1, 191)]
    chunks = chunk_document(unit(" ".join(words)), 100, 20)
    assert [chunk.text.split() for chunk in chunks] == [
        words[:100], words[80:180], words[160:190]
    ]


@pytest.mark.parametrize("text", ["", " \t\n\r\u00a0 "])
def test_empty_text(text):
    assert chunk_document(unit(text)) == []


def test_empty_units():
    assert chunk_document([]) == []


def test_short_text_and_whitespace():
    chunks = chunk_document(unit("  Hello,\t café!\n世界\u00a0 here  "))
    assert len(chunks) == 1
    assert chunks[0].text == "Hello, café! 世界 here"
    assert chunks[0].page_number is None


@pytest.mark.parametrize("count", [4, 7])
def test_no_redundant_overlap_only_tail(count):
    words = [str(number) for number in range(count)]
    chunks = chunk_document(unit(" ".join(words)), 4, 1)
    assert len(chunks) == (1 if count == 4 else 2)
    assert chunks[-1].text.split() == words[-4:]


def test_multiple_units_preserve_metadata_and_input():
    units = [
        unit("a b c", 1, source="report.pdf"),
        unit("  ", 2, source="report.pdf"),
        unit("d e", 3, source="report.pdf"),
    ]
    original = [item.model_dump() for item in units]
    chunks = chunk_document(iter(units), 2, 1)

    assert [chunk.text for chunk in chunks] == ["a b", "b c", "d e"]
    assert [chunk.page_number for chunk in chunks] == [1, 1, 3]
    assert [chunk.chunk_index for chunk in chunks] == [0, 1, 2]
    assert {chunk.document_id for chunk in chunks} == {"doc-1"}
    assert {chunk.source for chunk in chunks} == {"report.pdf"}
    assert [item.model_dump() for item in units] == original


def test_deterministic_ids_across_units():
    units = [unit("a b c"), unit("d e")]
    first = chunk_document(units, 2, 0)
    assert first == chunk_document(units, 2, 0)
    assert [chunk.chunk_id for chunk in first] == [
        "doc-1_chunk_0", "doc-1_chunk_1", "doc-1_chunk_2"
    ]


def test_indices_are_per_document():
    chunks = chunk_document([
        unit("a", document_id="first"),
        unit("b", document_id="second"),
        unit("c", document_id="first"),
    ])
    assert [chunk.chunk_id for chunk in chunks] == [
        "first_chunk_0", "second_chunk_0", "first_chunk_1"
    ]


@pytest.mark.parametrize("size", [0, -1, 1.5, True, "5"])
def test_invalid_chunk_size(size):
    with pytest.raises(ValueError, match="chunk_size"):
        chunk_document([], chunk_size=size, chunk_overlap=0)


@pytest.mark.parametrize("overlap", [-1, 3, 4, 0.5, True, "1"])
def test_invalid_overlap(overlap):
    with pytest.raises(ValueError, match="chunk_overlap"):
        chunk_document([], chunk_size=3, chunk_overlap=overlap)


def test_loader_output_can_be_chunked(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("one two three", encoding="utf-8")
    units = load_document(path)
    chunks = chunk_document(units, 2, 0)
    assert [chunk.text for chunk in chunks] == ["one two", "three"]
    assert all(chunk.document_id == units[0].document_id for chunk in chunks)
    assert all(chunk.source == path.name for chunk in chunks)
    assert all(chunk.page_number is None for chunk in chunks)
