import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.experiments import load_benchmark
from app.schemas.benchmark import BenchmarkItem


def example(**overrides):
    return {
        "question_id": "q1", "question": "What is the example value?",
        "ground_truth": "Example café", "answerable": True,
        "category": "factual", "difficulty": "easy", **overrides,
    }


def write_benchmark(tmp_path, payload):
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def test_valid_loading_and_order(tmp_path):
    payload = [example(question_id="q2", relevant_chunk_ids=["chunk_b", "chunk_a"]), example()]
    items = load_benchmark(str(write_benchmark(tmp_path, payload)))
    assert all(isinstance(item, BenchmarkItem) for item in items)
    assert [item.question_id for item in items] == ["q2", "q1"]
    assert items[0].relevant_chunk_ids == ["chunk_b", "chunk_a"]
    assert items[0].ground_truth == "Example café"
    assert items[1].relevant_chunk_ids == []


@pytest.mark.parametrize("duplicate", ["q1", " q1 "])
def test_duplicate_ids(tmp_path, duplicate):
    path = write_benchmark(tmp_path, [example(), example(question_id=duplicate)])
    with pytest.raises(ValueError, match="Duplicate question_id.*item 2"):
        load_benchmark(path)


@pytest.mark.parametrize("field", ["question_id", "question"])
@pytest.mark.parametrize("value", ["", " \t\n"])
def test_empty_required_text(tmp_path, field, value):
    with pytest.raises(ValueError, match=field):
        load_benchmark(write_benchmark(tmp_path, [example(**{field: value})]))


@pytest.mark.parametrize("ground_truth", [None, "", " \n"])
def test_answerable_requires_ground_truth(tmp_path, ground_truth):
    with pytest.raises(ValueError, match="ground_truth"):
        load_benchmark(write_benchmark(tmp_path, [example(ground_truth=ground_truth)]))


def test_answerable_missing_ground_truth(tmp_path):
    item = example()
    del item["ground_truth"]
    with pytest.raises(ValueError, match="ground_truth"):
        load_benchmark(write_benchmark(tmp_path, [item]))


def test_unanswerable_with_null_ground_truth(tmp_path):
    items = load_benchmark(write_benchmark(tmp_path, [example(
        answerable=False, ground_truth=None, category="unanswerable"
    )]))
    assert items[0].answerable is False
    assert items[0].ground_truth is None
    assert items[0].relevant_chunk_ids == []


@pytest.mark.parametrize("category", ["factual", "definition", "comparison", "multi_hop", "reasoning", "unanswerable"])
def test_supported_categories(category):
    assert BenchmarkItem(**example(category=category)).category == category


@pytest.mark.parametrize("difficulty", ["easy", "medium", "hard"])
def test_supported_difficulties(difficulty):
    assert BenchmarkItem(**example(difficulty=difficulty)).difficulty == difficulty


@pytest.mark.parametrize("overrides", [
    {"category": "unknown"}, {"difficulty": "expert"}, {"answerable": "false"},
    {"relevant_chunk_ids": [""]}, {"unexpected_field": 1},
])
def test_invalid_schema(tmp_path, overrides):
    with pytest.raises(ValueError, match="Invalid benchmark item 1") as caught:
        load_benchmark(write_benchmark(tmp_path, [example(**overrides)]))
    assert isinstance(caught.value.__cause__, ValidationError)


def test_default_chunk_lists_are_independent():
    first, second = BenchmarkItem(**example()), BenchmarkItem(**example())
    first.relevant_chunk_ids.append("chunk_1")
    assert second.relevant_chunk_ids == []


def test_malformed_json(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('[{"question_id":', encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid UTF-8 JSON benchmark") as caught:
        load_benchmark(path)
    assert isinstance(caught.value.__cause__, json.JSONDecodeError)


@pytest.mark.parametrize("payload", [{"items": []}, None, "text"])
def test_root_must_be_array(tmp_path, payload):
    with pytest.raises(ValueError, match="JSON array"):
        load_benchmark(write_benchmark(tmp_path, payload))


def test_non_object_item(tmp_path):
    with pytest.raises(ValueError, match="item 2"):
        load_benchmark(write_benchmark(tmp_path, [example(), 123]))


def test_empty_benchmark(tmp_path):
    assert load_benchmark(write_benchmark(tmp_path, [])) == []


def test_synthetic_sample():
    path = Path(__file__).resolve().parents[1] / "data/benchmarks/synthetic_sample.json"
    items = load_benchmark(path)
    assert len(items) == 3
    assert sum(item.answerable for item in items) == 2
    assert items[-1].ground_truth is None
