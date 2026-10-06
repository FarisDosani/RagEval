import pytest

from app.evaluation import (
    evaluate_retrieval, mean_reciprocal_rank, recall_at_k, reciprocal_rank,
)
from app.schemas.benchmark import BenchmarkItem


def item(question_id, relevant=(), answerable=True):
    return BenchmarkItem(
        question_id=question_id, question="Example question", answerable=answerable,
        ground_truth="Example answer" if answerable else None,
        relevant_chunk_ids=list(relevant), category="factual" if answerable else "unanswerable",
        difficulty="easy",
    )


def test_first_rank():
    assert recall_at_k(["a"], ["a", "b"], 1) == 1.0
    assert reciprocal_rank(["a"], ["a", "b"]) == 1.0


def test_below_first_rank():
    assert recall_at_k(["a"], ["x", "y", "a"], 2) == 0.0
    assert recall_at_k(["a"], ["x", "y", "a"], 3) == 1.0
    assert reciprocal_rank(["a"], ["x", "y", "a"]) == pytest.approx(1 / 3)


def test_multiple_relevant_ids_use_hit_not_fractional_recall():
    assert recall_at_k(["a", "b", "c"], ["b"], 1) == 1.0
    assert reciprocal_rank(["a", "b"], ["x", "b", "a"]) == 0.5


@pytest.mark.parametrize("retrieved", [[], ["x", "y"]])
def test_no_hit(retrieved):
    assert recall_at_k(["a"], retrieved, 5) == 0.0
    assert reciprocal_rank(["a"], retrieved) == 0.0


def test_k_larger_than_ranking():
    assert recall_at_k(["a"], ["x", "a"], 100) == 1.0


@pytest.mark.parametrize("k", [0, -1, True, 1.5, "3", None])
def test_invalid_k(k):
    with pytest.raises(ValueError, match="k must"):
        recall_at_k([], [], k)
    with pytest.raises(ValueError, match="k must"):
        evaluate_retrieval([], {}, [k])


def test_no_relevant_labels_are_not_applicable():
    assert recall_at_k([], ["x"], 1) is None
    assert reciprocal_rank([], ["x"]) is None


def test_mean_reciprocal_rank():
    pairs = [(["a"], ["a"]), (["b"], ["x", "b"]), (["c"], []), ([], ["x"])]
    assert mean_reciprocal_rank(iter(pairs)) == pytest.approx(0.5)


def test_aggregate_and_skip_counts():
    benchmark = [
        item("q1", ["a"]), item("q2", ["b", "c"]), item("q3", ["d"]),
        item("q4", answerable=False), item("q5"),
        item("q6", ["inconsistent-label"], answerable=False),
    ]
    result = evaluate_retrieval(benchmark, {
        "q1": iter(["a"]), "q2": iter(["x", "y", "c"]), "q3": [],
    })
    assert result.evaluated_questions == 3
    assert result.skipped_questions == 3
    assert result.recall_at_k == pytest.approx({1: 1 / 3, 3: 2 / 3, 5: 2 / 3})
    assert result.mrr == pytest.approx((1 + 1 / 3) / 3)


def test_configurable_k_and_full_ranking_mrr():
    result = evaluate_retrieval(
        [item("q", ["a"])], {"q": ["x", "y", "a"]}, k_values=iter([2, 4, 2])
    )
    assert result.recall_at_k == {2: 0.0, 4: 1.0}
    assert result.mrr == pytest.approx(1 / 3)


@pytest.mark.parametrize("benchmark", [[], [item("q", answerable=False)], [item("q")]])
def test_no_evaluable_questions(benchmark):
    result = evaluate_retrieval(benchmark, {})
    assert result.evaluated_questions == 0
    assert result.skipped_questions == len(benchmark)
    assert result.recall_at_k == {1: None, 3: None, 5: None}
    assert result.mrr is None
    assert mean_reciprocal_rank([]) is None
    assert mean_reciprocal_rank([([], [])]) is None


def test_missing_results_are_distinct_from_empty_retrieval():
    with pytest.raises(ValueError, match="Missing retrieval results.*q"):
        evaluate_retrieval([item("q", ["a"])], {})
    result = evaluate_retrieval([item("q", ["a"])], {"q": []})
    assert result.evaluated_questions == 1
    assert result.mrr == 0.0


def test_duplicate_benchmark_ids_rejected():
    with pytest.raises(ValueError, match="Duplicate benchmark"):
        evaluate_retrieval([item("q"), item("q")], {})


def test_duplicate_retrieval_positions_preserved():
    assert reciprocal_rank(["a", "a"], ["x", "x", "a"]) == pytest.approx(1 / 3)
    assert recall_at_k(["a", "a"], ["x", "x", "a"], 2) == 0.0


def test_mrr_without_recall_cutoffs():
    result = evaluate_retrieval([item("q", ["a"])], {"q": ["a"]}, k_values=[])
    assert result.recall_at_k == {}
    assert result.mrr == 1.0
