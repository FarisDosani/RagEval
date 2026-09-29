from uuid import uuid4

import pytest

from app.evaluation.failure_analysis import analyze_experiment, analyze_question, summarize_failures
from app.experiments.models import ExperimentConfig, ExperimentResult, QuestionExperimentResult
from app.schemas.benchmark import BenchmarkItem


def item(**updates):
    return BenchmarkItem.model_validate({
        "question_id": "q1", "question": "Example?", "ground_truth": "Example",
        "answerable": True, "relevant_chunk_ids": ["relevant"],
        "category": "factual", "difficulty": "easy", **updates,
    })


def question(**updates):
    return QuestionExperimentResult.model_validate({
        "question_id": "q1", "original_question": "Example?", "rewritten_query": None,
        "retrieved_chunk_ids": ["relevant"], "recall_at_k": {1: 1.0}, "reciprocal_rank": 1.0,
        "generation": {"question": "Example?", "answer": "Example", "citations": [],
                       "retrieved_chunks": [], "model": "synthetic", "latency_ms": 0},
        "correctness": {"score": 1.0, "label": "correct", "reason": "Matches"},
        "groundedness": {"score": 1.0, "label": "grounded", "reason": "Supported"},
        "citation_accuracy": {"score": 1.0, "label": "accurate", "reason": "Supported",
                              "valid_citations": 1, "invalid_citations": 0, "total_citations": 1},
        "hallucination_refusal": {"label": "supported_answer", "hallucinated": False,
                                  "refused": False, "reason": "Supported", "unsupported_claims": []},
        "latency_ms": 0, **updates,
    })


def analyze(result, benchmark=None, top_k=1):
    return analyze_question(result, benchmark or item(), top_k=top_k)


def test_retrieval_miss_and_empty_retrieval():
    for ids in (["other"], []):
        result = analyze(question(retrieved_chunk_ids=ids, recall_at_k={1: 0.0}, reciprocal_rank=0.0))
        assert result.failure_categories == ["retrieval_miss"]


def test_low_rank_requires_recorded_evidence_beyond_top_k():
    q = question(retrieved_chunk_ids=["other", "relevant"], recall_at_k={1: 0.0}, reciprocal_rank=0.5)
    assert analyze(q).failure_categories == ["retrieval_low_rank"]
    assert "rank 2" in analyze(q).notes[0]
    assert analyze(q, top_k=2).failure_categories == []
    # A small-cutoff miss and absent rank metric do not imply a low rank.
    assert analyze(question(recall_at_k={1: None}, reciprocal_rank=None)).failure_categories == []


@pytest.mark.parametrize("label,score", [("incorrect", 0.0), ("partially_correct", 0.5)])
def test_generation_correctness(label, score):
    result = analyze(question(correctness={"label": label, "score": score, "reason": "Mismatch"}))
    assert result.failure_categories == ["generation_incorrect"]
    assert label in result.notes[0]


@pytest.mark.parametrize("label,score", [("ungrounded", 0.0), ("partially_grounded", 0.5)])
def test_ungrounded(label, score):
    assert analyze(question(groundedness={"label": label, "score": score, "reason": "Unsupported"})).failure_categories == ["generation_ungrounded"]


def test_citation_failure_and_not_applicable():
    citation = question().citation_accuracy.model_dump()
    citation.update(score=0.5, label="partially_accurate")
    assert analyze(question(citation_accuracy=citation)).failure_categories == ["citation_failure"]
    citation.update(score=None, label="not_applicable", skipped=True)
    assert analyze(question(citation_accuracy=citation)).failure_categories == []


@pytest.mark.parametrize("label,hallucinated,refused,expected", [
    ("hallucinated_answer", True, False, "hallucination"),
    ("incorrect_refusal", False, True, "incorrect_refusal"),
    ("failed_refusal", None, False, "failed_refusal"),
    ("correct_refusal", False, True, None),
])
def test_behavior_labels(label, hallucinated, refused, expected):
    behavior = dict(label=label, hallucinated=hallucinated, refused=refused,
                    reason="Recorded", unsupported_claims=["Unsupported"] if hallucinated else [])
    result = analyze(question(hallucination_refusal=behavior), item(answerable=False) if label in ("correct_refusal", "failed_refusal") else item())
    assert result.failure_categories == ([expected] if expected else [])


def test_multiple_failures_and_aggregate():
    q = question(retrieved_chunk_ids=[], correctness={"score": 0.0, "label": "incorrect", "reason": "Wrong"},
                 groundedness={"score": 0.0, "label": "ungrounded", "reason": "Unsupported"})
    failure = analyze(q)
    assert failure.failure_categories == ["retrieval_miss", "generation_incorrect", "generation_ungrounded"]
    success = analyze(question(question_id="q2"), item(question_id="q2"))
    assert success.failure_categories == []
    summary = summarize_failures(iter([failure, success]))
    assert summary.total_questions_analyzed == 2
    assert summary.questions_with_failures == summary.failure_free_questions == 1
    assert summary.category_counts["retrieval_miss"] == 1
    assert summary.category_rates["retrieval_miss"] == 0.5
    assert summary.category_rates["citation_failure"] == 0.0
    assert all(value is None for value in summarize_failures([]).category_rates.values())


def test_missing_and_not_applicable_retrieval_metrics():
    q = question(retrieved_chunk_ids=[], recall_at_k={}, reciprocal_rank=None, correctness=None)
    assert analyze(q, item(answerable=False)).failure_categories == []
    result = analyze(q, item(relevant_chunk_ids=[]))
    assert result.failure_categories == [] and "not assessed" in result.notes[0]
    result = analyze_question(q, None, top_k=1)
    assert result.failure_categories == [] and "unavailable" in result.notes[0]


def test_experiment_postprocessing_order_and_no_execution(monkeypatch):
    # Any accidental service execution must fail; only stored outputs are used.
    from app.generation.llm_service import LLMService
    def forbidden(*args, **kwargs):
        raise AssertionError("No LLM calls")
    monkeypatch.setattr(LLMService, "generate", forbidden)
    questions = [question(question_id="q2"), question()]
    # Aggregate experiment metrics are deliberately omitted: analysis never uses them.
    experiment = ExperimentResult.model_construct(
        experiment_id=uuid4(), questions=questions, benchmark=[item(), item(question_id="q2")],
        config=ExperimentConfig(experiment_name="test", retrieval_strategy="dense", model="synthetic", benchmark_name="test"),
    )
    before = [q.model_dump() for q in questions]
    result = analyze_experiment(experiment)
    assert result.experiment_id == experiment.experiment_id
    assert [q.question_id for q in result.questions] == ["q2", "q1"]
    assert result.summary.failure_free_questions == 2
    assert [q.model_dump() for q in questions] == before
    experiment.questions = [questions[0], questions[0]]
    with pytest.raises(ValueError, match="Duplicate result"):
        analyze_experiment(experiment)


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        analyze(question(), top_k=top_k)


def test_mismatched_question_id():
    with pytest.raises(ValueError, match="IDs must match"):
        analyze(question(), item(question_id="other"))
