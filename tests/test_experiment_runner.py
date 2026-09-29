import json
from unittest.mock import Mock
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.experiments.experiment_runner import ExperimentRunner
from app.experiments.models import ExperimentConfig, ExperimentResult
from app.generation import llm_service
from app.generation.prompts import INSUFFICIENT_INFORMATION
from app.schemas.benchmark import BenchmarkItem
from app.schemas.generation import Citation, GenerationUsage, RAGResponse
from app.schemas.retrieval import RetrievalResult
from app.evaluation.answer_correctness import CorrectnessResult
from app.evaluation.groundedness import GroundednessResult
from app.evaluation.citation_accuracy import CitationAccuracyResult
from app.evaluation.hallucination_refusal import HallucinationRefusalResult


def config(**kwargs):
    return ExperimentConfig(**{
        "experiment_name": "synthetic-test", "retrieval_strategy": "dense", "top_k": 1,
        "model": "test-model", "benchmark_name": "synthetic", **kwargs,
    })


def item(answerable=True, question_id="q1"):
    return BenchmarkItem(question_id=question_id, question="Original question?", answerable=answerable,
                         ground_truth="Reference" if answerable else None,
                         relevant_chunk_ids=["c1"] if answerable else [], category="factual", difficulty="easy")


def chunk():
    return RetrievalResult(chunk_id="c1", document_id="d1", source="example.txt", page_number=None,
                           chunk_index=0, text="Synthetic evidence", score=0.9)


@pytest.fixture
def components(monkeypatch):
    monkeypatch.setattr(llm_service, "OpenAI", Mock(side_effect=AssertionError("No real client")))
    retrievers = {name: Mock() for name in ("dense", "bm25", "hybrid")}
    for retriever in retrievers.values():
        retriever.search.return_value = [chunk()]
        retriever.search_text.return_value = [chunk()]
    generator = Mock()
    generator.llm.model = "test-model"
    generator.generate_from_chunks.return_value = RAGResponse(
        question="Original question?", answer="Supported answer", citations=[Citation(chunk_id="c1", source="example.txt", page_number=None)],
        retrieved_chunks=[chunk()], model="test-model", latency_ms=3,
        usage=GenerationUsage(latency_ms=2, prompt_tokens=10, completion_tokens=5, total_tokens=15, cost=0.01),
    )
    correctness = Mock()
    correctness.evaluate_benchmark_item.side_effect = lambda i, a: CorrectnessResult(score=1, label="correct", reason="Matches") if i.answerable else None
    groundedness = Mock()
    groundedness.evaluate.return_value = GroundednessResult(score=1, label="grounded", reason="Supported")
    citations = Mock()
    citations.evaluate.return_value = CitationAccuracyResult(score=1, label="accurate", reason="Supports",
                                                           valid_citations=1, invalid_citations=0, total_citations=1)
    behavior = Mock()
    behavior.evaluate.return_value = HallucinationRefusalResult(label="supported_answer", hallucinated=False,
                                                              refused=False, reason="Supported", unsupported_claims=[])
    reranker = Mock()
    reranker.rerank.return_value = [chunk()]
    rewriter = Mock()
    rewriter.rewrite_query.return_value = "Rewritten retrieval query"
    runner = ExperimentRunner(retrievers=retrievers, generator=generator, correctness=correctness,
                              groundedness=groundedness, citation_accuracy=citations,
                              hallucination_refusal=behavior, reranker=reranker, query_rewriter=rewriter)
    return runner


@pytest.mark.parametrize("strategy", ["dense", "bm25", "hybrid"])
def test_strategy_flow_persistence_and_aggregate(strategy, components, tmp_path):
    runner = components
    result = runner.run([item()], config(retrieval_strategy=strategy), output_dir=tmp_path)
    retriever = runner.retrievers[strategy]
    if strategy == "dense":
        retriever.search_text.assert_called_once_with("Original question?", top_k=1)
    elif strategy == "bm25":
        retriever.search.assert_called_once_with("Original question?", top_k=1)
    else:
        retriever.search.assert_called_once_with("Original question?", candidate_k=20, final_top_k=1)
    runner.generator.generate_from_chunks.assert_called_once_with("Original question?", [chunk()])
    runner.reranker.rerank.assert_not_called()
    runner.query_rewriter.rewrite_query.assert_not_called()
    assert result.questions[0].recall_at_k == {1: 1.0, 3: 1.0, 5: 1.0}
    assert result.questions[0].reciprocal_rank == 1.0
    summary = result.summary
    assert summary.evaluated_question_count == 1
    assert summary.retrieval.mrr == 1
    assert summary.correctness.mean_score == summary.groundedness.mean_groundedness == 1
    assert summary.citation_accuracy.mean_citation_accuracy == 1
    assert summary.hallucination_refusal.hallucination_rate == 0
    assert summary.total_tokens == 15 and summary.total_cost == 0.01
    path = tmp_path / f"{result.experiment_id}.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["config"]["retrieval_strategy"] == strategy
    assert saved["questions"][0]["generation"]["answer"] == "Supported answer"
    assert ExperimentResult.model_validate_json(path.read_text()) == result
    assert result.timestamp.utcoffset().total_seconds() == 0


def test_modifiers_original_question_and_clock(components, tmp_path, monkeypatch):
    from app.experiments import experiment_runner
    monkeypatch.setattr(experiment_runner, "perf_counter", Mock(side_effect=[1.0, 1.25]))
    result = components.run([item()], config(reranker_enabled=True, query_rewrite_enabled=True), output_dir=tmp_path)
    components.retrievers["dense"].search_text.assert_called_once_with("Rewritten retrieval query", top_k=20)
    components.reranker.rerank.assert_called_once_with("Rewritten retrieval query", [chunk()], top_k=1)
    components.generator.generate_from_chunks.assert_called_once_with("Original question?", [chunk()])
    assert result.questions[0].rewritten_query == "Rewritten retrieval query"
    assert result.questions[0].original_question == "Original question?"
    assert result.summary.average_latency_ms == 250


def test_unanswerable_preserves_skips(components, tmp_path):
    response = components.generator.generate_from_chunks.return_value
    components.generator.generate_from_chunks.return_value = response.model_copy(update={
        "answer": INSUFFICIENT_INFORMATION, "citations": [],
    })
    components.citation_accuracy.evaluate.return_value = CitationAccuracyResult(
        score=None, label="not_applicable", skipped=True, reason="Refusal", valid_citations=0, invalid_citations=0, total_citations=0,
    )
    components.hallucination_refusal.evaluate.return_value = HallucinationRefusalResult(
        label="correct_refusal", hallucinated=False, refused=True, reason="Refused", unsupported_claims=[],
    )
    result = components.run([item(False)], config(), output_dir=tmp_path)
    assert result.questions[0].correctness is None
    assert result.questions[0].reciprocal_rank is None
    assert result.summary.correctness.skipped_count == 1
    assert result.summary.retrieval.skipped_questions == 1
    assert result.summary.citation_accuracy.skipped_count == 1
    assert result.summary.hallucination_refusal.correct_refusal_rate == 1
    components.hallucination_refusal.evaluate.assert_called_once_with(False, INSUFFICIENT_INFORMATION, [chunk()])


def test_usage_totals_and_unknowns(components, tmp_path):
    response = components.generator.generate_from_chunks.return_value
    second = response.model_copy(update={"usage": GenerationUsage(prompt_tokens=4, completion_tokens=None, total_tokens=None, cost=None)})
    components.generator.generate_from_chunks.side_effect = [response, second]
    result = components.run([item(question_id="q1"), item(question_id="q2")], config(), output_dir=tmp_path)
    assert result.summary.total_prompt_tokens == 14
    assert result.summary.total_completion_tokens is None
    assert result.summary.total_tokens is None
    assert result.summary.total_cost is None


def test_no_overwrite_or_unnecessary_calls(components, tmp_path):
    identifier = uuid4()
    first = components.run([item()], config(), output_dir=tmp_path, experiment_id=identifier)
    path = tmp_path / f"{identifier}.json"
    original = path.read_bytes()
    components.generator.reset_mock()
    with pytest.raises(FileExistsError):
        components.run([item()], first.config, output_dir=tmp_path, experiment_id=identifier)
    assert path.read_bytes() == original
    components.generator.generate_from_chunks.assert_not_called()


@pytest.mark.parametrize("component,method", [
    ("generator", "generate_from_chunks"), ("correctness", "evaluate_benchmark_item"),
    ("groundedness", "evaluate"), ("citation_accuracy", "evaluate"), ("hallucination_refusal", "evaluate"),
])
def test_component_errors_propagate(component, method, components, tmp_path):
    error = RuntimeError("component failed")
    getattr(getattr(components, component), method).side_effect = error
    with pytest.raises(RuntimeError) as caught:
        components.run([item()], config(), output_dir=tmp_path)
    assert caught.value is error
    assert list(tmp_path.glob("*.json")) == []


@pytest.mark.parametrize("patch", [{"top_k": 0}, {"candidate_k": True}, {"retrieval_strategy": "other"}, {"model": " "}, {"chunk_size": -1}])
def test_config_validation(patch):
    with pytest.raises(ValidationError):
        config(**patch)


def test_preflight_validation(components, tmp_path):
    components.reranker = None
    with pytest.raises(ValueError, match="reranker"):
        components.run([item()], config(reranker_enabled=True), output_dir=tmp_path)
    with pytest.raises(ValueError, match="Duplicate"):
        components.run([item(), item()], config(), output_dir=tmp_path)
    with pytest.raises(ValueError, match="model"):
        components.run([item()], config(model="wrong"), output_dir=tmp_path)
    components.generator.generate_from_chunks.assert_not_called()


def test_empty_benchmark_and_hash_stability(components, tmp_path):
    first = components.run([], config(), output_dir=tmp_path)
    second = components.run([], config(), output_dir=tmp_path)
    assert first.config_sha256 == second.config_sha256
    assert first.benchmark_sha256 == second.benchmark_sha256
    assert first.experiment_id != second.experiment_id
    assert first.summary.evaluated_question_count == 0
    assert first.summary.average_latency_ms is None
    assert first.summary.total_cost == 0


@pytest.mark.parametrize("stage", ["retrieval", "rewriter", "reranker"])
def test_retrieval_modifier_errors(stage, components, tmp_path):
    methods = {
        "retrieval": components.retrievers["dense"].search_text,
        "rewriter": components.query_rewriter.rewrite_query,
        "reranker": components.reranker.rerank,
    }
    error = RuntimeError(f"{stage} failed")
    methods[stage].side_effect = error
    with pytest.raises(RuntimeError) as caught:
        components.run([item()], config(query_rewrite_enabled=True, reranker_enabled=True), output_dir=tmp_path)
    assert caught.value is error
    components.generator.generate_from_chunks.assert_not_called()
    assert list(tmp_path.glob("*.json")) == []


def test_complete_usage_sums_and_order(components, tmp_path):
    result = components.run([item(question_id="second"), item(question_id="first")], config(), output_dir=tmp_path)
    assert [question.question_id for question in result.questions] == ["second", "first"]
    assert result.summary.total_prompt_tokens == 20
    assert result.summary.total_completion_tokens == 10
    assert result.summary.total_tokens == 30
    assert result.summary.total_cost == pytest.approx(0.02)
