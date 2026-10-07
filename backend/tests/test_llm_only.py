from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from app.main import create_app
from app.generation.llm_only import generate_llm_only, LLM_ONLY_PROMPT
from app.generation.llm_service import LLMError
from app.generation.prompts import INSUFFICIENT_INFORMATION
from app.generation.rag_pipeline import RAGPipeline
from app.schemas.generation import LLMGenerationResult
from app.schemas.benchmark import BenchmarkItem
from app.schemas.retrieval import RetrievalResult
from app.experiments.models import ExperimentConfig, ExperimentResult
from app.experiments.experiment_runner import ExperimentRunner
from app.experiments.comparison_runner import ComparisonRunner
from app.evaluation.answer_correctness import CorrectnessResult
from app.evaluation.groundedness import GroundednessResult
from app.evaluation.citation_accuracy import CitationAccuracyResult
from app.evaluation.hallucination_refusal import HallucinationRefusalResult, evaluate_refusal_only
from app.evaluation.failure_analysis import analyze_experiment


def answer(text='Known answer.'):
    return LLMGenerationResult(text=text, model='test-model', latency_ms=12,
                               prompt_tokens=10, completion_tokens=4, total_tokens=14, cost=.01)


def item(identifier='q1', answerable=True):
    return BenchmarkItem(question_id=identifier, question='Question?', answerable=answerable,
                         ground_truth='Known answer.' if answerable else None,
                         relevant_chunk_ids=['c1'] if answerable else [],
                         category='factual' if answerable else 'unanswerable', difficulty='easy')


def config(strategy='llm_only', **kwargs):
    return ExperimentConfig(experiment_name=strategy, retrieval_strategy=strategy,
                            model='test-model', benchmark_name='test', **kwargs)


@pytest.fixture
def runner():
    llm = Mock(model='test-model')
    llm.generate_result.return_value = answer()
    pipeline = RAGPipeline(Mock(), llm, Mock())
    correctness = Mock()
    correctness.evaluate_benchmark_item.side_effect = lambda i, a: None if not i.answerable else CorrectnessResult(
        score=0 if a==INSUFFICIENT_INFORMATION else 1,
        label='incorrect' if a==INSUFFICIENT_INFORMATION else 'correct', reason='Compared to ground truth')
    return ExperimentRunner(retrievers={}, generator=pipeline, correctness=correctness,
                            groundedness=Mock(), citation_accuracy=Mock(), hallucination_refusal=Mock(),
                            reranker=Mock(), query_rewriter=Mock())


def test_query_without_retriever_and_usage(runner, monkeypatch):
    from app.generation import llm_only
    monkeypatch.setattr(llm_only, 'perf_counter', Mock(side_effect=[1., 1.25]))
    with TestClient(create_app(pipeline=runner.generator)) as client:
        response = client.post('/query', json={'question':'Question?', 'retrieval_strategy':'llm_only', 'top_k':99})
    assert response.status_code == 200, response.text
    data = response.json()
    assert data['retrieval_strategy'] == 'llm_only'
    assert data['retrieved_chunks'] == data['citations'] == []
    assert data['answer'] == 'Known answer.' and data['model'] == 'test-model'
    assert data['latency_ms'] == 250
    assert data['usage'] == {'latency_ms':12., 'prompt_tokens':10, 'completion_tokens':4, 'total_tokens':14, 'cost':.01}
    runner.generator.llm.generate_result.assert_called_once_with(LLM_ONLY_PROMPT, 'Question?')
    assert not runner.generator.vector_store.mock_calls
    assert not runner.generator.embedding_service.mock_calls


def test_experiment_na_metrics_and_refusal_denominators(runner, tmp_path):
    runner.generator.llm.generate_result.side_effect = [answer(), answer(INSUFFICIENT_INFORMATION), answer(INSUFFICIENT_INFORMATION), answer()]
    result = runner.run([item('a'), item('b'), item('c',False), item('d',False)], config(), output_dir=tmp_path)
    for q in result.questions:
        assert q.retrieved_chunk_ids == q.generation.retrieved_chunks == q.generation.citations == []
        assert q.reciprocal_rank is None and all(v is None for v in q.recall_at_k.values())
        assert q.groundedness is None
        assert q.citation_accuracy.skipped and q.citation_accuracy.score is None
    summary = result.summary
    assert summary.retrieval.evaluated_questions == 0 and summary.retrieval.skipped_questions == 4
    assert summary.retrieval.mrr is None
    assert summary.groundedness.mean_groundedness is None and summary.groundedness.evaluated_count == 0
    assert summary.citation_accuracy.mean_citation_accuracy is None and summary.citation_accuracy.skipped_count == 4
    assert summary.correctness.mean_score == .5
    behavior = summary.hallucination_refusal
    assert behavior.hallucination_rate is None
    assert behavior.correct_refusal_rate == behavior.failed_refusal_rate == behavior.unnecessary_refusal_rate == .5
    assert behavior.answerable_count == behavior.unanswerable_count == 2
    assert behavior.support_unassessed_count == 1
    assert summary.total_tokens == 56 and summary.total_cost == pytest.approx(.04)
    for component in (runner.groundedness, runner.citation_accuracy, runner.hallucination_refusal, runner.reranker, runner.query_rewriter, runner.generator.vector_store, runner.generator.embedding_service):
        assert not component.mock_calls
    analysis = analyze_experiment(result)
    assert analysis.questions[0].failure_categories == []
    assert set(analysis.questions[1].failure_categories) == {'generation_incorrect', 'incorrect_refusal'}
    assert analysis.questions[3].failure_categories == ['failed_refusal']
    assert analysis.summary.category_counts['retrieval_miss'] == analysis.summary.category_counts['retrieval_low_rank'] == 0
    assert ExperimentResult.model_validate_json((tmp_path/f'{result.experiment_id}.json').read_text()) == result


def test_comparison_dense_and_baseline(runner, tmp_path):
    chunk = RetrievalResult(chunk_id='c1', document_id='doc', source='source.txt', page_number=None, chunk_index=0, text='Evidence', score=.9)
    dense = Mock(); dense.search_text.return_value = [chunk]
    runner.retrievers['dense'] = dense
    runner.groundedness.evaluate.return_value = GroundednessResult(score=1, label='grounded', reason='Supported')
    runner.citation_accuracy.evaluate.return_value = CitationAccuracyResult(score=1, label='accurate', reason='Supported', valid_citations=1, invalid_citations=0, total_citations=1)
    runner.hallucination_refusal.evaluate.return_value = HallucinationRefusalResult(label='supported_answer', hallucinated=False, refused=False, reason='Supported', unsupported_claims=[])
    result = ComparisonRunner(runner).run([item()], [config('dense'), config()], output_dir=tmp_path/'comparisons', experiment_output_dir=tmp_path/'experiments')
    dense.search_text.assert_called_once()
    assert result.metrics[0].mrr == 1
    assert result.metrics[1].mrr is None
    assert result.metrics[1].mean_groundedness is None
    assert result.metrics[1].mean_citation_accuracy is None
    assert result.table_rows()[1]['mean_correctness'] == 1
    runner.groundedness.evaluate.assert_called_once()


@pytest.mark.parametrize('field', ['reranker_enabled', 'query_rewrite_enabled'])
def test_retrieval_modifiers_rejected(field):
    with pytest.raises(ValidationError, match='llm_only'):
        config(**{field:True})


@pytest.mark.parametrize('text', ['', '  '])
def test_empty_input_never_calls_provider(text):
    llm = Mock()
    with pytest.raises(ValueError): generate_llm_only(llm, text)
    assert not llm.mock_calls


def test_provider_error_propagates(runner, tmp_path):
    runner.generator.llm.generate_result.side_effect = LLMError('provider failed')
    with pytest.raises(LLMError): runner.run([item()], config(), output_dir=tmp_path)
    assert not list(tmp_path.glob('*.json'))


def test_missing_usage_stays_none(runner, tmp_path):
    runner.generator.llm.generate_result.return_value = LLMGenerationResult(text='Answer', model='test-model')
    summary = runner.run([item()], config(), output_dir=tmp_path).summary
    assert summary.total_tokens is None and summary.total_cost is None


def test_refusal_only_does_not_claim_support():
    result = evaluate_refusal_only(True, 'A substantive answer')
    assert result.label == 'unassessed_answer' and result.hallucinated is None
    assert not result.unsupported_claims
    with pytest.raises(ValidationError):
        HallucinationRefusalResult(label='unassessed_answer', hallucinated=False, refused=False, reason='Invalid', unsupported_claims=[])
