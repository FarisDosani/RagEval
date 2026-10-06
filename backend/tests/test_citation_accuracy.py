import json
from unittest.mock import Mock

import pytest

from app.evaluation.citation_accuracy import (
    CITATION_SUPPORT_PROMPT, CitationAccuracyEvaluator, CitationAccuracyParsingError,
    summarize_citation_accuracy,
)
from app.generation import llm_service
from app.generation.llm_service import LLMError, LLMService
from app.generation.prompts import INSUFFICIENT_INFORMATION
from app.schemas.generation import Citation
from app.schemas.retrieval import RetrievalResult


@pytest.fixture(autouse=True)
def block_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("No real API client in citation tests"))
    monkeypatch.setattr(llm_service, "OpenAI", constructor)
    return constructor


def citation(**overrides):
    return Citation(**{"chunk_id": "c1", "source": "library.pdf", "page_number": 2, **overrides})


def chunk(**overrides):
    return RetrievalResult(**{
        "chunk_id": "c1", "source": "library.pdf", "page_number": 2, "document_id": "d1",
        "chunk_index": 0, "text": "The library opens at 09:00.", "score": 0.9, **overrides,
    })


def judge(score=1.0, label="accurate"):
    llm = Mock(spec=LLMService)
    llm.generate.return_value = json.dumps({"score": score, "label": label, "reason": "Support assessment."})
    return CitationAccuracyEvaluator(llm), llm


@pytest.mark.parametrize("score,label", [(1.0, "accurate"), (0.5, "partially_accurate"), (0.0, "inaccurate")])
def test_valid_citation_support(score, label, block_client):
    evaluator, llm = judge(score, label)
    result = evaluator.evaluate("The library opens at 09:00.", [citation()], [chunk()])
    assert (result.score, result.label) == (score, label)
    assert (result.valid_citations, result.invalid_citations, result.total_citations) == (1, 0, 1)
    assert not result.skipped
    assert evaluator.llm is llm
    block_client.assert_not_called()


@pytest.mark.parametrize("overrides", [{"chunk_id": "missing"}, {"source": "wrong.pdf"}, {"page_number": 9}, {"page_number": None}])
def test_invalid_metadata_never_calls_judge(overrides):
    evaluator, llm = judge()
    result = evaluator.evaluate("Factual answer", [citation(**overrides)], [chunk()])
    assert result.score == 0.0
    assert (result.valid_citations, result.invalid_citations, result.total_citations) == (0, 1, 1)
    llm.generate.assert_not_called()


@pytest.mark.parametrize("support_score,support_label,expected", [
    (1.0, "accurate", 0.5), (0.5, "partially_accurate", 0.5), (0.0, "inaccurate", 0.0),
])
def test_mixed_validity_caps_support(support_score, support_label, expected):
    evaluator, llm = judge(support_score, support_label)
    result = evaluator.evaluate("Answer", [citation(), citation(chunk_id="bad")], [chunk()])
    assert result.score == expected
    assert (result.valid_citations, result.invalid_citations, result.total_citations) == (1, 1, 2)
    assert "metadata validation" in result.reason
    assert len(json.loads(llm.generate.call_args.args[1])["cited_evidence"]) == 1


def test_prompt_contains_only_cited_evidence_and_reuses_service():
    evaluator, llm = judge()
    chunks = [chunk(), chunk(chunk_id="uncited", text="UNRELATED SECRET CONTEXT")]
    before = [item.model_dump() for item in chunks]
    for _ in range(2):
        evaluator.evaluate("Answer", [citation()], iter(chunks))
    system, user = llm.generate.call_args.args
    assert system == CITATION_SUPPORT_PROMPT
    assert "Do not use outside knowledge" in system
    assert "each citation supports factual claim(s)" in system
    assert "Return only one JSON object" in system
    assert "UNRELATED" not in user
    assert json.loads(user) == {"generated_answer": "Answer", "cited_evidence": [
        {**citation().model_dump(), "text": chunk().text}
    ]}
    assert llm.generate.call_count == 2
    assert [item.model_dump() for item in chunks] == before


def test_no_citations():
    evaluator, llm = judge()
    result = evaluator.evaluate("The library opens at 09:00.", [], [chunk()])
    assert result.score == 0.0
    assert result.total_citations == 0
    llm.generate.assert_not_called()


@pytest.mark.parametrize("citations", [[], [citation()], [citation(chunk_id="bad")]])
def test_refusal_skipped(citations):
    evaluator, llm = judge()
    result = evaluator.evaluate(INSUFFICIENT_INFORMATION, citations, [chunk()])
    assert result.skipped and result.score is None
    assert result.label == "not_applicable"
    assert result.total_citations == len(citations)
    llm.generate.assert_not_called()


def test_refusal_plus_claim_not_skipped():
    evaluator, _ = judge()
    result = evaluator.evaluate(INSUFFICIENT_INFORMATION + " It opens at 09:00.", [], [])
    assert not result.skipped and result.score == 0.0


def test_no_retrieved_or_blank_evidence():
    evaluator, llm = judge()
    assert evaluator.evaluate("Answer", [citation()], []).invalid_citations == 1
    result = evaluator.evaluate("Answer", [citation()], [chunk(text=" \n")])
    assert result.valid_citations == 1 and result.score == 0.0
    llm.generate.assert_not_called()


def test_none_pages_and_duplicate_citations():
    evaluator, _ = judge()
    result = evaluator.evaluate("Answer", [citation(page_number=None)] * 2, [chunk(page_number=None)] * 2)
    assert result.valid_citations == result.total_citations == 2
    assert result.score == 1.0


def test_conflicting_evidence_rejected():
    evaluator, llm = judge()
    with pytest.raises(ValueError, match="Conflicting retrieved evidence"):
        evaluator.evaluate("Answer", [citation()], [chunk(), chunk(text="Different")])
    llm.generate.assert_not_called()


@pytest.mark.parametrize("patch", [
    {"score": 0.5}, {"score": 0.2}, {"score": True}, {"score": "1"},
    {"score": float("nan")}, {"score": float("inf")}, {"label": "grounded"},
    {"reason": ""}, {"reason": " \n"}, {"reason": 5}, {"extra": 1},
])
def test_invalid_judge_fields(patch):
    evaluator, llm = judge()
    llm.generate.return_value = json.dumps({"score": 1, "label": "accurate", "reason": "Supported", **patch})
    with pytest.raises(CitationAccuracyParsingError):
        evaluator.evaluate("Answer", [citation()], [chunk()])


@pytest.mark.parametrize("response", [
    "bad JSON", "", "{}", "[]", "null", '{"score":1,',
    '{"score":1,"label":"accurate"}',
    '{"score":0,"score":1,"label":"accurate","reason":"Supported"}',
    '```json\n{"score":1,"label":"accurate","reason":"Supported"}\n```',
])
def test_malformed_json(response):
    evaluator, llm = judge()
    llm.generate.return_value = response
    with pytest.raises(CitationAccuracyParsingError) as caught:
        evaluator.evaluate("Answer", [citation()], [chunk()])
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("answer", ["", " \n", None, 42])
def test_invalid_answer(answer):
    evaluator, llm = judge()
    with pytest.raises(ValueError, match="generated_answer"):
        evaluator.evaluate(answer, [], [])
    llm.generate.assert_not_called()


def test_aggregate():
    results = [judge(score, label)[0].evaluate("Answer", [citation()], [chunk()]) for score, label in [
        (1, "accurate"), (1, "accurate"), (0.5, "partially_accurate"), (0, "inaccurate")
    ]]
    results.append(judge()[0].evaluate(INSUFFICIENT_INFORMATION, [], []))
    summary = summarize_citation_accuracy(iter(results))
    assert summary.mean_citation_accuracy == 0.625
    assert (summary.accurate_count, summary.partially_accurate_count, summary.inaccurate_count) == (2, 1, 1)
    assert (summary.evaluated_count, summary.skipped_count) == (4, 1)


def test_empty_and_all_skipped_aggregate():
    refusal = judge()[0].evaluate(INSUFFICIENT_INFORMATION, [], [])
    for results in [[], [refusal]]:
        summary = summarize_citation_accuracy(results)
        assert summary.mean_citation_accuracy is None
        assert summary.evaluated_count == 0
        assert summary.skipped_count == len(results)


def test_provider_errors_propagate():
    evaluator, llm = judge()
    error = LLMError("Provider failed")
    llm.generate.side_effect = error
    with pytest.raises(LLMError) as caught:
        evaluator.evaluate("Answer", [citation()], [chunk()])
    assert caught.value is error
