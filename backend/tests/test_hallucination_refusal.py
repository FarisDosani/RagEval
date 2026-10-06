import json
from unittest.mock import Mock

import pytest

from app.evaluation.hallucination_refusal import (
    HALLUCINATION_PROMPT, HallucinationParsingError, HallucinationRefusalEvaluator,
    summarize_hallucination_refusal,
)
from app.generation import llm_service
from app.generation.llm_service import LLMError, LLMService
from app.generation.prompts import INSUFFICIENT_INFORMATION
from app.schemas.retrieval import RetrievalResult


@pytest.fixture(autouse=True)
def block_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("No real API clients in these tests"))
    monkeypatch.setattr(llm_service, "OpenAI", constructor)
    return constructor


def evidence(text="The library opens at 09:00."):
    return [RetrievalResult(chunk_id="c1", document_id="d1", source="library.txt",
                            page_number=None, chunk_index=0, text=text, score=0.9)]


def judge(hallucinated=False, claims=None):
    llm = Mock(spec=LLMService)
    llm.generate.return_value = json.dumps({
        "hallucinated": hallucinated, "reason": "Evidence comparison.",
        "unsupported_claims": claims or [],
    })
    return HallucinationRefusalEvaluator(llm), llm


@pytest.mark.parametrize("hallucinated,claims,label", [
    (False, [], "supported_answer"),
    (True, ["The library closes at 17:00."], "hallucinated_answer"),
])
def test_semantic_judgment(hallucinated, claims, label, block_client):
    evaluator, llm = judge(hallucinated, claims)
    result = evaluator.evaluate(True, "Library answer", evidence())
    assert result.label == label
    assert result.hallucinated is hallucinated
    assert result.unsupported_claims == claims
    assert not result.refused
    assert evaluator.llm is llm
    block_client.assert_not_called()


def test_prompt_payload_and_reuse():
    evaluator, llm = judge()
    for _ in range(2):
        evaluator.evaluate(True, "It opens at nine.", iter(evidence()))
    assert llm.generate.call_count == 2
    system, user = llm.generate.call_args.args
    assert system == HALLUCINATION_PROMPT
    assert "Do not use outside knowledge" in system
    assert "unsupported" in system and "only one JSON object" in system
    assert json.loads(user) == {"generated_answer": "It opens at nine.", "retrieved_evidence": [
        {"chunk_id": "c1", "source": "library.txt", "page_number": None, "text": evidence()[0].text}
    ]}


@pytest.mark.parametrize("answerable,label", [(False, "correct_refusal"), (True, "incorrect_refusal")])
@pytest.mark.parametrize("answer", [INSUFFICIENT_INFORMATION, " \n" + INSUFFICIENT_INFORMATION + "\t "])
def test_exact_refusals(answerable, label, answer):
    evaluator, llm = judge()
    result = evaluator.evaluate(answerable, answer, [])
    assert result.label == label
    assert result.refused and result.hallucinated is False
    assert result.unsupported_claims == []
    llm.generate.assert_not_called()


def test_failed_refusal_with_evidence_is_not_semantically_judged():
    evaluator, llm = judge()
    result = evaluator.evaluate(False, "It opens at 09:00.", evidence())
    assert result.label == "failed_refusal"
    assert not result.refused and result.hallucinated is None
    assert result.unsupported_claims == []
    llm.generate.assert_not_called()


@pytest.mark.parametrize("chunks", [[], evidence(" \n")])
@pytest.mark.parametrize("answerable,label", [(True, "hallucinated_answer"), (False, "failed_refusal")])
def test_no_evidence_substantive_answer(chunks, answerable, label):
    evaluator, llm = judge()
    result = evaluator.evaluate(answerable, "It opens at 09:00.", chunks)
    assert result.label == label
    assert result.hallucinated is True
    assert result.unsupported_claims == ["It opens at 09:00."]
    llm.generate.assert_not_called()


def test_refusal_with_extra_claim_is_not_exact():
    evaluator, llm = judge()
    result = evaluator.evaluate(False, INSUFFICIENT_INFORMATION + " It opens at 09:00.", evidence())
    assert result.label == "failed_refusal"
    assert not result.refused
    llm.generate.assert_not_called()


@pytest.mark.parametrize("answer", ["", " \n\t", None, 42])
def test_empty_or_invalid_answer(answer):
    evaluator, llm = judge()
    with pytest.raises(ValueError, match="generated_answer"):
        evaluator.evaluate(True, answer, [])
    llm.generate.assert_not_called()


@pytest.mark.parametrize("answerable", ["false", 0, 1, None])
def test_invalid_answerable(answerable):
    evaluator, _ = judge()
    with pytest.raises(ValueError, match="answerable"):
        evaluator.evaluate(answerable, "Answer", [])


@pytest.mark.parametrize("response", [
    "bad JSON", "", "{}", "[]", "null", '{"hallucinated":',
    '{"hallucinated":false,"reason":"OK"}',
    '{"hallucinated":true,"hallucinated":false,"reason":"OK","unsupported_claims":[]}',
    '```json\n{"hallucinated":false,"reason":"OK","unsupported_claims":[]}\n```',
])
def test_malformed_json(response):
    evaluator, llm = judge()
    llm.generate.return_value = response
    with pytest.raises(HallucinationParsingError) as caught:
        evaluator.evaluate(True, "Answer", evidence())
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("patch", [
    {"hallucinated": "false"}, {"hallucinated": 0}, {"hallucinated": True},
    {"unsupported_claims": ["Unsupported"]}, {"unsupported_claims": [""]},
    {"unsupported_claims": "claim"}, {"unsupported_claims": [12]},
    {"reason": ""}, {"reason": " \n"}, {"extra": "field"},
])
def test_invalid_judge_fields(patch):
    evaluator, llm = judge()
    llm.generate.return_value = json.dumps({"hallucinated": False, "reason": "OK", "unsupported_claims": [], **patch})
    with pytest.raises(HallucinationParsingError):
        evaluator.evaluate(True, "Answer", evidence())


def test_provider_error_propagates():
    evaluator, llm = judge()
    error = LLMError("Provider failed")
    llm.generate.side_effect = error
    with pytest.raises(LLMError) as caught:
        evaluator.evaluate(True, "Answer", evidence())
    assert caught.value is error


def test_aggregate_denominators():
    evaluator, _ = judge()
    supported = evaluator.evaluate(True, "Answer", evidence())
    hallucinated = evaluator.evaluate(True, "Answer", [])
    correct = evaluator.evaluate(False, INSUFFICIENT_INFORMATION, [])
    incorrect = evaluator.evaluate(True, INSUFFICIENT_INFORMATION, [])
    failed = evaluator.evaluate(False, "Answer", [])
    summary = summarize_hallucination_refusal(iter([supported, supported, hallucinated, incorrect, correct, correct, failed]))
    assert summary.total_evaluated == 7
    assert summary.substantive_answerable_count == 3
    assert summary.answerable_count == 4
    assert summary.unanswerable_count == 3
    assert summary.hallucination_rate == pytest.approx(1 / 3)
    assert summary.correct_refusal_rate == pytest.approx(2 / 3)
    assert summary.failed_refusal_rate == pytest.approx(1 / 3)
    assert summary.unnecessary_refusal_rate == 0.25
    assert summary.label_counts == {"supported_answer": 2, "hallucinated_answer": 1,
                                    "incorrect_refusal": 1, "correct_refusal": 2, "failed_refusal": 1}


def test_empty_aggregate():
    summary = summarize_hallucination_refusal([])
    assert summary.total_evaluated == 0
    assert summary.hallucination_rate is None
    assert summary.correct_refusal_rate is None
    assert summary.failed_refusal_rate is None
    assert summary.unnecessary_refusal_rate is None


def test_only_refusals_has_no_hallucination_denominator():
    evaluator, _ = judge()
    summary = summarize_hallucination_refusal([
        evaluator.evaluate(True, INSUFFICIENT_INFORMATION, []),
    ])
    assert summary.hallucination_rate is None
    assert summary.correct_refusal_rate is None
    assert summary.failed_refusal_rate is None
    assert summary.unnecessary_refusal_rate == 1.0
