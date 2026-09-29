import json
from unittest.mock import Mock

import pytest

from app.evaluation.groundedness import (
    GROUNDEDNESS_PROMPT, GroundednessEvaluator, GroundednessParsingError,
    GroundednessResult, summarize_groundedness,
)
from app.generation import llm_service
from app.generation.llm_service import LLMError, LLMService
from app.generation.prompts import INSUFFICIENT_INFORMATION
from app.schemas.retrieval import RetrievalResult


@pytest.fixture(autouse=True)
def block_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("No real API clients in groundedness tests"))
    monkeypatch.setattr(llm_service, "OpenAI", constructor)
    return constructor


def context(text="The library opens at 09:00."):
    return [RetrievalResult(
        chunk_id="c1", document_id="d1", source="library.txt", page_number=None,
        chunk_index=0, text=text, score=0.8,
    )]


def judge(response):
    llm = Mock(spec=LLMService)
    llm.generate.return_value = response
    return GroundednessEvaluator(llm), llm


@pytest.mark.parametrize("score,label,answer,reason", [
    (1.0, "grounded", "The library opens at nine in the morning.", "Harmless paraphrase of 09:00."),
    (0.5, "partially_grounded", "It opens at 09:00 and closes at 17:00.", "Closing time is an unsupported extra claim."),
    (0.0, "ungrounded", "It opens at 12:00.", "Contradicts the retrieved opening time."),
])
def test_judge_results(score, label, answer, reason, block_client):
    evaluator, llm = judge(json.dumps({"score": score, "label": label, "reason": reason}))
    result = evaluator.evaluate("When does the library open?", answer, context())
    assert isinstance(result, GroundednessResult)
    assert (result.score, result.label, result.reason) == (score, label, reason)
    assert evaluator.llm is llm
    block_client.assert_not_called()


def test_prompt_context_and_service_reuse():
    evaluator, llm = judge('{"score":1,"label":"grounded","reason":"Supported"}')
    chunks = context()
    original = chunks[0].model_dump()
    for _ in range(2):
        evaluator.evaluate("Question?", "Answer", iter(chunks))
    assert llm.generate.call_count == 2
    system, user = llm.generate.call_args.args
    assert system == GROUNDEDNESS_PROMPT
    for phrase in ["Do not use outside knowledge", "model memory", "unsupported factual claims", "harmless paraphrases", "Return only one JSON object"]:
        assert phrase in system
    assert json.loads(user) == {
        "question": "Question?", "generated_answer": "Answer",
        "retrieved_context": [{"chunk_id": "c1", "source": "library.txt", "page_number": None, "text": chunks[0].text}],
    }
    assert "ground_truth" not in json.loads(user)
    assert chunks[0].model_dump() == original


@pytest.mark.parametrize("patch", [
    {"score": 0.5}, {"score": 0.25}, {"score": "1"}, {"score": True},
    {"score": float("nan")}, {"score": float("inf")}, {"label": "correct"},
    {"reason": ""}, {"reason": " \n"}, {"reason": 2}, {"extra": "field"},
])
def test_invalid_assessment(patch):
    evaluator, _ = judge(json.dumps({"score": 1, "label": "grounded", "reason": "Supported", **patch}))
    with pytest.raises(GroundednessParsingError):
        evaluator.evaluate("Q", "A", context())


@pytest.mark.parametrize("response", [
    "invalid JSON", "", "{}", "[]", "null", '{"score":1,',
    '{"score":1,"label":"grounded"}',
    '{"score":0,"score":1,"label":"grounded","reason":"Supported"}',
    '```json\n{"score":1,"label":"grounded","reason":"Supported"}\n```',
])
def test_malformed_judge_output(response):
    evaluator, _ = judge(response)
    with pytest.raises(GroundednessParsingError) as caught:
        evaluator.evaluate("Q", "A", context())
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("field", ["question", "generated_answer"])
@pytest.mark.parametrize("value", ["", " \t\n", None, 42])
def test_empty_or_invalid_inputs(field, value):
    evaluator, llm = judge("unused")
    inputs = {"question": "Q", "generated_answer": "A", field: value}
    with pytest.raises(ValueError, match=field):
        evaluator.evaluate(**inputs, retrieved_chunks=[])
    llm.generate.assert_not_called()


@pytest.mark.parametrize("chunks", [[], context(" \n\t")])
def test_no_context_refusal(chunks):
    evaluator, llm = judge("unused")
    result = evaluator.evaluate("Q", INSUFFICIENT_INFORMATION, chunks)
    assert result.score == 1.0
    assert result.label == "grounded"
    assert result.reason
    llm.generate.assert_not_called()


@pytest.mark.parametrize("answer", ["The library opens at 09:00.", INSUFFICIENT_INFORMATION + " Also, it closes at 17:00.", " " + INSUFFICIENT_INFORMATION])
def test_no_context_non_exact_refusal(answer):
    evaluator, llm = judge("unused")
    result = evaluator.evaluate("Q", answer, [])
    assert result.score == 0.0
    assert result.label == "ungrounded"
    llm.generate.assert_not_called()


def test_aggregate():
    results = [GroundednessResult(score=s, label=l, reason="Reason") for s, l in [
        (1, "grounded"), (1, "grounded"), (0.5, "partially_grounded"), (0, "ungrounded")
    ]]
    summary = summarize_groundedness(iter(results))
    assert summary.mean_groundedness == 0.625
    assert summary.grounded_count == 2
    assert summary.partially_grounded_count == 1
    assert summary.ungrounded_count == 1
    assert summary.evaluated_count == 4


def test_empty_aggregate():
    summary = summarize_groundedness([])
    assert summary.mean_groundedness is None
    assert summary.evaluated_count == 0


def test_provider_errors_propagate():
    evaluator, llm = judge("unused")
    error = LLMError("Provider failed")
    llm.generate.side_effect = error
    with pytest.raises(LLMError) as caught:
        evaluator.evaluate("Q", "A", context())
    assert caught.value is error
