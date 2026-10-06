import json
from unittest.mock import Mock

import pytest

from app.evaluation.answer_correctness import (
    AnswerCorrectnessEvaluator, CORRECTNESS_PROMPT, CorrectnessParsingError,
    CorrectnessResult, summarize_correctness,
)
from app.generation import llm_service
from app.generation.llm_service import LLMError, LLMService
from app.schemas.benchmark import BenchmarkItem


@pytest.fixture(autouse=True)
def block_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("No API clients in judge tests"))
    monkeypatch.setattr(llm_service, "OpenAI", constructor)
    return constructor


def evaluator(response):
    llm = Mock(spec=LLMService)
    llm.generate.return_value = response
    return AnswerCorrectnessEvaluator(llm), llm


@pytest.mark.parametrize("score,label", [(1.0, "correct"), (0.5, "partially_correct"), (0.0, "incorrect")])
def test_valid_assessments_and_reuse(score, label, block_client):
    judge, llm = evaluator(json.dumps({"score": score, "label": label, "reason": "Reference comparison."}))
    result = judge.evaluate("Question?", "Reference", "Answer")
    assert isinstance(result, CorrectnessResult)
    assert (result.score, result.label, result.reason) == (score, label, "Reference comparison.")
    assert judge.llm is llm
    judge.evaluate("Second?", "Reference 2", "Answer 2")
    assert llm.generate.call_count == 2
    block_client.assert_not_called()


def test_rubric_and_input_passed_unchanged():
    judge, llm = evaluator('{"score":1,"label":"correct","reason":"Matches."}')
    question, truth, answer = "Question café?", "Two facts.\nQuoted: \"yes\"", "User answer"
    judge.evaluate(question, truth, answer)
    system, user = llm.generate.call_args.args
    assert system == CORRECTNESS_PROMPT
    for requirement in ["only against the supplied ground truth", "Do not use outside knowledge",
                        "harmless wording differences", "missing required facts", "contradictory/incorrect",
                        "Return only one JSON object"]:
        assert requirement in system
    assert json.loads(user) == {"question": question, "ground_truth": truth, "generated_answer": answer}


@pytest.mark.parametrize("patch", [
    {"score": 0.5}, {"score": 0.2}, {"score": "1.0"}, {"score": True},
    {"score": float("nan")}, {"score": float("inf")}, {"label": "unknown"},
    {"reason": ""}, {"reason": " \n"}, {"reason": 12}, {"extra": "field"},
])
def test_invalid_assessment(patch):
    judge, _ = evaluator(json.dumps({"score": 1.0, "label": "correct", "reason": "Matches", **patch}))
    with pytest.raises(CorrectnessParsingError):
        judge.evaluate("Question", "Reference", "Answer")


@pytest.mark.parametrize("response", [
    "not JSON", "", "{}", "[]", "null", '{"score":1,',
    '```json\n{"score":1,"label":"correct","reason":"Matches"}\n```',
    '{"score":0,"score":1,"label":"correct","reason":"Matches"}',
    '{"score":1,"label":"correct"}',
])
def test_malformed_output_raises(response):
    judge, _ = evaluator(response)
    with pytest.raises(CorrectnessParsingError) as caught:
        judge.evaluate("Question", "Reference", "Answer")
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("name", ["question", "ground_truth", "generated_answer"])
@pytest.mark.parametrize("value", ["", " \t\n", None])
def test_invalid_inputs(name, value):
    judge, llm = evaluator("unused")
    inputs = {"question": "Q", "ground_truth": "G", "generated_answer": "A", name: value}
    with pytest.raises(ValueError, match=name):
        judge.evaluate(**inputs)
    llm.generate.assert_not_called()


def benchmark(answerable):
    return BenchmarkItem(
        question_id="q1", question="Question?", ground_truth="Reference" if answerable else None,
        answerable=answerable, category="factual" if answerable else "unanswerable", difficulty="easy",
    )


def test_unanswerable_skipped_without_judge():
    judge, llm = evaluator("unused")
    assert judge.evaluate_benchmark_item(benchmark(False), "") is None
    llm.generate.assert_not_called()


def test_answerable_benchmark_evaluated():
    judge, llm = evaluator('{"score":0.5,"label":"partially_correct","reason":"Missing detail"}')
    assert judge.evaluate_benchmark_item(benchmark(True), "Answer").score == 0.5
    assert json.loads(llm.generate.call_args.args[1])["ground_truth"] == "Reference"


def test_aggregate_mean_and_counts():
    results = [CorrectnessResult(score=s, label=l, reason="Reason") for s, l in [
        (1, "correct"), (1, "correct"), (0.5, "partially_correct"), (0, "incorrect")
    ]]
    summary = summarize_correctness(iter([*results, None, None]))
    assert summary.mean_score == 0.625
    assert summary.correct_count == 2
    assert summary.partially_correct_count == 1
    assert summary.incorrect_count == 1
    assert summary.evaluated_count == 4
    assert summary.skipped_count == 2


@pytest.mark.parametrize("results", [[], [None, None]])
def test_no_applicable_results(results):
    summary = summarize_correctness(results)
    assert summary.mean_score is None
    assert summary.evaluated_count == 0
    assert summary.skipped_count == len(results)
    assert summary.incorrect_count == 0


def test_provider_errors_propagate():
    judge, llm = evaluator("unused")
    error = LLMError("Provider failed")
    llm.generate.side_effect = error
    with pytest.raises(LLMError) as caught:
        judge.evaluate("Question", "Reference", "Answer")
    assert caught.value is error
