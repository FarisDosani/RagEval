import json
from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.experiments.comparison_runner import ComparisonResult, ComparisonRunner
from app.experiments.experiment_runner import ExperimentRunner
from app.experiments.models import ExperimentConfig, ExperimentResult, ExperimentSummary
from app.schemas.benchmark import BenchmarkItem


def config(strategy="dense", **changes):
    return ExperimentConfig(**{
        "experiment_name": strategy, "retrieval_strategy": strategy,
        "model": "synthetic-model", "benchmark_name": "synthetic-benchmark", **changes,
    })


def benchmark():
    return [BenchmarkItem(question_id="q", question="Synthetic?", ground_truth="Yes", answerable=True,
                          category="factual", difficulty="easy")]


def summary():
    # Synthetic values only; no evaluation/provider work is performed.
    return ExperimentSummary.model_validate({
        "evaluated_question_count": 1,
        "retrieval": {"evaluated_questions": 1, "skipped_questions": 0, "recall_at_k": {1: 1.0, 3: None}, "mrr": 1.0},
        "correctness": {"mean_score": 0.5, "correct_count": 0, "partially_correct_count": 1,
                        "incorrect_count": 0, "evaluated_count": 1, "skipped_count": 0},
        "groundedness": {"mean_groundedness": 1.0, "grounded_count": 1, "partially_grounded_count": 0,
                         "ungrounded_count": 0, "evaluated_count": 1},
        "citation_accuracy": {"mean_citation_accuracy": None, "accurate_count": 0, "partially_accurate_count": 0,
                              "inaccurate_count": 0, "evaluated_count": 0, "skipped_count": 1},
        "hallucination_refusal": {"total_evaluated": 1, "substantive_answerable_count": 1, "answerable_count": 1,
                                  "unanswerable_count": 0, "hallucination_rate": 0.0, "correct_refusal_rate": None,
                                  "failed_refusal_rate": None, "unnecessary_refusal_rate": 0.0,
                                  "label_counts": {"supported_answer": 1}},
        "average_latency_ms": 12.5, "total_prompt_tokens": None, "total_completion_tokens": None,
        "total_tokens": None, "total_cost": None,
    })


@pytest.fixture
def runner():
    experiment_runner = Mock(spec=ExperimentRunner)
    def run(items, settings, **kwargs):
        return ExperimentResult(
            experiment_id=uuid4(), timestamp=datetime.now(timezone.utc), config=settings,
            config_sha256="synthetic-config-hash", benchmark_sha256="synthetic-benchmark-hash",
            benchmark=items, questions=[], summary=summary(),
        )
    experiment_runner.run.side_effect = run
    return ComparisonRunner(experiment_runner)


@pytest.mark.parametrize("strategies", [("dense", "bm25"), ("dense", "bm25", "hybrid")])
def test_comparison_reuses_runner_and_saves(strategies, runner, tmp_path):
    configs = [config(strategy) for strategy in strategies]
    items = benchmark()
    result = runner.run(iter(items), iter(configs), output_dir=tmp_path / "comparisons", experiment_output_dir=tmp_path / "experiments")
    assert runner.experiment_runner.run.call_count == len(configs)
    for call, settings in zip(runner.experiment_runner.run.call_args_list, configs):
        assert call.args == (items, settings)
        assert call.kwargs == {"output_dir": tmp_path / "experiments"}
    assert result.configs == configs
    assert result.experiment_ids == [row.experiment_id for row in result.metrics]
    assert len(set(result.experiment_ids)) == len(configs)
    assert [row["strategy"] for row in result.table_rows()] == list(strategies)
    path = tmp_path / "comparisons" / f"{result.comparison_id}.json"
    assert ComparisonResult.model_validate_json(path.read_text()) == result
    saved = json.loads(path.read_text())
    assert saved["experiment_ids"] == [str(identifier) for identifier in result.experiment_ids]
    assert saved["benchmark_name"] == "synthetic-benchmark"
    assert saved["model"] == "synthetic-model"
    assert "winner" not in saved and "ranking" not in saved


@pytest.mark.parametrize("field", ["benchmark_name", "model"])
def test_fairness_mismatches_rejected_before_runs(field, runner, tmp_path):
    with pytest.raises(ValueError, match=field):
        runner.run(benchmark(), [config(), config("bm25", **{field: "different"})], output_dir=tmp_path)
    runner.experiment_runner.run.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("key", ["corpus_id", "index_id"])
@pytest.mark.parametrize("other", [None, "different", ""])
def test_declared_identity_validation(key, other, runner, tmp_path):
    metadata = {} if other is None else {key: other}
    with pytest.raises(ValueError, match=key):
        runner.run(benchmark(), [config(component_metadata={key: "shared"}), config("bm25", component_metadata=metadata)], output_dir=tmp_path)
    runner.experiment_runner.run.assert_not_called()


def test_comparison_variables_can_differ(runner, tmp_path):
    configs = [config(component_metadata={"corpus_id": "shared"}), config(
        "hybrid", top_k=3, reranker_enabled=True, query_rewrite_enabled=True,
        component_metadata={"corpus_id": "shared"},
    )]
    result = runner.run(benchmark(), configs, output_dir=tmp_path)
    assert result.table_rows()[1]["top_k"] == 3
    assert result.table_rows()[1]["reranker"] is True
    assert result.table_rows()[1]["query_rewrite"] is True


def test_missing_metrics_remain_none_and_cutoffs_align(runner, tmp_path):
    original = runner.experiment_runner.run.side_effect
    def run(items, settings, **kwargs):
        result = original(items, settings, **kwargs)
        if settings.retrieval_strategy == "bm25":
            result.summary.retrieval.recall_at_k.clear()
            result.summary.retrieval.recall_at_k[5] = 0.5
        return result
    runner.experiment_runner.run.side_effect = run
    result = runner.run(benchmark(), [config(), config("bm25")], output_dir=tmp_path)
    assert result.metrics[0].recall_at_k == {1: 1.0, 3: None, 5: None}
    assert result.metrics[1].recall_at_k == {1: None, 3: None, 5: 0.5}
    row = result.metrics[0]
    assert row.total_tokens is None and row.total_cost is None
    assert row.correct_refusal_rate is None and row.mean_citation_accuracy is None
    assert row.hallucination_rate == 0.0  # Preserve a measured zero distinctly.
    assert row.mrr == 1 and row.mean_correctness == 0.5
    assert row.mean_groundedness == 1 and row.average_latency_ms == 12.5


def test_no_overwrite(runner, tmp_path):
    identifier = uuid4()
    path = tmp_path / f"{identifier}.json"
    path.write_text("existing", encoding="utf-8")
    with pytest.raises(FileExistsError):
        runner.run(benchmark(), [config(), config("bm25")], output_dir=tmp_path, comparison_id=identifier)
    assert path.read_text() == "existing"
    runner.experiment_runner.run.assert_not_called()


def test_runner_failure_propagates(runner, tmp_path):
    error = RuntimeError("experiment failed")
    runner.experiment_runner.run.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        runner.run(benchmark(), [config(), config("bm25")], output_dir=tmp_path)
    assert caught.value is error
    assert not list(tmp_path.glob("*.json"))


@pytest.mark.parametrize("configs", [[], [config()]])
def test_requires_multiple_configs(configs, runner, tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        runner.run(benchmark(), configs, output_dir=tmp_path)
    runner.experiment_runner.run.assert_not_called()


def test_benchmark_is_isolated_per_run(runner, tmp_path):
    original = runner.experiment_runner.run.side_effect
    seen = []
    def run(items, settings, **kwargs):
        seen.append(items[0].question)
        items[0].question = "Modified by component"
        return original(items, settings, **kwargs)
    runner.experiment_runner.run.side_effect = run
    items = benchmark()
    runner.run(items, [config(), config("bm25")], output_dir=tmp_path)
    assert seen == ["Synthetic?", "Synthetic?"]
    assert items[0].question == "Synthetic?"
