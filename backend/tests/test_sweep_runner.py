from datetime import datetime, timezone
from unittest.mock import Mock
from uuid import uuid4

import pytest

from app.experiments.sweep_runner import RebuiltCorpus, SweepResult, SweepRunner, plan_sweep
from app.experiments.experiment_runner import ExperimentRunner
from app.experiments.models import ExperimentConfig, ExperimentResult, ExperimentSummary
from app.schemas.benchmark import BenchmarkItem


def config():
    return ExperimentConfig(experiment_name="demo", retrieval_strategy="dense", model="test", benchmark_name="synthetic")


def benchmark():
    return [BenchmarkItem(question_id="q", question="Question?", answerable=True, ground_truth="Answer",
                          relevant_chunk_ids=["old-chunk"], category="factual", difficulty="easy")]


def synthetic_summary():
    return ExperimentSummary.model_validate({
        "evaluated_question_count": 1,
        "retrieval": {"evaluated_questions": 1, "skipped_questions": 0, "recall_at_k": {1: 0.5}, "mrr": 0.5},
        "correctness": {"mean_score": 1, "correct_count": 1, "partially_correct_count": 0, "incorrect_count": 0, "evaluated_count": 1, "skipped_count": 0},
        "groundedness": {"mean_groundedness": 1, "grounded_count": 1, "partially_grounded_count": 0, "ungrounded_count": 0, "evaluated_count": 1},
        "citation_accuracy": {"mean_citation_accuracy": None, "accurate_count": 0, "partially_accurate_count": 0, "inaccurate_count": 0, "evaluated_count": 0, "skipped_count": 1},
        "hallucination_refusal": {"total_evaluated": 1, "substantive_answerable_count": 1, "answerable_count": 1, "unanswerable_count": 0, "hallucination_rate": 0, "correct_refusal_rate": None, "failed_refusal_rate": None, "unnecessary_refusal_rate": 0, "label_counts": {"supported_answer": 1}},
        "average_latency_ms": 10, "total_prompt_tokens": None, "total_completion_tokens": None, "total_tokens": None, "total_cost": None,
    })


def mocked_runner():
    runner = Mock(spec=ExperimentRunner)
    runner.retrievers = {"dense": object()}
    def run(items, settings, **kwargs):
        return ExperimentResult(experiment_id=uuid4(), timestamp=datetime.now(timezone.utc), config=settings,
                                config_sha256="synthetic", benchmark_sha256="synthetic", benchmark=items,
                                questions=[], summary=synthetic_summary())
    runner.run.side_effect = run
    return runner


def test_top_k_configs_execution_and_persistence(tmp_path):
    runner = mocked_runner()
    base = config()
    result = SweepRunner(runner).run(iter(benchmark()), base, "top_k", [5, 1, 3], output_dir=tmp_path, experiment_output_dir=tmp_path / "experiments")
    assert result.parameter_values == [5, 1, 3]
    assert [call.args[1].top_k for call in runner.run.call_args_list] == [5, 1, 3]
    assert runner.run.call_count == 3
    assert base.top_k == 5 and base.experiment_name == "demo"
    assert result.experiment_ids == [entry.experiment_id for entry in result.experiments]
    assert result.experiments[0].summary.retrieval.mrr == 0.5
    assert result.experiments[0].summary.total_cost is None
    assert result.experiments[0].summary.total_tokens is None
    assert SweepResult.model_validate_json((tmp_path / f"{result.sweep_id}.json").read_text()) == result


@pytest.mark.parametrize("kind", ["top_k", "chunk_size"])
@pytest.mark.parametrize("values", [[], [0], [-1], [True], [1.5], ["2"], [1, 1]])
def test_invalid_values(kind, values):
    with pytest.raises(ValueError):
        plan_sweep(config(), kind, values)


def test_chunk_size_plan_and_missing_builder(tmp_path):
    assert [c.chunk_size for c in plan_sweep(config(), "chunk_size", [256, 512])] == [256, 512]
    runner = mocked_runner()
    with pytest.raises(ValueError, match="rebuild hook"):
        SweepRunner(runner).run(benchmark(), config(), "chunk_size", [256], output_dir=tmp_path)
    runner.run.assert_not_called()
    assert not list(tmp_path.iterdir())


def test_chunk_size_rebuild_and_remap(tmp_path):
    base_runner = mocked_runner()
    built = []
    def build(settings, items):
        runner = mocked_runner()
        built.append(runner)
        items[0].relevant_chunk_ids = [f"chunk-{settings.chunk_size}"]
        return RebuiltCorpus(runner, items, settings.chunk_size, f"index-{settings.chunk_size}")
    builder = Mock(side_effect=build)
    items = benchmark()
    result = SweepRunner(base_runner, builder).run(items, config(), "chunk_size", [256, 512], output_dir=tmp_path)
    assert builder.call_count == 2
    base_runner.run.assert_not_called()
    for runner, size in zip(built, [256, 512]):
        call = runner.run.call_args
        assert call.args[1].chunk_size == size
        assert call.args[1].component_metadata["index_id"] == f"index-{size}"
        assert call.args[0][0].relevant_chunk_ids == [f"chunk-{size}"]
    assert items[0].relevant_chunk_ids == ["old-chunk"]
    assert result.parameter_values == [256, 512]
    assert (tmp_path / f"{result.sweep_id}.json").is_file()


@pytest.mark.parametrize("failure", ["size", "stale", "benchmark", "empty_id"])
def test_invalid_rebuild_rejected(failure, tmp_path):
    base_runner = mocked_runner()
    new_runner = mocked_runner()
    def build(settings, items):
        if failure == "benchmark":
            items[0].question = "Changed question"
        return RebuiltCorpus(
            base_runner if failure == "stale" else new_runner, items,
            999 if failure == "size" else settings.chunk_size, "" if failure == "empty_id" else "new-index",
        )
    with pytest.raises(ValueError):
        SweepRunner(base_runner, build).run(benchmark(), config(), "chunk_size", [256], output_dir=tmp_path)
    base_runner.run.assert_not_called()
    new_runner.run.assert_not_called()


def test_repeated_index_identity_rejected(tmp_path):
    runners = []
    def build(settings, items):
        runner = mocked_runner()
        runners.append(runner)
        return RebuiltCorpus(runner, items, settings.chunk_size, "same-index")
    with pytest.raises(ValueError, match="distinct"):
        SweepRunner(mocked_runner(), build).run(benchmark(), config(), "chunk_size", [256, 512], output_dir=tmp_path)
    runners[1].run.assert_not_called()


def test_no_overwrite_before_rebuild(tmp_path):
    identifier = uuid4()
    path = tmp_path / f"{identifier}.json"
    path.write_text("existing")
    runner, builder = mocked_runner(), Mock()
    with pytest.raises(FileExistsError):
        SweepRunner(runner, builder).run(benchmark(), config(), "chunk_size", [256], output_dir=tmp_path, sweep_id=identifier)
    assert path.read_text() == "existing"
    builder.assert_not_called()
    runner.run.assert_not_called()


def test_errors_propagate(tmp_path):
    runner = mocked_runner()
    error = RuntimeError("run failed")
    runner.run.side_effect = error
    with pytest.raises(RuntimeError) as caught:
        SweepRunner(runner).run(benchmark(), config(), "top_k", [1], output_dir=tmp_path)
    assert caught.value is error
    assert not list(tmp_path.glob("*.json"))
