from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from pydantic import BaseModel

from app.experiments.experiment_runner import DEFAULT_RESULTS_DIR, ExperimentRunner
from app.experiments.models import ExperimentConfig, ExperimentSummary
from app.schemas.benchmark import BenchmarkItem

DEFAULT_COMPARISONS_DIR = Path(__file__).resolve().parents[2] / "results" / "comparisons"


class ComparisonMetrics(BaseModel):
    experiment_id: UUID
    recall_at_k: dict[int, float | None]
    mrr: float | None
    mean_correctness: float | None
    mean_groundedness: float | None
    mean_citation_accuracy: float | None
    hallucination_rate: float | None
    correct_refusal_rate: float | None
    average_latency_ms: float | None
    total_tokens: int | None
    total_cost: float | None
    summary: ExperimentSummary  # Preserve denominators, skipped counts and usage scope.


class ComparisonResult(BaseModel):
    comparison_id: UUID
    timestamp: datetime
    benchmark_name: str
    model: str
    experiment_ids: list[UUID]
    configs: list[ExperimentConfig]
    metrics: list[ComparisonMetrics]

    def table_rows(self) -> list[dict]:
        """Backend rows in input configuration order; no ranking or winner."""
        return [
            {
                "experiment_name": config.experiment_name,
                "strategy": config.retrieval_strategy, "top_k": config.top_k,
                "reranker": config.reranker_enabled, "query_rewrite": config.query_rewrite_enabled,
                **metric.model_dump(mode="json", exclude={"summary"}),
                "usage_scope": metric.summary.usage_scope,
            }
            for config, metric in zip(self.configs, self.metrics, strict=True)
        ]


class ComparisonRunner:
    """Run explicit configs over independent copies of the same benchmark.

    Same benchmark_name and model are required. Optional component_metadata keys
    corpus_id and index_id, when supplied anywhere, must be nonblank and identical
    everywhere. These are declared identities, not verification of index contents.
    Strategy, top-k and modifiers may differ. Metrics remain in config order.
    A failure propagates; previously saved individual experiments are retained.
    """

    def __init__(self, experiment_runner: ExperimentRunner):
        self.experiment_runner = experiment_runner

    def run(
        self, benchmark: Iterable[BenchmarkItem], configs: Iterable[ExperimentConfig], *,
        output_dir: str | Path = DEFAULT_COMPARISONS_DIR,
        experiment_output_dir: str | Path = DEFAULT_RESULTS_DIR,
        comparison_id: UUID | None = None,
    ) -> ComparisonResult:
        configs = [config.model_copy(deep=True) for config in configs]
        if len(configs) < 2:
            raise ValueError("Comparison requires at least two configurations")
        for field in ("benchmark_name", "model"):
            if len({getattr(config, field) for config in configs}) != 1:
                raise ValueError(f"Compared configurations must share the same {field}")
        for key in ("corpus_id", "index_id"):
            values = [config.component_metadata.get(key) for config in configs]
            if any(value is not None for value in values):
                if any(value is None or not value.strip() for value in values) or len(set(values)) != 1:
                    raise ValueError(f"Compared configurations must share the same declared {key}")
        items = [item.model_copy(deep=True) for item in benchmark]
        if len({item.question_id for item in items}) != len(items):
            raise ValueError("Duplicate benchmark question IDs")
        identifier = UUID(str(comparison_id)) if comparison_id is not None else uuid4()
        path = Path(output_dir) / f"{identifier}.json"
        if path.exists():
            raise FileExistsError(f"Comparison result already exists: {path}")
        timestamp = datetime.now(timezone.utc)
        experiments = [
            self.experiment_runner.run(
                [item.model_copy(deep=True) for item in items], config.model_copy(deep=True),
                output_dir=experiment_output_dir,
            )
            for config in configs
        ]
        # Align cutoffs across rows; a cutoff not measured by a run remains None.
        ks = sorted({k for experiment in experiments for k in experiment.summary.retrieval.recall_at_k})
        metrics = []
        for experiment in experiments:
            summary = experiment.summary
            metrics.append(ComparisonMetrics(
                experiment_id=experiment.experiment_id,
                recall_at_k={k: summary.retrieval.recall_at_k.get(k) for k in ks},
                mrr=summary.retrieval.mrr, mean_correctness=summary.correctness.mean_score,
                mean_groundedness=summary.groundedness.mean_groundedness,
                mean_citation_accuracy=summary.citation_accuracy.mean_citation_accuracy,
                hallucination_rate=summary.hallucination_refusal.hallucination_rate,
                correct_refusal_rate=summary.hallucination_refusal.correct_refusal_rate,
                average_latency_ms=summary.average_latency_ms,
                total_tokens=summary.total_tokens, total_cost=summary.total_cost,
                summary=summary,
            ))
        result = ComparisonResult(
            comparison_id=identifier, timestamp=timestamp, benchmark_name=configs[0].benchmark_name,
            model=configs[0].model, experiment_ids=[e.experiment_id for e in experiments],
            configs=configs, metrics=metrics,
        )
        serialized = result.model_dump_json(indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return result
