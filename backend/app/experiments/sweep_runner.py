from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import UUID, uuid4

from pydantic import BaseModel

from app.experiments.experiment_runner import DEFAULT_RESULTS_DIR, ExperimentRunner
from app.experiments.models import ExperimentConfig, ExperimentSummary
from app.schemas.benchmark import BenchmarkItem

SweepType = Literal["top_k", "chunk_size"]
DEFAULT_SWEEPS_DIR = Path(__file__).resolve().parents[2] / "results" / "sweeps"


def plan_sweep(base_config: ExperimentConfig, sweep_type: SweepType, values: Iterable[int]) -> list[ExperimentConfig]:
    """Validate all values before execution, preserving input order and base config."""
    if sweep_type not in ("top_k", "chunk_size"):
        raise ValueError("sweep_type must be top_k or chunk_size")
    values = list(values)
    if not values:
        raise ValueError("Sweep values must not be empty")
    if any(type(value) is not int or value <= 0 for value in values):
        raise ValueError("Sweep values must be positive integers")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate sweep values are not allowed")
    return [ExperimentConfig.model_validate({
        **base_config.model_dump(), sweep_type: value,
        "experiment_name": f"{base_config.experiment_name}_{sweep_type}_{value}",
    }) for value in values]


@dataclass
class RebuiltCorpus:
    """Builder-owned, executable components for one chunk size.

    The builder must rechunk original documents, regenerate embeddings where
    needed, rebuild all selected indexes (including both hybrid branches), and
    return a fresh selected retriever. It must remap relevant_chunk_ids to the new
    chunks while preserving every other benchmark field/order. index_id identifies
    this built artifact; the sweep cannot verify its underlying corpus contents.
    """
    runner: ExperimentRunner
    benchmark: list[BenchmarkItem]
    chunk_size: int
    index_id: str


CorpusBuilder = Callable[[ExperimentConfig, list[BenchmarkItem]], RebuiltCorpus]


class SweepEntry(BaseModel):
    parameter_value: int
    experiment_id: UUID
    config: ExperimentConfig
    benchmark_sha256: str
    summary: ExperimentSummary


class SweepResult(BaseModel):
    sweep_id: UUID
    sweep_type: SweepType
    parameter_values: list[int]
    base_config: ExperimentConfig
    benchmark_name: str
    model: str
    timestamp: datetime
    experiment_ids: list[UUID]
    experiments: list[SweepEntry]


class SweepRunner:
    """Delegate execution/saving of individual runs to ExperimentRunner.

    Top-k uses the existing corpus. Chunk-size execution requires a builder; it
    never falls back to the base runner. Prior experiment files remain on failure.
    Summary values, missing values, usage scope and denominators are preserved.
    """

    def __init__(self, experiment_runner: ExperimentRunner, corpus_builder: CorpusBuilder | None = None):
        self.experiment_runner = experiment_runner
        self.corpus_builder = corpus_builder

    def run(
        self, benchmark: Iterable[BenchmarkItem], base_config: ExperimentConfig,
        sweep_type: SweepType, values: Iterable[int], *,
        output_dir: str | Path = DEFAULT_SWEEPS_DIR,
        experiment_output_dir: str | Path = DEFAULT_RESULTS_DIR,
        sweep_id: UUID | None = None,
    ) -> SweepResult:
        base_config = base_config.model_copy(deep=True)
        configs = plan_sweep(base_config, sweep_type, values)
        if sweep_type == "chunk_size" and self.corpus_builder is None:
            raise ValueError("chunk_size sweep requires a corpus/index rebuild hook")
        items = [item.model_copy(deep=True) for item in benchmark]
        if len({item.question_id for item in items}) != len(items):
            raise ValueError("Duplicate benchmark question IDs")
        identifier = UUID(str(sweep_id)) if sweep_id is not None else uuid4()
        path = Path(output_dir) / f"{identifier}.json"
        if path.exists():
            raise FileExistsError(f"Sweep result already exists: {path}")
        timestamp = datetime.now(timezone.utc)
        entries = []
        used_index_ids = set()
        old_index_id = base_config.component_metadata.get("index_id")
        if old_index_id:
            used_index_ids.add(old_index_id)
        used_retrievers = [self.experiment_runner.retrievers.get(base_config.retrieval_strategy)]
        for config in configs:
            run_items = [item.model_copy(deep=True) for item in items]
            runner = self.experiment_runner
            if sweep_type == "chunk_size":
                rebuilt = self.corpus_builder(config.model_copy(deep=True), run_items)
                if not isinstance(rebuilt, RebuiltCorpus):
                    raise ValueError("Rebuild hook must return RebuiltCorpus")
                if type(rebuilt.chunk_size) is not int or rebuilt.chunk_size != config.chunk_size:
                    raise ValueError("Rebuilt corpus chunk_size does not match requested size")
                if not isinstance(rebuilt.index_id, str) or not rebuilt.index_id.strip() or rebuilt.index_id in used_index_ids:
                    raise ValueError("Rebuild requires a distinct non-empty index_id for each size")
                selected = rebuilt.runner.retrievers.get(config.retrieval_strategy)
                if selected is None or any(selected is previous for previous in used_retrievers):
                    raise ValueError("Rebuild must supply a fresh selected retriever; stale index reuse is forbidden")
                before = [item.model_dump(exclude={"relevant_chunk_ids"}) for item in items]
                after = [item.model_dump(exclude={"relevant_chunk_ids"}) for item in rebuilt.benchmark]
                if before != after:
                    raise ValueError("Rebuild may only remap benchmark relevant_chunk_ids")
                used_index_ids.add(rebuilt.index_id)
                used_retrievers.append(selected)
                runner = rebuilt.runner
                run_items = [item.model_copy(deep=True) for item in rebuilt.benchmark]
                config = config.model_copy(update={"component_metadata": {
                    **config.component_metadata, "index_id": rebuilt.index_id,
                }}, deep=True)
            experiment = runner.run(run_items, config, output_dir=experiment_output_dir)
            entries.append(SweepEntry(
                parameter_value=getattr(config, sweep_type), experiment_id=experiment.experiment_id,
                config=experiment.config, benchmark_sha256=experiment.benchmark_sha256, summary=experiment.summary,
            ))
        result = SweepResult(
            sweep_id=identifier, sweep_type=sweep_type,
            parameter_values=[getattr(config, sweep_type) for config in configs],
            base_config=base_config, benchmark_name=base_config.benchmark_name, model=base_config.model,
            timestamp=timestamp, experiment_ids=[entry.experiment_id for entry in entries], experiments=entries,
        )
        serialized = result.model_dump_json(indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return result
