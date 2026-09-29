from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from app.evaluation.failure_analysis import ExperimentFailureAnalysis, analyze_experiment
from app.experiments.comparison_runner import ComparisonResult, ComparisonRunner
from app.experiments.models import ExperimentConfig, ExperimentResult, Name, PositiveInt
from app.experiments.sweep_runner import SweepResult, SweepRunner, SweepType
from app.schemas.benchmark import BenchmarkItem
from app.schemas.generation import RAGResponse

router = APIRouter()


class APIRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class QueryRequest(APIRequest):
    question: Name
    top_k: PositiveInt = 5


class ExperimentRequest(APIRequest):
    benchmark: list[BenchmarkItem]
    config: ExperimentConfig


class ComparisonRequest(APIRequest):
    benchmark: list[BenchmarkItem]
    configs: list[ExperimentConfig] = Field(min_length=2)


class SweepRequest(APIRequest):
    benchmark: list[BenchmarkItem]
    base_config: ExperimentConfig
    sweep_type: SweepType
    values: list[PositiveInt] = Field(min_length=1)


class AnalysisRequest(APIRequest):
    experiment_id: UUID


def service(request: Request, name: str):
    value = getattr(request.app.state, name, None)
    if value is None:
        raise HTTPException(503, "Research service is not configured; supply prebuilt components through create_app.")
    return value


def directory(request: Request, kind: str) -> Path:
    return request.app.state.results_dir / kind


def read_result(request: Request, kind: str, identifier: UUID, model):
    # UUID parsing prevents client paths; resolve also rejects symlink escapes.
    root = directory(request, kind).resolve()
    path = (root / f"{identifier}.json").resolve()
    if not path.is_relative_to(root):
        raise HTTPException(400, "Invalid result path")
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise HTTPException(404, "Saved result not found") from None
    try:
        result = model.model_validate_json(text)
        id_field = {"experiments": "experiment_id", "comparisons": "comparison_id", "sweeps": "sweep_id"}[kind]
        if getattr(result, id_field) != identifier:
            raise ValueError("Mismatched saved ID")
        return result
    except ValueError:
        raise HTTPException(500, "Saved result is invalid") from None


@router.post("/query", response_model=RAGResponse)
def query(body: QueryRequest, request: Request):
    return service(request, "pipeline").ask(body.question, top_k=body.top_k)


@router.post("/experiments/run", response_model=ExperimentResult)
def run_experiment(body: ExperimentRequest, request: Request):
    return service(request, "experiment_runner").run(
        body.benchmark, body.config, output_dir=directory(request, "experiments"))


@router.get("/experiments/{experiment_id}", response_model=ExperimentResult)
def get_experiment(experiment_id: UUID, request: Request):
    return read_result(request, "experiments", experiment_id, ExperimentResult)


@router.post("/comparisons/run", response_model=ComparisonResult)
def run_comparison(body: ComparisonRequest, request: Request):
    return ComparisonRunner(service(request, "experiment_runner")).run(
        body.benchmark, body.configs, output_dir=directory(request, "comparisons"),
        experiment_output_dir=directory(request, "experiments"))


@router.get("/comparisons/{comparison_id}", response_model=ComparisonResult)
def get_comparison(comparison_id: UUID, request: Request):
    return read_result(request, "comparisons", comparison_id, ComparisonResult)


@router.post("/sweeps/run", response_model=SweepResult)
def run_sweep(body: SweepRequest, request: Request):
    if body.sweep_type == "chunk_size":
        raise HTTPException(501, "chunk_size execution requires a trusted corpus/index rebuild hook and is not supported over HTTP.")
    return SweepRunner(service(request, "experiment_runner")).run(
        body.benchmark, body.base_config, body.sweep_type, body.values,
        output_dir=directory(request, "sweeps"), experiment_output_dir=directory(request, "experiments"))


@router.get("/sweeps/{sweep_id}", response_model=SweepResult)
def get_sweep(sweep_id: UUID, request: Request):
    return read_result(request, "sweeps", sweep_id, SweepResult)


@router.post("/failure-analysis", response_model=ExperimentFailureAnalysis)
def failure_analysis(body: AnalysisRequest, request: Request):
    return analyze_experiment(read_result(request, "experiments", body.experiment_id, ExperimentResult))
