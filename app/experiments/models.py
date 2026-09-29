from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas.benchmark import BenchmarkItem
from app.schemas.generation import RAGResponse
from app.evaluation.answer_correctness import CorrectnessResult, CorrectnessSummary
from app.evaluation.groundedness import GroundednessResult, GroundednessSummary
from app.evaluation.citation_accuracy import CitationAccuracyResult, CitationAccuracySummary
from app.evaluation.hallucination_refusal import HallucinationRefusalResult, HallucinationRefusalSummary
from app.evaluation.retrieval_metrics import RetrievalMetrics

Name = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
PositiveInt = Annotated[int, Field(strict=True, gt=0)]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    experiment_name: Name
    retrieval_strategy: Literal["dense", "bm25", "hybrid"]
    top_k: PositiveInt = 5
    candidate_k: PositiveInt = 20
    reranker_enabled: bool = Field(default=False, strict=True)
    query_rewrite_enabled: bool = Field(default=False, strict=True)
    model: Name
    benchmark_name: Name
    chunk_size: PositiveInt | None = None  # Metadata only; never rebuilds ingestion.
    recall_k_values: tuple[PositiveInt, ...] = (1, 3, 5)
    component_metadata: dict[str, str] = Field(default_factory=dict)


class QuestionExperimentResult(BaseModel):
    question_id: str
    original_question: str
    rewritten_query: str | None
    retrieved_chunk_ids: list[str]
    generation: RAGResponse  # Answer, citations, evidence, model and provider usage.
    recall_at_k: dict[int, float | None]
    reciprocal_rank: float | None
    correctness: CorrectnessResult | None
    groundedness: GroundednessResult
    citation_accuracy: CitationAccuracyResult
    hallucination_refusal: HallucinationRefusalResult
    latency_ms: float = Field(ge=0)  # Rewrite + retrieval + rerank + generation; excludes judges.


class ExperimentSummary(BaseModel):
    evaluated_question_count: int
    retrieval: RetrievalMetrics
    correctness: CorrectnessSummary
    groundedness: GroundednessSummary
    citation_accuracy: CitationAccuracySummary
    hallucination_refusal: HallucinationRefusalSummary
    average_latency_ms: float | None
    total_prompt_tokens: int | None
    total_completion_tokens: int | None
    total_tokens: int | None
    total_cost: float | None
    usage_scope: Literal["answer_generation_only"] = "answer_generation_only"


class ExperimentResult(BaseModel):
    experiment_id: UUID
    timestamp: datetime
    config: ExperimentConfig
    config_sha256: str
    benchmark_sha256: str
    benchmark: list[BenchmarkItem]
    questions: list[QuestionExperimentResult]
    summary: ExperimentSummary
