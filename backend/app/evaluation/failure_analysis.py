"""Deterministic diagnostics over saved results; no component execution."""

from collections.abc import Iterable
from typing import Literal, get_args
from uuid import UUID

from pydantic import BaseModel

from app.experiments.models import ExperimentResult, QuestionExperimentResult
from app.schemas.benchmark import BenchmarkItem

FailureCategory = Literal[
    "retrieval_miss", "retrieval_low_rank", "generation_incorrect",
    "generation_ungrounded", "citation_failure", "hallucination",
    "incorrect_refusal", "failed_refusal",
]


class QuestionFailureAnalysis(BaseModel):
    question_id: str
    failure_categories: list[FailureCategory]
    notes: list[str]


class FailureSummary(BaseModel):
    """Category rates use total_questions_analyzed, not applicable-only counts.

    Categories overlap. Failure-free means no detected failures, not proof of
    success when metrics are missing. Rates are None for an empty population.
    """

    total_questions_analyzed: int
    questions_with_failures: int
    failure_free_questions: int
    category_counts: dict[FailureCategory, int]
    category_rates: dict[FailureCategory, float | None]


class ExperimentFailureAnalysis(BaseModel):
    experiment_id: UUID
    questions: list[QuestionFailureAnalysis]
    summary: FailureSummary


def analyze_question(
    result: QuestionExperimentResult, benchmark_item: BenchmarkItem | None,
    *, top_k: int,
) -> QuestionFailureAnalysis:
    """Use stored ordered IDs as authoritative rank evidence.

    A miss means no relevant ID anywhere in the recorded results. Low rank
    means the first match is beyond top_k; a zero at a smaller recall cutoff
    alone is insufficient. These two categories are mutually exclusive.
    Unanswerable items and absent relevance labels are not retrieval failures.
    """
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if benchmark_item is not None and benchmark_item.question_id != result.question_id:
        raise ValueError("Benchmark and result question IDs must match")
    categories: list[FailureCategory] = []
    notes: list[str] = []

    def add(category: FailureCategory, note: str) -> None:
        categories.append(category)
        notes.append(note)

    if benchmark_item is None:
        notes.append("Retrieval not assessed: benchmark item unavailable.")
    elif benchmark_item.answerable:
        relevant = set(benchmark_item.relevant_chunk_ids)
        if not relevant:
            notes.append("Retrieval not assessed: no relevant chunk IDs supplied.")
        else:
            rank = next((rank for rank, chunk_id in enumerate(result.retrieved_chunk_ids, 1)
                         if chunk_id in relevant), None)
            if rank is None:
                add("retrieval_miss", "No relevant chunk in recorded retrieval results.")
            elif rank > top_k:
                add("retrieval_low_rank", f"First relevant chunk at rank {rank}, beyond top_k={top_k}.")

    if result.correctness is not None and result.correctness.label in ("incorrect", "partially_correct"):
        add("generation_incorrect", f"Correctness: {result.correctness.label}.")
    if result.groundedness is not None and result.groundedness.score < 1.0:
        add("generation_ungrounded", f"Groundedness score: {result.groundedness.score}.")
    behavior = result.hallucination_refusal
    citation = result.citation_accuracy
    if (citation is not None and not citation.skipped and citation.score is not None
            and citation.score < 1.0 and not (behavior is not None and behavior.refused)):
        add("citation_failure", f"Citation accuracy score: {citation.score}.")
    if behavior is not None:
        if behavior.label == "hallucinated_answer":
            add("hallucination", "Recorded behavior: hallucinated_answer.")
        elif behavior.label in ("incorrect_refusal", "failed_refusal"):
            add(behavior.label, f"Recorded behavior: {behavior.label}.")
    return QuestionFailureAnalysis(question_id=result.question_id, failure_categories=categories, notes=notes)


def summarize_failures(results: Iterable[QuestionFailureAnalysis]) -> FailureSummary:
    items = list(results)
    counts = dict.fromkeys(get_args(FailureCategory), 0)
    for item in items:
        for category in set(item.failure_categories):
            counts[category] += 1
    failed = sum(bool(item.failure_categories) for item in items)
    return FailureSummary(
        total_questions_analyzed=len(items), questions_with_failures=failed,
        failure_free_questions=len(items) - failed, category_counts=counts,
        category_rates={category: count / len(items) if items else None for category, count in counts.items()},
    )


def analyze_experiment(experiment_result: ExperimentResult) -> ExperimentFailureAnalysis:
    """Preserve result order; missing benchmark items skip retrieval analysis."""
    benchmark = {item.question_id: item for item in experiment_result.benchmark}
    if len(benchmark) != len(experiment_result.benchmark):
        raise ValueError("Duplicate benchmark question IDs")
    if len({q.question_id for q in experiment_result.questions}) != len(experiment_result.questions):
        raise ValueError("Duplicate result question IDs")
    questions = [analyze_question(q, benchmark.get(q.question_id), top_k=experiment_result.config.top_k)
                 for q in experiment_result.questions]
    return ExperimentFailureAnalysis(
        experiment_id=experiment_result.experiment_id, questions=questions,
        summary=summarize_failures(questions),
    )
