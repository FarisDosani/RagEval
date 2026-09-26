from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from itertools import islice

from app.schemas.benchmark import BenchmarkItem


def _validate_k(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be an integer greater than 0")


def recall_at_k(
    relevant_chunk_ids: Iterable[str], retrieved_chunk_ids: Iterable[str], k: int
) -> float | None:
    """Benchmark hit Recall@K: 1 for any top-k hit, otherwise 0.

    This is not fractional recall over all relevant chunks. Empty relevance
    labels return None (not applicable), never a retrieval failure.
    """
    _validate_k(k)
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return None
    return float(any(chunk_id in relevant for chunk_id in islice(retrieved_chunk_ids, k)))


def reciprocal_rank(
    relevant_chunk_ids: Iterable[str], retrieved_chunk_ids: Iterable[str]
) -> float | None:
    """1 / first relevant rank, 0 for no hit, None for no relevance labels.

    Ranks use the supplied order, including duplicate retrieval positions.
    No cutoff is imposed; pass a truncated ranking when a cutoff is desired.
    """
    relevant = set(relevant_chunk_ids)
    if not relevant:
        return None
    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def mean_reciprocal_rank(
    question_results: Iterable[tuple[Iterable[str], Iterable[str]]],
) -> float | None:
    """Mean RR for (relevant IDs, ranked retrieved IDs) pairs.

    Skip pairs without relevance labels; return None if none can be evaluated.
    Use evaluate_retrieval for benchmark answerable flags and transparent counts.
    """
    ranks = [reciprocal_rank(relevant, retrieved) for relevant, retrieved in question_results]
    applicable = [rank for rank in ranks if rank is not None]
    return sum(applicable) / len(applicable) if applicable else None


@dataclass(frozen=True)
class RetrievalMetrics:
    evaluated_questions: int
    skipped_questions: int
    recall_at_k: dict[int, float | None]
    mrr: float | None


def evaluate_retrieval(
    benchmark: Iterable[BenchmarkItem],
    retrieved_by_question: Mapping[str, Iterable[str]],
    k_values: Iterable[int] = (1, 3, 5),
) -> RetrievalMetrics:
    """Aggregate hit Recall@K and full-ranking MRR using question ID mapping.

    Unanswerable items (answerable=False) and empty relevance labels are skipped
    even if retrieval results exist. All-skipped/empty input yields None metrics.
    Eligible questions require an explicit mapping entry: [] means retrieval ran
    and found nothing, whereas a missing entry is an error. Extra mapping keys
    are ignored. Duplicate benchmark question IDs are rejected.
    """
    ks = list(k_values)
    for k in ks:
        _validate_k(k)
    ks = list(dict.fromkeys(ks))
    pairs: list[tuple[list[str], list[str]]] = []
    skipped = 0
    seen: set[str] = set()
    for item in benchmark:
        if item.question_id in seen:
            raise ValueError(f"Duplicate benchmark question_id: {item.question_id!r}")
        seen.add(item.question_id)
        if not item.answerable or not item.relevant_chunk_ids:
            skipped += 1
            continue
        if item.question_id not in retrieved_by_question:
            raise ValueError(f"Missing retrieval results for question_id: {item.question_id!r}")
        pairs.append((item.relevant_chunk_ids, list(retrieved_by_question[item.question_id])))

    recalls: dict[int, float | None] = {}
    for k in ks:
        values = [recall_at_k(relevant, retrieved, k) for relevant, retrieved in pairs]
        applicable = [value for value in values if value is not None]
        recalls[k] = sum(applicable) / len(applicable) if applicable else None
    return RetrievalMetrics(
        evaluated_questions=len(pairs), skipped_questions=skipped,
        recall_at_k=recalls, mrr=mean_reciprocal_rank(pairs),
    )
