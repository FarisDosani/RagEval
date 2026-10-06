import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import UUID, uuid4

from app.experiments.models import ExperimentConfig, ExperimentResult, ExperimentSummary, QuestionExperimentResult
from app.schemas.benchmark import BenchmarkItem
from app.generation.rag_pipeline import RAGPipeline
from app.generation.query_rewriter import QueryRewriter
from app.retrieval.reranker import CrossEncoderReranker
from app.evaluation.retrieval_metrics import evaluate_retrieval, recall_at_k, reciprocal_rank
from app.evaluation.answer_correctness import AnswerCorrectnessEvaluator, summarize_correctness
from app.evaluation.groundedness import GroundednessEvaluator, summarize_groundedness
from app.evaluation.citation_accuracy import CitationAccuracyEvaluator, summarize_citation_accuracy
from app.evaluation.hallucination_refusal import HallucinationRefusalEvaluator, summarize_hallucination_refusal

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "experiments"


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _complete_total(values):
    values = list(values)
    return None if any(value is None for value in values) else sum(values)


class ExperimentRunner:
    """Orchestrate prebuilt components; never load models, ingest data or create clients.

    retrievers maps strategy names to existing VectorStore/BM25Retriever/
    HybridRetriever instances. Configure custom dense embedding services on an
    injected adapter if needed. Record external model/index settings in
    config.component_metadata. Benchmark and final evidence snapshots plus hashes
    support auditing; LLM determinism is not guaranteed.

    Retrieval metrics use final evidence after reranking. Latency excludes judges.
    Token/cost totals cover answer generation only, not rewrite/judge calls, and
    become None if any generation lacks the corresponding metadata.
    """

    def __init__(
        self, *, retrievers: Mapping[str, object], generator: RAGPipeline,
        correctness: AnswerCorrectnessEvaluator, groundedness: GroundednessEvaluator,
        citation_accuracy: CitationAccuracyEvaluator, hallucination_refusal: HallucinationRefusalEvaluator,
        reranker: CrossEncoderReranker | None = None, query_rewriter: QueryRewriter | None = None,
    ):
        self.retrievers = dict(retrievers)
        self.generator = generator
        self.correctness = correctness
        self.groundedness = groundedness
        self.citation_accuracy = citation_accuracy
        self.hallucination_refusal = hallucination_refusal
        self.reranker = reranker
        self.query_rewriter = query_rewriter

    def run(
        self, benchmark: Iterable[BenchmarkItem], config: ExperimentConfig, *,
        output_dir: str | Path = DEFAULT_RESULTS_DIR, experiment_id: UUID | None = None,
        checkpoint_path: str | Path | None = None,
        excluded_wait_ms: Callable[[], float] | None = None,
    ) -> ExperimentResult:
        config = config.model_copy(deep=True)
        items = [item.model_copy(deep=True) for item in benchmark]
        if len({item.question_id for item in items}) != len(items):
            raise ValueError("Duplicate benchmark question IDs")
        if config.retrieval_strategy not in self.retrievers:
            raise ValueError("Selected retrieval strategy has no injected retriever")
        if config.reranker_enabled and self.reranker is None:
            raise ValueError("reranker_enabled requires an injected reranker")
        if config.query_rewrite_enabled and self.query_rewriter is None:
            raise ValueError("query_rewrite_enabled requires an injected query rewriter")
        if self.generator.llm.model != config.model:
            raise ValueError("Configured model must match the generation service model")
        config_hash = _digest(config.model_dump(mode="json"))
        benchmark_hash = _digest([item.model_dump(mode="json") for item in items])
        checkpoint = Path(checkpoint_path) if checkpoint_path is not None else None
        saved = None
        if checkpoint is not None and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved['config_sha256'] != config_hash or saved['benchmark_sha256'] != benchmark_hash:
                raise ValueError("Checkpoint config/benchmark mismatch")
            if experiment_id is not None and str(experiment_id) != saved['experiment_id']:
                raise ValueError("Checkpoint experiment ID mismatch")
        identifier = UUID(saved['experiment_id']) if saved else UUID(str(experiment_id)) if experiment_id is not None else uuid4()
        path = Path(output_dir) / f"{identifier}.json"
        if path.exists():
            if saved is not None:
                completed = ExperimentResult.model_validate_json(path.read_text(encoding='utf-8'))
                if completed.config_sha256 != config_hash or completed.benchmark_sha256 != benchmark_hash:
                    raise ValueError('Completed checkpoint result identity mismatch')
                return completed
            raise FileExistsError(f"Experiment result already exists: {path}")
        timestamp = datetime.fromisoformat(saved['timestamp']) if saved else datetime.now(timezone.utc)
        retriever = self.retrievers[config.retrieval_strategy]
        limit = max(config.candidate_k, config.top_k) if config.reranker_enabled else config.top_k
        searches = {
            "dense": lambda query: retriever.search_text(query, top_k=limit),
            "bm25": lambda query: retriever.search(query, top_k=limit),
            "hybrid": lambda query: retriever.search(query, candidate_k=max(config.candidate_k, limit), final_top_k=limit),
        }
        questions = [QuestionExperimentResult.model_validate(q) for q in saved['questions']] if saved else []
        if [q.question_id for q in questions] != [i.question_id for i in items[:len(questions)]] or len(questions) > len(items):
            raise ValueError("Checkpoint questions must be an ordered benchmark prefix")

        def persist_checkpoint():
            if checkpoint is None:
                return
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix('.tmp')
            temporary.write_text(json.dumps({
                'experiment_id': str(identifier), 'timestamp': timestamp.isoformat(),
                'config_sha256': config_hash, 'benchmark_sha256': benchmark_hash,
                'questions': [q.model_dump(mode='json') for q in questions],
            }, ensure_ascii=False, indent=2), encoding='utf-8')
            temporary.replace(checkpoint)

        persist_checkpoint()
        for item in items[len(questions):]:
            wait_before = excluded_wait_ms() if excluded_wait_ms is not None else 0.0
            started = perf_counter()
            rewritten = self.query_rewriter.rewrite_query(item.question) if config.query_rewrite_enabled else None
            chunks = searches[config.retrieval_strategy](rewritten if rewritten is not None else item.question)
            if config.reranker_enabled:
                chunks = self.reranker.rerank(rewritten if rewritten is not None else item.question, chunks, top_k=config.top_k)
            generation = self.generator.generate_from_chunks(item.question, chunks)
            latency = (perf_counter() - started) * 1000
            if excluded_wait_ms is not None:
                latency = max(generation.latency_ms, latency - (excluded_wait_ms() - wait_before))
            ids = [chunk.chunk_id for chunk in chunks]
            relevant = item.relevant_chunk_ids if item.answerable else []
            questions.append(QuestionExperimentResult(
                question_id=item.question_id, original_question=item.question, rewritten_query=rewritten,
                retrieved_chunk_ids=ids, generation=generation, latency_ms=latency,
                recall_at_k={k: recall_at_k(relevant, ids, k) for k in config.recall_k_values},
                reciprocal_rank=reciprocal_rank(relevant, ids),
                correctness=self.correctness.evaluate_benchmark_item(item, generation.answer),
                groundedness=self.groundedness.evaluate(item.question, generation.answer, chunks),
                citation_accuracy=self.citation_accuracy.evaluate(generation.answer, generation.citations, chunks),
                hallucination_refusal=self.hallucination_refusal.evaluate(item.answerable, generation.answer, chunks),
            ))
            persist_checkpoint()
        summary = ExperimentSummary(
            evaluated_question_count=len(questions),
            retrieval=evaluate_retrieval(items, {q.question_id: q.retrieved_chunk_ids for q in questions}, config.recall_k_values),
            correctness=summarize_correctness(q.correctness for q in questions),
            groundedness=summarize_groundedness(q.groundedness for q in questions),
            citation_accuracy=summarize_citation_accuracy(q.citation_accuracy for q in questions),
            hallucination_refusal=summarize_hallucination_refusal(q.hallucination_refusal for q in questions),
            average_latency_ms=sum(q.latency_ms for q in questions) / len(questions) if questions else None,
            total_prompt_tokens=_complete_total(q.generation.usage.prompt_tokens for q in questions),
            total_completion_tokens=_complete_total(q.generation.usage.completion_tokens for q in questions),
            total_tokens=_complete_total(q.generation.usage.total_tokens for q in questions),
            total_cost=_complete_total(q.generation.usage.cost for q in questions),
        )
        result = ExperimentResult(
            experiment_id=identifier, timestamp=timestamp, config=config,
            config_sha256=_digest(config.model_dump(mode="json")),
            benchmark_sha256=_digest([item.model_dump(mode="json") for item in items]),
            benchmark=items, questions=questions, summary=summary,
        )
        serialized = result.model_dump_json(indent=2)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation also protects against a concurrent run taking this ID.
        with path.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
        return result
