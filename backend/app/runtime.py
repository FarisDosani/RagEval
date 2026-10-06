"""Opt-in local service construction; no work or provider calls at import."""
import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

from app.embeddings.embedding_service import EmbeddingService
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_document
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.generation.llm_service import LLMService
from app.generation.rag_pipeline import RAGPipeline
from app.generation.query_rewriter import QueryRewriter
from app.experiments.experiment_runner import ExperimentRunner
from app.evaluation.answer_correctness import AnswerCorrectnessEvaluator
from app.evaluation.groundedness import GroundednessEvaluator
from app.evaluation.citation_accuracy import CitationAccuracyEvaluator
from app.evaluation.hallucination_refusal import HallucinationRefusalEvaluator

logger = logging.getLogger("uvicorn.error")
ROOT = Path(__file__).resolve().parents[1]


class DenseRetriever:
    """Bind the same embedding instance to experiment/hybrid searches."""
    def __init__(self, store, embeddings):
        self.store, self.embeddings = store, embeddings

    def search_text(self, query, top_k=5):
        return self.store.search_text(query, top_k=top_k, service=self.embeddings)


@dataclass
class RuntimeServices:
    pipeline: RAGPipeline
    experiment_runner: ExperimentRunner

    def close(self):
        self.pipeline.llm.close()


def auto_init_enabled():
    value = os.environ.get("RAGEVAL_AUTO_INIT", "false").strip().lower()
    if value not in {"true", "false", "1", "0"}:
        raise ValueError("RAGEVAL_AUTO_INIT must be true or false")
    return value in {"true", "1"}


def build_runtime():
    path = Path(os.environ.get("RAGEVAL_CORPUS_PATH", "data/corpus/Week # 02 Slides.pdf"))
    path = path if path.is_absolute() else ROOT / path
    if not path.is_file():
        raise ValueError("RAGEVAL_CORPUS_PATH must point to an existing document")
    # Match the preserved baseline's content-based identity and chunk settings.
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    units = load_document(path)
    logger.info("Research corpus loaded: %s (%d extracted units)", path.name, len(units))
    chunks = chunk_document([u.model_copy(update={"document_id": digest + "_s512_o50"}) for u in units], 512, 50)
    if not chunks:
        raise ValueError("Research corpus has no extractable text")
    logger.info("Research chunks ready: %d (512 words, 50 overlap)", len(chunks))
    llm = LLMService()
    try:
        embeddings = EmbeddingService()
        store = VectorStore(embeddings.embed_chunks(chunks))
        dense = DenseRetriever(store, embeddings)
        bm25 = BM25Retriever(chunks)
        hybrid = HybridRetriever(dense, bm25)
        logger.info("Research embeddings, FAISS, BM25 and hybrid indexes ready")
        pipeline = RAGPipeline(store, llm, embeddings)
        runner = ExperimentRunner(
            retrievers={"dense": dense, "bm25": bm25, "hybrid": hybrid}, generator=pipeline,
            correctness=AnswerCorrectnessEvaluator(llm), groundedness=GroundednessEvaluator(llm),
            citation_accuracy=CitationAccuracyEvaluator(llm), hallucination_refusal=HallucinationRefusalEvaluator(llm),
            reranker=CrossEncoderReranker(), query_rewriter=QueryRewriter(llm),
        )
        logger.info("Research services ready: provider=%s model=%s", llm.provider, llm.model)
        return RuntimeServices(pipeline, runner)
    except Exception:
        llm.close()
        raise
