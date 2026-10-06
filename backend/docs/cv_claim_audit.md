# CV claim audit

Scope: current source, tests, and the retained simplified Phase 25 evidence.
Backend source/test/artifact paths below are relative to `backend/`; frontend paths
are relative to the repository root.
No provider calls or experiments are needed for this audit. Real study evidence is in
[the final summary](../results/final/ca_small_gemini_final_summary.json).

Use precise CV wording: built and tested a RAG evaluation framework; demonstrated
dense/BM25/hybrid on a controlled 10-question subset of a real slide corpus.
Do not imply all implemented strategies were evaluated or that this small study
establishes general superiority, measured monetary savings, or a deployed service.

## Python

- Implemented: yes. Source: app/.
- Tested: tests/.
- Demonstrated in real run: Yes: all retained research execution.
- Infrastructure only: No.
- Limitation: Python 3.11 is the documented backend runtime.

## FastAPI

- Implemented: yes. Source: app/main.py; app/api/routes.py.
- Tested: tests/test_api.py; tests/test_health.py.
- Demonstrated in real run: Saved-artifact reads are verified in cleanup; real study execution used the CLI.
- Infrastructure only: Execution endpoints are infrastructure.
- Limitation: Default app requires injected services for execution (503); no authentication.

## FAISS

- Implemented: yes. Source: app/retrieval/vector_store.py.
- Tested: tests/test_retrieval.py.
- Demonstrated in real run: Yes: dense and hybrid study.
- Infrastructure only: No.
- Limitation: In-memory normalized inner-product search; no persistent vector database.

## sentence-transformers

- Implemented: yes. Source: app/embeddings/embedding_service.py; app/retrieval/reranker.py.
- Tested: tests/test_embeddings.py; tests/test_reranker.py.
- Demonstrated in real run: Embeddings yes; cross-encoder not used in final study.
- Infrastructure only: Reranker real comparison remains pending.
- Limitation: Model weights are lazy downloads; long text can exceed model input limits.

## rank-bm25

- Implemented: yes. Source: app/retrieval/bm25_retriever.py.
- Tested: tests/test_bm25.py.
- Demonstrated in real run: Yes: BM25 and hybrid study.
- Infrastructure only: No.
- Limitation: Lightweight deterministic tokenization; no language-specific morphology.

## pytest

- Implemented: yes. Source: tests/; pytest.ini.
- Tested: Full offline suite.
- Demonstrated in real run: Yes: validation around real execution, not a research metric.
- Infrastructure only: No.
- Limitation: Mocked providers test contracts, not production uptime.

## Docker

- Implemented: yes. Source: Dockerfile; .dockerignore.
- Tested: No Docker integration test; backend API has pytest coverage.
- Demonstrated in real run: Not built/run during this cleanup.
- Infrastructure only: Packaging definition in current audit.
- Limitation: Do not claim a newly verified container deployment; frontend/data/results are excluded from image.

## LLMs

- Implemented: yes. Source: app/generation/llm_service.py; app/generation/gemini_transport.py.
- Tested: tests/test_generation.py; tests/test_gemini_transport.py.
- Demonstrated in real run: Yes: native Gemini generation and judging.
- Infrastructure only: OpenAI-compatible real comparison not demonstrated here.
- Limitation: Retained study uses one Gemini model; no multi-model conclusion.

## Prompt engineering

- Implemented: yes. Source: app/generation/prompts.py; app/evaluation/.
- Tested: tests/test_judge_contract_prompts.py; tests/test_hallucination_refusal.py.
- Demonstrated in real run: Yes: contextual generation and strict JSON judge instructions.
- Infrastructure only: No.
- Limitation: Prompt use is demonstrated, but causal quality improvement was not measured by an ablation.

## RAG

- Implemented: yes. Source: app/generation/rag_pipeline.py.
- Tested: tests/test_generation.py.
- Demonstrated in real run: Yes: three 10-question real runs.
- Infrastructure only: No.
- Limitation: One slide corpus and one model; no large-scale generalization.

## Vector search

- Implemented: yes. Source: app/retrieval/vector_store.py.
- Tested: tests/test_retrieval.py.
- Demonstrated in real run: Yes: dense and hybrid.
- Infrastructure only: No.
- Limitation: Exact in-memory FAISS search; no distributed or approximate-index claim.

## BM25

- Implemented: yes. Source: app/retrieval/bm25_retriever.py.
- Tested: tests/test_bm25.py.
- Demonstrated in real run: Yes.
- Infrastructure only: No.
- Limitation: Corpus-specific lexical ranking.

## Hybrid retrieval

- Implemented: yes. Source: app/retrieval/hybrid_retriever.py.
- Tested: tests/test_hybrid.py.
- Demonstrated in real run: Yes: RRF with dense/BM25 candidates.
- Infrastructure only: No.
- Limitation: No learned fusion or new retrieval algorithm is claimed.

## Reranking

- Implemented: yes. Source: app/retrieval/reranker.py.
- Tested: tests/test_reranker.py.
- Demonstrated in real run: No final study run.
- Infrastructure only: Yes, for research comparison.
- Limitation: Lazy CrossEncoder is implemented with mocked scoring tests; real effectiveness unmeasured.

## Query rewriting

- Implemented: yes. Source: app/generation/query_rewriter.py.
- Tested: tests/test_query_rewriter.py.
- Demonstrated in real run: One compatibility check only; not a final strategy comparison.
- Infrastructure only: Yes, for comparative research.
- Limitation: Do not claim measured retrieval improvement from rewriting.

## Recall@K

- Implemented: yes. Source: app/evaluation/retrieval_metrics.py.
- Tested: tests/test_retrieval_metrics.py.
- Demonstrated in real run: Yes: main study and retrieval-only k=1/3/5.
- Infrastructure only: No.
- Limitation: Any-relevant-hit recall, not fraction of all relevant chunks.

## MRR

- Implemented: yes. Source: app/evaluation/retrieval_metrics.py.
- Tested: tests/test_retrieval_metrics.py.
- Demonstrated in real run: Yes.
- Infrastructure only: No.
- Limitation: Rank is measured within returned results; retrieval-only MRR uses each truncated top-k ranking.

## Hallucination evaluation

- Implemented: yes. Source: app/evaluation/hallucination_refusal.py.
- Tested: tests/test_hallucination_refusal.py.
- Demonstrated in real run: Yes, with refusal metrics.
- Infrastructure only: No.
- Limitation: Unsupported-claim judgment against retrieved evidence; not an independent factual-truth detector.

## Groundedness evaluation

- Implemented: yes. Source: app/evaluation/groundedness.py.
- Tested: tests/test_groundedness.py.
- Demonstrated in real run: Yes.
- Infrastructure only: No.
- Limitation: Same model generates and judges; strict parsing does not guarantee judge accuracy.

## Benchmark design

- Implemented: yes. Source: app/schemas/benchmark.py; app/experiments/phase25_benchmark.py; data/benchmarks/ca_small_benchmark.json.
- Tested: tests/test_benchmark.py; tests/test_phase25.py.
- Demonstrated in real run: Yes: 30 authored items, fixed 10-question evaluation subset.
- Infrastructure only: No.
- Limitation: Source-backed labels, but no claim of independent expert annotation or standardized benchmark status.

## Experiment design

- Implemented: yes. Source: app/experiments/experiment_runner.py; app/experiments/phase25_execute.py.
- Tested: tests/test_experiment_runner.py; tests/test_phase25.py.
- Demonstrated in real run: Yes: same corpus/subset/model/top-k across three strategies.
- Infrastructure only: No.
- Limitation: One run per strategy; no repeated trials or confidence intervals.

## Failure analysis

- Implemented: yes. Source: app/evaluation/failure_analysis.py.
- Tested: tests/test_failure_analysis.py.
- Demonstrated in real run: Yes: all main and smoke results.
- Infrastructure only: No.
- Limitation: Deterministic categories derived from stored scores; not causal root-cause proof.

## Latency tracking

- Implemented: yes. Source: app/generation/llm_service.py; app/generation/gemini_transport.py; app/experiments/experiment_runner.py.
- Tested: tests/test_generation.py; tests/test_gemini_transport.py; tests/test_experiment_runner.py.
- Demonstrated in real run: Yes.
- Infrastructure only: No.
- Limitation: Client elapsed time, not provider compute time; new experiment timing excludes deliberate quota waits and judges.

## Token usage tracking

- Implemented: yes. Source: app/schemas/generation.py; app/experiments/phase25.py.
- Tested: tests/test_generation.py; tests/test_gemini_transport.py; tests/test_phase25.py.
- Demonstrated in real run: Yes: 168,587 cumulative tokens.
- Infrastructure only: No.
- Limitation: Generation-only experiment totals differ from cumulative API accounting including judges/compatibility.

## Cost tracking

- Implemented: yes. Source: app/generation/llm_service.py; app/generation/gemini_transport.py.
- Tested: tests/test_generation.py; tests/test_gemini_transport.py.
- Demonstrated in real run: Missing-value handling demonstrated; no real monetary value reported.
- Infrastructure only: Numeric capture is mocked-provider evidence only.
- Limitation: Native Gemini cost is null. Do not claim measured cost savings or assume a zero charge.

## Side-by-side comparison

- Implemented: yes. Source: app/experiments/comparison_runner.py.
- Tested: tests/test_comparison_runner.py.
- Demonstrated in real run: Yes: main comparison 29261415-a8ea-4828-8128-3154ca5aaafa.
- Infrastructure only: No.
- Limitation: Metrics with denominators; no weighted score, ranking, or statistical winner.

## Chunk-size / top-k experiment infrastructure

- Implemented: yes. Source: app/experiments/sweep_runner.py; app/experiments/phase25_execute.py.
- Tested: tests/test_sweep_runner.py; tests/test_retrieval_metrics.py.
- Demonstrated in real run: Retrieval-only top-k yes; generation sweeps and chunk-size sensitivity not run.
- Infrastructure only: Chunk-size hook and generation sweeps only.
- Limitation: Chunk-size builder must rebuild/remap evidence. All current units are under 256 words; meaningful sensitivity study skipped.

## Reproducible experiment tracking

- Implemented: yes. Source: app/experiments/models.py; app/experiments/experiment_runner.py; app/experiments/phase25_resume.py.
- Tested: tests/test_experiment_runner.py; tests/test_phase25_resume.py.
- Demonstrated in real run: Yes: IDs, hashes, snapshots, six replayed calls, immutable completed results.
- Infrastructure only: No.
- Limitation: Supports audit/reconstruction, not bit-identical stochastic LLM output. Finished raw diagnostic bodies were pruned.

## Answer correctness

- Implemented: yes. Source: app/evaluation/answer_correctness.py.
- Tested: tests/test_answer_correctness.py.
- Demonstrated in real run: Yes.
- Infrastructure only: No.
- Limitation: Ground-truth comparison on eight answerable questions; unanswerable items skipped.

## Citation accuracy

- Implemented: yes. Source: app/evaluation/citation_accuracy.py.
- Tested: tests/test_citation_accuracy.py.
- Demonstrated in real run: Yes.
- Infrastructure only: No.
- Limitation: Checks metadata validity and semantic support; generation lists context citations, not explicit claim-level links.

## Frontend dashboard

- Implemented: yes. Source: frontend/.
- Tested: npm run lint; npm run build.
- Demonstrated in real run: Saved result viewing supported; not itself an experimental treatment.
- Infrastructure only: User interface.
- Limitation: No browser E2E suite or authentication; execution forms need configured backend services.

## LLM-only baseline (unsupported claim)

- Implemented: no dedicated baseline in the current ExperimentRunner/config.
- Tested: standalone LLM service is tested; an LLM-only experiment baseline is not.
- Demonstrated in real run: no.
- Infrastructure only: standalone text generation exists, but is not a baseline workflow.
- Limitation: do not list LLM-only baseline support as completed. It remains future work.
