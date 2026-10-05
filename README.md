# RAGEval

RAGEval is a research-oriented evaluation framework for Retrieval-Augmented Generation systems.

## Goals

RAGEval will compare RAG architectures across:

- Retrieval quality
- Answer correctness
- Groundedness
- Citation quality
- Hallucination rate
- Latency
- Token usage
- Cost

## Current Status

Phases 1–24 implemented; simplified Phase 25 real-data evaluation completed below.

## Docker

Use Docker Desktop with Linux containers. The image uses Python 3.11 slim,
installs the pinned requirements, and serves FastAPI on port 8000. No embedding
or reranker models are downloaded during the build or app startup.

```powershell
docker build -t rageval .
docker run --env-file .env -p 8000:8000 rageval
```

Supply `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` through the runtime
environment (`--env-file` or `-e`). The image excludes `.env` and does not contain
API credentials. Use your own API key and configured model (baseline:
`kr/claude-sonnet-4.5`).

For OmniRoute running on the Windows host:

- Local Python execution: `LLM_BASE_URL=http://localhost:20128/v1`
- Docker Desktop execution: `LLM_BASE_URL=http://host.docker.internal:20128/v1`

The `.env` supplied to Docker may therefore need `host.docker.internal` instead
of `localhost`, which refers to the container itself inside Docker.

Check <http://localhost:8000/>, <http://localhost:8000/health>, and
<http://localhost:8000/docs> after starting the container.

The default `app.main:app` preserves existing API behavior: research execution
requires prebuilt services injected through `create_app` and returns 503 until
configured. Environment variables configure the existing LLM service when it is
constructed; they do not build a corpus or wire research components automatically.
Chunk-size sweep execution over HTTP remains unsupported (501).

Local data and results are excluded from the image. To retain saved results
across container removal, optionally add `-v rageval-results:/app/results` to
the `docker run` command.

## Phase 25: controlled real-data evaluation

The real corpus is `data/corpus/Week # 02 Slides.pdf`, covering technology
trends, power/energy, and dependability: 45 pages, 41 nonempty extracted units,
2,156 words, and 41 chunks at 512 words with 50-word overlap. The preserved
30-question benchmark contains 24 answerable and 6 unanswerable questions.

The completed evaluation uses a fixed **10-question subset** (8 answerable,
2 unanswerable), with multiple categories and easy/medium/hard questions.
Selected IDs are `ca_small_001`, `003`, `008`, `012`, `015`, `018`, `023`,
`024`, `025`, and `029` (all share the `ca_small_` prefix). The exact list and
benchmark hash are in `data/processed/phase25/evaluation_subset.json`.

Dense, BM25, and hybrid retrieval were compared at `top_k=5` using native
Gemini `gemini-3.5-flash-lite`, with the existing strict evaluation protocol:

- Dense: hit Recall@5 1.00, MRR 0.7604, mean correctness 0.625,
  mean groundedness 0.90; 3 of 10 questions had detected failures.
- BM25: hit Recall@5 1.00, MRR 0.8750, mean correctness 0.750,
  mean groundedness 1.00; 2 of 10 questions had detected failures.
- Hybrid: hit Recall@5 1.00, MRR 0.8750, mean correctness 0.750,
  mean groundedness 1.00; 2 of 10 questions had detected failures.

Retrieval/correctness denominators exclude the two unanswerable items.
Citation accuracy was 1.00 for applicable answers in each strategy; refusals
are excluded. Both unanswerable questions received correct refusals in each
strategy. Failed-refusal and judged hallucination rates were 0; the latter
excludes answerable refusals. Incorrect refusals still count as failures.
All denominators, latency, tokens, and per-question results are preserved.

The smoke comparison is `09cda796-2a0e-496a-af4d-0a78a2a021d3`.
The 10-question comparison is `29261415-a8ea-4828-8128-3154ca5aaafa`.
Individual results are under `results/experiments/`, comparisons under
`results/comparisons/`, and analyses under `results/failure_analysis/`.
The consolidated artifact is `results/ca_small_gemini_final_summary.json`.

A separate hybrid retrieval-only check at top-k 1/3/5 used **zero Gemini
calls**: hit recall was 0.75/1.00/1.00 and truncated-ranking MRR was
0.75/0.875/0.875. Evidence and results are in
`results/phase25/retrieval_top_k.json`. Hit recall measures whether any labeled
chunk was found; it does not establish complete coverage of multi-chunk answers.

Gemini free-tier quota limited experiment scale. The original HTTP 429 was
preserved; the paced continuation completed 136 new calls without another
interruption, using 148,925 tokens. Cumulative fresh Phase 25 usage, including
compatibility checks and successful calls from the interrupted attempt, is
168,587 tokens over 158 successful calls (159 attempts). Cost is unavailable
and recorded as `null`, not estimated. Experiment token totals cover answer
generation only; the cumulative ledger also includes judge calls.

Set `LLM_PROVIDER=gemini`, `GEMINI_MODEL=gemini-3.5-flash-lite`, and supply
`GEMINI_API_KEY` securely through the local environment. From the project root:

```powershell
.\venv\Scripts\python.exe -m app.experiments.phase25_execute minimal
# Only when explicitly resuming a preserved quota interruption:
.\venv\Scripts\python.exe -m app.experiments.phase25_execute minimal --resume-rate-limit
```

These commands resume unfinished work and skip completed experiments. Run one
executor at a time. Completed questions are checkpointed, and identical saved
responses can resume an interrupted question without another provider call.
New requests target 5.5-second spacing; deliberate pacing is excluded from new
experiment latency. A 429 saves available retry timing and stops without an
automatic retry. Judge/schema failures also stop execution without score repair.

The 30-question full runs, LLM-only/modifier strategies, and generation sweeps
were excluded from this simplified scope. The chunk-size sweep was skipped:
all page units are shorter than 256 words (maximum 132), so the existing
page-preserving chunker produces identical boundaries at 256/512/1024 words.

This is a small feasibility demonstration, not evidence for broad statistical
conclusions. Generation and judging use the same model; these scores are not
independent human assessments. Evidence is extracted slide text, so image-only
equations and diagrams are outside this benchmark. Validation: 50 focused tests
and all 571 tests passed.
