# RAGEval

## Overview

RAGEval is a Python research framework for comparing retrieval strategies and
evaluating retrieval-augmented answers. It includes a FastAPI backend, a Next.js
research dashboard, typed experiment artifacts, and strict LLM-based judges.
The repository contains a completed small real-data study.

## Research Question

How do dense, lexical, and hybrid retrieval affect evidence retrieval, answer
quality, refusal behavior, latency, and token usage on the same QA benchmark?
Hold the corpus, questions, model, and evaluation protocol fixed, and preserve
metric denominators and unavailable values when comparing results.

## Features

- PDF/TXT/DOCX ingestion with source/page metadata and word-based chunking.
- Sentence-transformer embeddings, FAISS cosine search, BM25, and hybrid RRF.
- Optional cross-encoder reranking and query rewriting.
- RAG generation through native Gemini or OpenAI-compatible providers.
- Hit Recall@K, reciprocal rank/MRR, correctness, groundedness, citation accuracy,
  hallucination/refusal evaluation, and deterministic failure analysis.
- Monotonic latency measurement, token usage, and provider-reported cost when
  available; missing metadata remains `null`.
- Typed benchmarks, experiment execution, side-by-side comparisons, top-k and
  chunk-size sweep infrastructure, checkpoints, and paced real-study execution.
- FastAPI endpoints, a Next.js dashboard, and a backend Docker image definition.

Reranking, rewriting, and generation-based sweeps are implemented and tested but
were not evaluated in the final real study. A dedicated **LLM-only experiment
baseline is not implemented**: `ExperimentConfig` accepts dense/BM25/hybrid.
Calling the standalone LLM service is not equivalent to having that baseline.

## Architecture

```text
Documents -> ingestion -> chunks -> embeddings / BM25 indexes
Question -> optional rewrite -> dense / BM25 / hybrid retrieval -> optional rerank
         -> original question + evidence -> generation -> evaluation
         -> experiment JSON -> comparisons / failure analysis
```

`LLMService.generate()` and `generate_result()` isolate provider transport.
Native Gemini separates system instructions and user content through
`generateContent`; the OpenAI-compatible path uses `stream=False`. Judges retain
strict JSON validation and separate correctness/evidence rubrics. Importing the
FastAPI app does not initialize clients or download models at import. Opt-in
local startup builds services once per server process.

## Project Structure

```text
backend/
  app/              # API, ingestion, retrieval, generation, evaluation, runners
  tests/            # Offline tests with mocked external services
  data/             # Corpus, benchmarks, processed units/chunks/embeddings
  results/          # Experiments, comparisons, failure_analysis, final
  docs/             # CV evidence and cleanup audit
  requirements.txt
  pytest.ini
  Dockerfile
  .dockerignore
frontend/           # Unchanged Next.js research dashboard
.gitignore
.dockerignore
.env.example
README.md
```

Sweep infrastructure creates `backend/results/sweeps/` when used. No generation or
chunk-size sweep result is claimed for the simplified real study.

## Installation

Use Python 3.11 and Node.js 22.13 or newer. PowerShell, from the project root:

```powershell
Copy-Item .env.example .env
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Do not overwrite an existing local `.env`. Existing checkouts may retain the ignored
root `venv/`; from `backend/`, use `..\venv\Scripts\python.exe` instead of
`.\.venv\Scripts\python.exe` in the commands below. Windows virtual environments
should be recreated rather than moved. Dependency versions are pinned in
`backend/requirements.txt` and `frontend/package-lock.json`. Embedding/reranker weights
load lazily and may require a first-use download; tests do not download them.

## Environment Variables

Native Gemini, used by the real study:

```dotenv
LLM_PROVIDER=gemini
GEMINI_MODEL=gemini-3.5-flash-lite
GEMINI_API_KEY=your-local-key
```

Alternatively, use an OpenAI-compatible service:

```dotenv
LLM_PROVIDER=openai
LLM_BASE_URL=http://localhost:20128/v1
LLM_MODEL=your-model-id
LLM_API_KEY=your-local-key
```

These are placeholders, not credentials. Process environment variables override
`backend/.env` if present, otherwise the root `.env`. Native Gemini does not use its OpenAI-compatible endpoint.
Never put provider credentials in frontend `NEXT_PUBLIC_*` variables.

## Running Backend

From `backend/` (activate its virtual environment to use plain `uvicorn` or `pytest`):

```powershell
.\.venv\Scripts\Activate.ps1
$env:RAGEVAL_AUTO_INIT = "true"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Set `RAGEVAL_AUTO_INIT=true` in the process environment (PowerShell above), not
only in `.env`. Provider credentials still load from the existing root `.env`.
Use `LLM_PROVIDER=gemini`, `GEMINI_MODEL=gemini-3.5-flash-lite`, and your
`GEMINI_API_KEY`. Optional process variable `RAGEVAL_CORPUS_PATH` defaults to
`data/corpus/Week # 02 Slides.pdf`; relative paths resolve against `backend/`.

Startup extracts the document, chunks at 512 words / 50 overlap, embeds and builds
FAISS/BM25/hybrid once per worker. `/query` uses dense retrieval. Experiments share
these indexes and the configured model; use matching model/chunk settings and
benchmark relevance IDs. Reranker weights remain lazy. Reloads/new workers rebuild;
requests do not. Startup fails clearly for invalid configuration or empty corpus.
Existing artifacts are neither rewritten nor experiments executed at startup.

Or use the interpreter directly, with the same process environment:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open `/`, `/health`, or `/docs` at `http://localhost:8000`. The default app can
read saved results and perform failure analysis. With auto-init enabled, query
and experiment endpoints have runtime services. With auto-init disabled (default),
execution returns **503** unless prebuilt services are injected through
`create_app(pipeline=..., experiment_runner=...)`. Explicit injection takes precedence. HTTP chunk-size execution returns **501** because a trusted rebuild
hook is required. The real-study CLI wires its own components.

## Running Frontend

In a separate terminal, from the repository root:

```powershell
cd frontend
npm ci
Copy-Item .env.example .env.local
npm run dev
```

Open `http://localhost:3000`. `/backend/*` proxies to `NEXT_PUBLIC_API_BASE_URL`
(default `http://localhost:8000`), configured before startup/build. Load the main
comparison ID below in the Comparisons page to inspect saved real results.
There is no authentication or database.

```powershell
npm run lint
npm run build
npm run start
```

Dependencies and generated output are ignored by Git; install and build locally
as needed.

## Docker

From the repository root, build with the backend context and its own `.dockerignore`.
The backend Dockerfile uses Python 3.11 slim, installs `libgomp1` and pinned Python
dependencies, copies app source, and runs Uvicorn on port 8000.

```powershell
docker build -t rageval-backend ./backend
docker run --env-file .env -p 8000:8000 rageval-backend
```

Runtime env files and keys are excluded. For OmniRoute on the Windows host,
local Python uses `http://localhost:20128/v1`; Docker Desktop uses
`http://host.docker.internal:20128/v1` as `LLM_BASE_URL`. Gemini's hosted endpoint
needs no host-address substitution.

The backend image contains neither local research artifacts nor the frontend.
Mount the project's `backend/results` directory at `/app/results` to read saved results
through the container API. Auto-init is off by default in Docker. To enable it,
mount the corpus at `/app/data/corpus/Week # 02 Slides.pdf` (or another configured
container path) and pass `-e RAGEVAL_AUTO_INIT=true`; model weights must be cached
or downloadable at runtime. Never download weights at image build time.
This cleanup audit did not rebuild or run Docker.

## Running Tests

From `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Tests use temporary fixtures/mocks and do not call providers or download models.
To verify without creating Python caches:

```powershell
$env:PYTHONDONTWRITEBYTECODE = '1'
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider
```

## Real Evaluation Study

**Week # 02 Slides.pdf** covers technology trends, power/energy, and dependability:
**45 pages, 41 text units, 2,156 words, and 41 chunks** at 512 words / 50 overlap.
The full benchmark has **30 questions: 24 answerable and 6 unanswerable**.

The final evaluation uses **10 fixed questions** (8 answerable, 2 unanswerable):
`ca_small_001`, `003`, `008`, `012`, `015`, `018`, `023`, `024`, `025`, `029`
(all share the `ca_small_` prefix). Exact IDs and the benchmark hash are in
`backend/data/processed/phase25/evaluation_subset.json`.

All strategies used native Gemini `gemini-3.5-flash-lite`, `top_k=5`, and the same
strict evaluation protocol:

- Dense: Recall@5 **1.00**, MRR **0.7604**, mean correctness **0.625**,
  groundedness **0.90**; **3** questions with detected failures.
- BM25: Recall@5 **1.00**, MRR **0.875**, mean correctness **0.750**,
  groundedness **1.00**; **2** questions with detected failures.
- Hybrid: Recall@5 **1.00**, MRR **0.875**, mean correctness **0.750**,
  groundedness **1.00**; **2** questions with detected failures.

All three correctly refused both unanswerable questions. Citation accuracy was
1.00 on applicable answers. Hallucination rate was 0 on substantive answerable
responses; this does not erase incorrect refusals. Artifacts retain denominators.

The separate **retrieval-only** hybrid top-k study made no Gemini calls. At
k=1/3/5, hit recall was **0.75/1.00/1.00**, and truncated-ranking MRR was
**0.75/0.875/0.875**. Retrieval improved from k=1 to k=3 and did not improve
further at k=5 on this subset. Hit recall does not establish complete coverage
of all relevant evidence in multi-hop questions.

Canonical evidence:

- [Final summary](backend/results/final/ca_small_gemini_final_summary.json).
- [Main comparison](backend/results/comparisons/29261415-a8ea-4828-8128-3154ca5aaafa.json).
- [Retrieval-only study](backend/results/final/retrieval_top_k.json).
- [Compact API usage ledger](backend/results/final/usage_ledger.json).
- Main experiments: dense `4dca9879-d7e1-4bdf-ac30-4dd08ccb7fc6`,
  BM25 `0f1cf518-9c3b-4ebb-a584-737caeea0e43`,
  hybrid `64a83121-34c5-4d49-a0b2-3c729aabd98d`.
- Smoke comparison: `09cda796-2a0e-496a-af4d-0a78a2a021d3`; retained to support
  execution gates and provenance.

The paced continuation made **136 new calls / 148,925 tokens** without another
interruption. Cumulative study accounting, including compatibility and successful
calls from the interrupted attempt, is **158 successes / 159 attempts / 168,587
tokens**. Cost was unavailable (`null`). Experiment usage covers generation only;
cumulative usage also includes judges. Raw diagnostic bodies were pruned after
consolidation; call IDs, prompt/response hashes, timing, usage, and status remain.

From `backend/`, resume or verify preserved completion:

```powershell
.\.venv\Scripts\python.exe -m app.experiments.phase25_execute minimal
# Explicitly resume a future saved quota interruption, respecting its cooldown:
.\.venv\Scripts\python.exe -m app.experiments.phase25_execute minimal --resume-rate-limit
```

With the completed summary, these commands skip completed runs and make no API
calls. For a new independent study, inject prebuilt components into
`ExperimentRunner`, `ComparisonRunner`, or `SweepRunner` with fresh IDs/configs.
Do not delete published evidence to force execution. Source prompts, benchmark
provenance, embeddings, configs, hashes, and saved results support reconstruction;
exact stochastic LLM outputs are not guaranteed. Run one paced executor at a time.

## Limitations

- Small lecture-slide corpus and 10-question subset; results are illustrative,
  not statistically generalizable.
- Gemini free-tier quota limited scale. One historical HTTP 429 is retained;
  requests target 5.5-second spacing and stop without an automatic retry loop.
- Generation and judging use the same model, not independent human assessment.
- Image-only equations and diagrams are outside the extracted-text benchmark.
- Chunk-size sensitivity was skipped: all text units have fewer than 256 words
  (maximum 132), yielding identical page-preserving boundaries at 256/512/1024.
  Rebuild-hook infrastructure remains implemented and tested.
- Reranking, rewriting, and broader sweeps are capabilities, not final-study
  comparisons. OpenAI-compatible providers have mocked coverage; native Gemini
  is demonstrated by the retained real study.
- Cost extraction exists, but no monetary cost is available for the real run.
- The API/dashboard are local research tools, not authenticated hosted services.

## Future Work

Larger corpora/benchmarks; independent human or alternative-model judging;
controlled provider/model comparisons; a dedicated LLM-only baseline; meaningful
chunk-size studies; repeated runs and statistical confidence estimates.

See [CV claim audit](backend/docs/cv_claim_audit.md) for claim-by-claim evidence and
[cleanup audit](backend/docs/cleanup_audit.md) for artifact decisions.
