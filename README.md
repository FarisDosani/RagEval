# RAGEval

Research-oriented RAG evaluation framework for comparing retrieval and generation pipelines.

## Overview

A plausible answer is not enough to evaluate a RAG system. RAGEval measures whether
retrieval finds useful evidence, whether answers are correct and grounded, how the
system handles unanswerable questions, and what each configuration costs in latency
and tokens. Saved results make failures and trade-offs inspectable.

Compare LLM-only, dense-vector, BM25, hybrid, reranked, and query-rewritten pipelines
using Python, FastAPI, FAISS, sentence-transformers, and rank-bm25. A Next.js dashboard
provides document uploads, interactive queries, comparisons, and failure analysis.

**Evidence boundary:** the retained real study compares **Dense, BM25, and Hybrid**
on a controlled **10-question subset** of a 30-question benchmark. LLM-only,
reranking, rewriting, and broader sweeps are implemented/tested capabilities, not
additional real-study results.

## Features

- **Generation baselines:** LLM-only direct answers and retrieval-augmented answers
  through native Gemini or an OpenAI-compatible provider.
- **Retrieval:** FAISS cosine search, BM25, and hybrid Reciprocal Rank Fusion (RRF),
  with optional cross-encoder reranking and query rewriting.
- **Documents:** PDF, UTF-8 TXT, and DOCX ingestion; page/source metadata;
  word-based chunking; session document upload and indexing.
- **Interactive queries:** Dense / BM25 / Hybrid / LLM Only selector, configurable
  top-k, ranked evidence, citations, model, latency, and usage display.
- **Research workflow:** typed controlled QA benchmarks, reproducible experiment
  tracking, checkpoint support, side-by-side comparisons, top-k sweeps, and
  chunk-size sweeps with an explicit corpus/index rebuild hook.
- **Diagnostics:** strict generation judges and deterministic failure analysis.
- **Interfaces:** FastAPI research API, Next.js frontend, and backend Docker support.

## Evaluation Metrics

- **Retrieval:** benchmark-style hit Recall@K is 1 when any relevant chunk appears
  in the first K results; MRR averages the reciprocal rank of the first relevant
  chunk. Items without relevant chunk labels are excluded with transparent counts.
- **Generation:** answer correctness against ground truth; groundedness against
  retrieved evidence; citation metadata validity and semantic support; unsupported
  claims and benchmark refusal behavior. Strict judges reject malformed output.
- **Efficiency:** monotonic elapsed latency, prompt/completion/total token usage,
  and cost only when reliable provider-reported cost metadata is available.
  Pricing is not hardcoded or inferred; unavailable values remain `null`.

**LLM-only applicability:** correctness, exact-refusal behavior, latency, tokens,
and available cost apply. Recall@K, MRR, evidence groundedness, citation accuracy,
and evidence-based hallucination rates are **N/A (`null`), never zero**. Evidence
judges are skipped. Substantive answerable responses have `unassessed_answer` support
status, not an invented hallucination assessment. Refusal rates use the benchmark's
answerability labels, not general-world answerability.

Experiment usage totals cover answer generation; the real-study ledger additionally
accounts for judge/diagnostic calls. Saved summaries preserve denominators and scope.

## Architecture

```text
Documents -> ingestion -> page-aware word chunks -> embeddings / lexical corpus
                                                       |
Question -> optional rewrite -> Dense | BM25 | Hybrid -> optional rerank
                                                       |
                         original question + evidence -> LLM
                                                       |
                                            Answer + evidence
                                                       |
                      Evaluation -> experiment JSON -> comparison / failure analysis

Question ------------------ LLM-only -----------------> LLM
                            (no retrieval, no context; evidence metrics N/A)
```

Provider transport is isolated behind `LLMService`. Gemini uses native
`generateContent`; OpenAI-compatible providers use non-streaming chat completions.
Model/client initialization does not occur at module import. Opt-in runtime startup
builds corpus indexes once per worker; queries reuse them. Reranker weights load
only when reranking is requested.

## Repository Structure

```text
RagEval/
  backend/
    app/              # API, runtime, ingestion, retrieval, generation, evaluation
    tests/            # Mocked/offline regression tests
    data/             # Corpus, benchmarks, processed evidence and embeddings
    results/          # Experiments, comparisons, failure_analysis, final summaries
    requirements.txt
    pytest.ini
    Dockerfile
    .dockerignore
  frontend/
    app/              # Next.js pages
    components/       # Query, corpus and research views
    lib/              # Backend API client
    tests/            # Frontend rendering/request checks
    package.json
    package-lock.json
  .gitignore
  .dockerignore
  .env.example
  README.md
```

Local secrets, virtual environments, node_modules, build output, caches, and runtime
uploads are ignored. Published research data/results remain versioned.

## Quick Start

Requires Python 3.11 and Node.js 22.13 or newer. Commands below use Windows
PowerShell. Start in the repository root.

### First-time backend setup

```powershell
py -3.11 -m venv venv
.\venv\Scripts\python.exe -m pip install -r backend\requirements.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Edit the local `.env` with your own credentials; never commit it:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-local-key
GEMINI_MODEL=gemini-3.5-flash-lite
```

Alternatively set `LLM_PROVIDER=openai`, `LLM_BASE_URL`, `LLM_API_KEY`, and
`LLM_MODEL` for an OpenAI-compatible service. Process variables override file
values. The backend reads `backend/.env` if present, otherwise root `.env`.

### Start the backend

```powershell
cd backend
..\venv\Scripts\Activate.ps1
$env:RAGEVAL_AUTO_INIT = "true"
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

If activation is unavailable, use `..\venv\Scripts\python.exe -m uvicorn` with the
same arguments. Existing environments need not be recreated.

Export `RAGEVAL_AUTO_INIT` in the **process environment**, not only `.env`.
Optional process settings:

- `RAGEVAL_CORPUS_PATH`: defaults to `data/corpus/Week # 02 Slides.pdf`; relative
  paths resolve against `backend/`.
- `RAGEVAL_MAX_UPLOAD_BYTES`: defaults to 10485760 (10 MiB).

Startup ingests/chunks the configured corpus at 512 words / 50 overlap and builds
FAISS, BM25, and Hybrid. First use may download embedding weights. Reloads rebuild
indexes; requests do not. Startup does not run experiments or overwrite study data.
With auto-init disabled, saved-result reads work; execution needs injected services
and otherwise returns 503. Use one worker for the session-based demo.

### Start the frontend

In a separate terminal, from the repository root:

```powershell
cd frontend
npm ci
if (-not (Test-Path .env.local)) { Copy-Item .env.example .env.local }
npm run dev
```

`npm ci` installs the committed lockfile; `npm install` is available when intentionally
updating dependencies. `NEXT_PUBLIC_API_BASE_URL` defaults to `http://localhost:8000`
and controls the `/backend/*` proxy. Never put provider secrets in public variables.

- Frontend: [localhost:3000](http://localhost:3000)
- Backend/health: [localhost:8000](http://localhost:8000), [/health](http://localhost:8000/health)
- Swagger: [localhost:8000/docs](http://localhost:8000/docs)

## Usage

1. Open Query and upload a PDF, TXT, or DOCX; wait for the unit/chunk counts.
2. Select Dense, BM25, Hybrid (default), or **LLM Only**.
3. Choose top-k where applicable, then ask a question.
4. Inspect the answer, ranked evidence, page/source references, citations, and usage.
5. In Experiments, supply benchmark JSON and explicit model/strategy settings.
6. In Comparisons, load a saved comparison or run configurations on the same benchmark.
7. Inspect Failure Analysis using a saved experiment ID.

LLM Only sends the question directly to the configured model without corpus
retrieval. Top-k is ignored, evidence/citations are empty, and the UI shows
**No retrieval used**. It provides a baseline for measuring whether RAG improves
performance. Retrieval rewriting/reranking are rejected for LLM-only configurations.

Uploads add to the current corpus. They reuse the embedding model and rebuild all
indexes before publishing a complete snapshot; failed uploads preserve the old
corpus and in-flight requests keep their original snapshot. Stored uploads are not
automatically re-indexed after restart. For controlled experiments, restart with the
original corpus and use matching model/chunk settings and relevance IDs; uploads
change the corpus. Generation and experiment execution may incur provider usage.

API details:

- `POST /documents/upload?filename=notes.txt`: raw bytes with
  `Content-Type: application/octet-stream`, not multipart; names cannot contain paths.
- `GET /documents`: current indexed metadata, including the startup document.
- `POST /query`: `question`, `top_k`, `retrieval_strategy`.
- Experiments/comparisons/top-k sweeps: run and saved-result endpoints in Swagger.
- HTTP chunk-size execution returns 501; Python sweeps require a trusted rebuild hook.

Unsupported, empty, malformed, and text-empty uploads return 400; oversize uploads
return 413. Reverse-proxy body limits must also permit the configured size. Scores
are strategy-specific and are not directly comparable between retrievers.

## Research Study

The source **Week # 02 Slides.pdf** covers technology trends, power/energy, and
dependability: **45 pages, 41 extracted units, 2,156 words, 41 baseline chunks**.
The benchmark contains **30 questions (24 answerable, 6 unanswerable)**. Final runs
use a fixed **10-question subset (8 answerable, 2 unanswerable)** with native Gemini
`gemini-3.5-flash-lite` and top-k=5.

Recorded results:

- **Dense:** Recall@5 1.00; MRR 0.7604; correctness 0.625; groundedness 0.90;
  3 questions with detected failures.
- **BM25:** Recall@5 1.00; MRR 0.875; correctness 0.750; groundedness 1.00;
  2 questions with detected failures.
- **Hybrid:** Recall@5 1.00; MRR 0.875; correctness 0.750; groundedness 1.00;
  2 questions with detected failures.

The separate **retrieval-only Hybrid top-k study** made no Gemini calls:

- **k=1:** hit Recall 0.75, truncated-ranking MRR 0.75.
- **k=3:** hit Recall 1.00, truncated-ranking MRR 0.875.
- **k=5:** hit Recall 1.00, truncated-ranking MRR 0.875.

These are small-sample observations, not a general strategy ranking. Hit Recall
measures at least one relevant hit, not complete multi-hop evidence coverage.
LLM-only, reranking, query rewriting, and generation/chunk-size sweeps were **not**
part of this final comparison; they remain supported/tested framework capabilities.

Evidence and reproducibility:

- [Final summary](backend/results/final/ca_small_gemini_final_summary.json)
- [30-question benchmark](backend/data/benchmarks/ca_small_benchmark.json)
  and [fixed subset](backend/data/processed/phase25/evaluation_subset.json)
- [Main comparison](backend/results/comparisons/29261415-a8ea-4828-8128-3154ca5aaafa.json)
  (load ID `29261415-a8ea-4828-8128-3154ca5aaafa` in the dashboard)
- [Retrieval-only study](backend/results/final/retrieval_top_k.json)
- [Compact usage ledger](backend/results/final/usage_ledger.json): 158 successful
  calls / 159 attempts / 168,587 tokens including diagnostics and judges; cost `null`.
- Main experiment IDs: Dense `4dca9879-d7e1-4bdf-ac30-4dd08ccb7fc6`,
  BM25 `0f1cf518-9c3b-4ebb-a584-737caeea0e43`, Hybrid `64a83121-34c5-4d49-a0b2-3c729aabd98d`.

Configs, benchmark/corpus hashes, chunks, embeddings, outputs, and provenance are
retained. Six experiment files include three main and three smoke runs. Historical
raw diagnostic bodies were consolidated into the ledger; exact stochastic model
outputs are not guaranteed reproducible. Do not delete results to force reruns.

## Failure Analysis

Post-processing uses existing results to classify retrieval misses, low-rank evidence,
incorrect/partial answers, ungrounded generation, citation failures, hallucination,
incorrect refusal, and failed refusal. Multiple categories may apply to one question.
Counts and rates retain explicit denominators; missing evidence does not invent rank
information. LLM-only skips retrieval failures and evidence-based judgments. A
failure-free result means no detected failure, not proof of correctness when metrics
are unavailable. No new model calls are needed for this analysis.

## Testing

Latest verified baseline: **609 backend tests** and **3 frontend checks** passed;
frontend lint and production build passed. Tests mock providers/models.

```powershell
# From repository root, using the installed environment:
cd backend
..\venv\Scripts\python.exe -m pytest
cd ..\frontend
node --test tests/llm-only.test.mjs
npm run lint
npm run build
```

Dependencies are pinned in `backend/requirements.txt` and the npm lockfile.

## Docker

From the repository root:

```powershell
docker build -t rageval-backend ./backend
docker run --rm --env-file .env -p 8000:8000 rageval-backend
```

The backend context uses Python 3.11 slim, `libgomp1`, pinned requirements, and
`COPY app/ ./app/`. It runs Uvicorn on port 8000. Its own `.dockerignore` excludes
secrets, data, results, tests, caches, and frontend artifacts. Weights are not
initialized at image build time. The basic command supports health/docs; auto-init
requires corpus access. For the local demo:

```powershell
docker run --rm --env-file .env -p 8000:8000 `
  -e RAGEVAL_AUTO_INIT=true `
  --mount "type=bind,source=$((Get-Location).Path)\backend\data,target=/app/data" `
  --mount "type=bind,source=$((Get-Location).Path)\backend\results,target=/app/results" `
  rageval-backend
```

The data mount also stores runtime uploads; the results mount permits saved-result
reads and new experiment writes. Embedding weights must be cached or downloadable
at startup. Gemini needs no host-address change. For a host OmniRoute service,
local Python uses `http://localhost:20128/v1`; Windows Docker Desktop uses
`http://host.docker.internal:20128/v1` as `LLM_BASE_URL` in its runtime env file.

## Limitations

- Session-local indexes reset on restart; uploaded files remain on disk but are not
  automatically re-indexed. Use a single worker for local research/demo sessions.
- No authentication, persistent index service, or production deployment hardening.
- Small corpus and 10-question subset; generation and judging use the same model.
- Text extraction does not capture image-only equations/diagrams or perform OCR.
- Source units are below 256 words (maximum 132), so 256/512/1024 word settings
  produce identical page-preserving boundaries. No meaningful chunk-size sensitivity
  result is claimed; rebuild-hook sweep infrastructure is implemented/tested.
- The real final study covers Dense/BM25/Hybrid only, despite broader framework support.
- Monetary cost is unavailable unless the provider returns usable cost metadata.

## Future Work

Persistent corpus/index storage, broader benchmark datasets, larger controlled
studies with independent evaluation, and deployment hardening.
