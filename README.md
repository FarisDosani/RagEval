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

Phase 1 — Project setup

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
