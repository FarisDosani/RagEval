# Cleanup and verification audit

The sections below record the prior artifact cleanup. All backend-relative paths
now live under `backend/`; the later structure-only cleanup is recorded at the end.

## Scope and safety

This audit reorganized the completed simplified Phase 25 repository. No API
requests, model downloads, benchmark executions, or new research features were
performed. Only `git status` was used as a Git command. The Git index was read
locally to identify tracked paths for the secret scan.

No implemented retrieval, generation, evaluation, sweep, comparison, or frontend
capability was removed. The API contract and benchmark labels were unchanged.

## Artifact decisions

- **KEEP — production and tests:** `app/**`, `tests/**`, frontend source/config,
  Python requirements, npm manifest/lockfile, Dockerfile, and meaningful tests.
  The `phase25*.py` modules remain because the CLI, resume logic, benchmark
  provenance, and tests reference them; they are not abandoned provider probes.
- **KEEP — data:** `data/corpus/Week # 02 Slides.pdf`, both benchmark JSON files,
  and all six `data/processed/phase25/*.json` files. Units, chunks, embeddings,
  provenance, manifest, and subset IDs are needed for reconstruction/tests.
- **KEEP — final evidence:** the three main experiments, main comparison, their
  failure analyses, retrieval-only top-k study, and final summary. Results retain
  answers, evidence, judge scores/reasons, configs, IDs, hashes, and denominators.
- **KEEP — smoke evidence:** three smoke experiments, their comparison and
  analyses. The summary and resume gates reference them, so deleting them would
  make completion checks incomplete. There is one canonical main comparison.
- **MOVE:** `results/ca_small_gemini_final_summary.json` to
  `results/final/ca_small_gemini_final_summary.json`; and
  `results/phase25/retrieval_top_k.json` to `results/final/retrieval_top_k.json`.
  Code, README, and embedded summary references were updated.
- **DELETE — 159 raw call captures:** `results/phase25/<run-id>/calls/*.json`.
  Before deletion, all accounting was consolidated into
  `results/final/usage_ledger.json`: call/run IDs, context, timestamps, request and
  response hashes, usage, latency, and success/error status. Its 158 successes,
  159 attempts, and 168,587 tokens match the published claims exactly.
- **DELETE — duplicate outputs:** 41 generation snapshots, 6 replay traces, and
  1 compatibility report beneath those finished run directories. Compatibility
  pass states and usage remain in the final summary/ledger. Published typed
  results preserve generated answers and retrieved evidence.
- **DELETE — completed scratch state:** 5 completed checkpoint files and 1
  expired pacing file. Each checkpoint's questions and config/benchmark hashes
  were checked for exact equality against its completed experiment before deletion.
  New runs still create resumable checkpoints and active-call replay records.
- **DELETE — obsolete caches:** all 1,793 discovered main-workspace/venv
  `__pycache__`/`.pytest_cache` directories, including seven abandoned provider
  bytecode files under `results/ca_small_diagnostics`. No old provider source
  scripts or old large-corpus experiment results existed in the audited main tree.
- **DELETE — generated frontend files:** `frontend/node_modules` and
  `frontend/.next`, after lint and production build passed. `npm ci` and
  `npm run build` restore them; source and lockfile remain unchanged.
- **DELETE — empty scratch directories:** finished runtime directories,
  empty `tmp/pdfs/ca_small`, temporary audit bookkeeping, and empty obsolete
  diagnostics directories. No code/test/README dependency was found for them.
- **IGNORE / KEEP LOCALLY:** `.env`, `.env.docker`, `venv/`, and local editor files.
  Secrets are not copied into documentation or results. Example env files remain
  versionable. Local Python dependencies are retained, excluding generated caches.
- **IGNORE / LEAVE UNTOUCHED:** `.kilo/worktrees/broken-deerstalker` is a separate
  Git worktree, not main-project research evidence. It was not deleted or modified;
  removing it safely would require separate worktree lifecycle work. `.kilo/`
  is excluded by both Git and Docker ignore rules.

Raw captures were removed only after reference search and accounting validation.
Finished call bodies can no longer be replayed; future active request records
retain replay support. Exact stochastic regeneration is not promised.

## Canonical result set

`results/` retains 6 experiments, 2 comparisons, 6 failure analyses, and 3 final
artifacts (summary, retrieval-only study, compact usage ledger). The three main
experiment IDs and main comparison are documented in the README. No fake sweep
file or empty sweep result was introduced.

## Verification

- Python: **574 passed**, including new artifact-reference/accounting regressions.
  Run with bytecode and pytest cache writing disabled during cleanup verification.
- Backend: imports, `/`, `/health`, `/docs`, all three main result reads, main
  comparison read, and failure analysis passed through TestClient. Provider
  construction and model loading were explicitly forbidden during this check.
- Default unconfigured query execution still returns 503; no API behavior changed.
- Completed-study CLI: both stages skipped; zero new provider calls.
- Frontend: **lint passed; production build passed**. Node 22.23.3 / npm 10.9.9.
  All reported application routes and the not-found page were generated.
- Secret scan: current tracked paths and new project artifacts checked for
  obvious key/private-key patterns and exact local API-key values without printing
  values. `.env` and `.env.docker` are not in the Git index. No findings.
  This is a current-file scan, not a claim about all Git history.
- Docker: source/ignore rules audited; image build/runtime were not rerun here.

Final scan: 127 current project files checked; zero secret matches and zero
generated cache/dependency directories in the main workspace. The separate
ignored `.kilo` checkout was excluded. Git reports 215 unstaged removals
(213 redundant artifacts plus the two relocated source paths), 7 modified
tracked files, and 4 untracked path groups for the new example, documentation,
final artifacts, and integrity test. Nothing was staged, committed, or pushed.

## Documentation and claim corrections

README now covers setup, provider configuration, API injection, dashboard,
Docker, tests, actual study, reconstruction/resume behavior, and limitations.
`docs/cv_claim_audit.md` separates implemented/tested infrastructure from real
study evidence. In particular, there is no dedicated LLM-only experiment baseline,
no real monetary cost measurement, no real reranker comparison, and no meaningful
chunk-size sensitivity result. Those capabilities/results are not overstated.

## Repository structure cleanup

Moved app, tests, data, results, docs, requirements, pytest configuration, and
Dockerfile into backend. Research artifacts were moved without content changes.
The clean detached .kilo checkout had no application references and was removed
with git worktree remove; no branches or commits were deleted. This supersedes
the earlier decision to leave that checkout untouched.

Root environment files and the ignored local venv remain in place. Default LLM
configuration uses backend/.env when present and falls back to root .env; explicit
environment paths and process-variable precedence remain unchanged. Module-relative
data/results paths naturally resolve under backend. Frontend source and proxy are
unchanged. Docker builds use backend as context with its own secret exclusions.

Structure verification: **575 backend tests passed** from backend; import app and
actual Uvicorn health startup passed. TestClient returned 200 for /, /health, /docs,
and the published main experiment/comparison. All Markdown file links resolved.
All 26 data/result files matched their pre-move SHA-256 checksums.

Frontend npm ci, lint, and production build passed with no source or lockfile
changes in this structure cleanup. Installation required an approved sandbox
retry after EPERM; npm reported five high-severity dependency advisories. No
dependency upgrades were attempted in this structure-only task. Generated frontend
output remains local and ignored. Docker context/COPY/ignore paths were checked;
the image was not rebuilt. Git status was reviewed; moved files appear as old-path
deletions plus untracked backend until staged by the owner. Prior cleanup changes
remain preserved. No staging, commits, pushes, provider calls, or experiments.
