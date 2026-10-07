# RAGEval frontend

Run `npm install`, copy `.env.example` to `.env.local`, then `npm run dev`.
Open http://localhost:3000. Use `npm run lint` and `npm run build` to verify.

`NEXT_PUBLIC_API_BASE_URL` selects the FastAPI backend (default http://localhost:8000).
Restart Next.js after changing it; set it before production builds. The Next.js
rewrite forwards `/backend/*` to that URL, avoiding browser CORS changes.
Never put LLM credentials in frontend environment variables.

Benchmark run forms accept the existing JSON array schema; no file upload or
sample results are generated. Load persisted results by UUID on their respective
pages. The default backend returns 503 until research services are injected.
HTTP chunk-size execution is intentionally unavailable.

Types in `types/api.ts` mirror the existing FastAPI OpenAPI schemas. Cost values
use provider-reported units, and unavailable values remain unavailable. No results
are persisted in browser storage. The frontend uses no chart library.

Required runtime: Node.js 22.13 or newer. Use `npm ci` for the pinned dependency
tree. Dependencies and `.next` output are generated locally and ignored.
