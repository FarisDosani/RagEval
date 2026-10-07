import type { ExperimentResult, ComparisonResult, SweepResult, ExperimentFailureAnalysis, ExperimentRequest, ComparisonRequest, SweepRequest, DocumentInfo, QueryResponse, RetrievalStrategy } from '../types/api';
export async function request<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try { response = await fetch(`/backend${path}`, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? undefined : { 'Content-Type': body instanceof Blob ? 'application/octet-stream' : 'application/json' }, body: body === undefined ? undefined : body instanceof Blob ? body : JSON.stringify(body), cache: 'no-store' }); }
  catch { throw new Error('Backend unreachable. Check that the API is running and the frontend API base URL is configured.'); }
  if (!response.ok) {
    if (response.status === 503) throw new Error('Research service is not configured (503). Restart the backend with RAGEVAL_AUTO_INIT=true. Saved results can still be loaded.');
    if (response.status === 501) throw new Error('Chunk-size sweeps are not available through HTTP yet.');
    if (response.status === 404) throw new Error('No saved result found for this ID (404).');
    const data = await response.json().catch(() => null);
    throw new Error(`Request failed (${response.status}). ${typeof data?.detail === 'string' ? data.detail : 'Check the request and backend service.'}`);
  }
  return response.json() as Promise<T>;
}
const id = (value: string) => { const clean = value.trim(); if (!/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(clean)) throw new Error('Enter a valid UUID.'); return encodeURIComponent(clean); };
export const api = {
  health: () => request<{status:string}>('/health'),
  query: (question: string, top_k: number, retrieval_strategy: RetrievalStrategy = 'hybrid') => request<QueryResponse>('/query', {question, ...(retrieval_strategy==='llm_only'?{}:{top_k}), retrieval_strategy}),
  documents: () => request<DocumentInfo[]>('/documents'),
  upload: (file: File) => request<DocumentInfo>(`/documents/upload?filename=${encodeURIComponent(file.name)}`, file),
  experiment: (value: string) => request<ExperimentResult>(`/experiments/${id(value)}`),
  comparison: (value: string) => request<ComparisonResult>(`/comparisons/${id(value)}`),
  sweep: (value: string) => request<SweepResult>(`/sweeps/${id(value)}`),
  runExperiment: (body: ExperimentRequest) => request<ExperimentResult>('/experiments/run', body),
  runComparison: (body: ComparisonRequest) => request<ComparisonResult>('/comparisons/run', body),
  runSweep: (body: SweepRequest) => request<SweepResult>('/sweeps/run', body),
  analysis: (value: string) => request<ExperimentFailureAnalysis>('/failure-analysis', {experiment_id: decodeURIComponent(id(value))}),
};
