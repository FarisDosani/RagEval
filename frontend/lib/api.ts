import type { ExperimentResult, ComparisonResult, SweepResult, ExperimentFailureAnalysis, RAGResponse, ExperimentRequest, ComparisonRequest, SweepRequest } from '../types/api';
export async function request<T>(path: string, body?: unknown): Promise<T> {
  let response: Response;
  try { response = await fetch(`/backend${path}`, { method: body === undefined ? 'GET' : 'POST', headers: body === undefined ? undefined : { 'Content-Type': 'application/json' }, body: body === undefined ? undefined : JSON.stringify(body), cache: 'no-store' }); }
  catch { throw new Error('Backend unreachable. Check that the API is running and the frontend API base URL is configured.'); }
  if (!response.ok) {
    if (response.status === 503) throw new Error('Research service is not configured (503). The backend needs prebuilt runtime services. Saved results can still be loaded.');
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
  query: (question: string, top_k: number) => request<RAGResponse>('/query', {question, top_k}),
  experiment: (value: string) => request<ExperimentResult>(`/experiments/${id(value)}`),
  comparison: (value: string) => request<ComparisonResult>(`/comparisons/${id(value)}`),
  sweep: (value: string) => request<SweepResult>(`/sweeps/${id(value)}`),
  runExperiment: (body: ExperimentRequest) => request<ExperimentResult>('/experiments/run', body),
  runComparison: (body: ComparisonRequest) => request<ComparisonResult>('/comparisons/run', body),
  runSweep: (body: SweepRequest) => request<SweepResult>('/sweeps/run', body),
  analysis: (value: string) => request<ExperimentFailureAnalysis>('/failure-analysis', {experiment_id: decodeURIComponent(id(value))}),
};
