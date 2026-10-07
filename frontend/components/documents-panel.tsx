'use client';
import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { DocumentInfo } from '../types/api';

export function DocumentsPanel({disabled, onBusyChange, onUploaded}: {
  disabled: boolean; onBusyChange: (busy: boolean) => void; onUploaded: () => void;
}) {
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const [file, setFile] = useState<File>();
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    api.documents().then(data => {if (active) setDocuments(data);})
      .catch(e => {if (active) setError(e instanceof Error ? e.message : 'Unable to load documents.');})
      .finally(() => {if (active) setLoading(false);});
    return () => {active = false;};
  }, []);
  async function upload() {
    if (!file) return;
    onBusyChange(true); setUploading(true); setError(''); setMessage('');
    try {
      const result = await api.upload(file);
      onUploaded();
      setMessage(`${result.name} indexed: ${result.extracted_unit_count} extracted units, ${result.chunk_count} chunks.`);
      setDocuments(await api.documents());
    } catch (e) {setError(e instanceof Error ? e.message : 'Upload failed.');}
    finally {setUploading(false); onBusyChange(false);}
  }
  return <section className="panel" aria-busy={uploading}>
    <div className="section-heading"><h2>Documents / Corpus</h2><span className="tag">CURRENT SESSION</span></div>
    <p className="hint">Upload PDF, UTF-8 TXT, or DOCX. The default upload limit is 10 MiB. New documents are added to the current corpus.</p>
    <form onSubmit={e => {e.preventDefault(); void upload();}}>
      <label>Document<input type="file" accept=".pdf,.txt,.docx" required disabled={disabled}
        onChange={e => setFile(e.target.files?.[0])}/></label>
      <button disabled={disabled || !file}>{uploading ? 'Uploading and indexing…' : 'Upload document'}</button>
    </form>
    {uploading && <p role="status" className="notice">Uploading and rebuilding indexes. Please wait…</p>}
    {message && <p role="status" className="notice">{message}</p>}
    {error && <p role="alert" className="error">{error}</p>}
    <h3>Indexed documents</h3>
    {loading ? <p>Loading corpus…</p> : documents.length ? <ul>{documents.map(doc =>
      <li key={doc.document_id}><strong>{doc.name}</strong> · {doc.extracted_unit_count} units · {doc.chunk_count} chunks · {doc.index_status}</li>
    )}</ul> : <p>No indexed documents available.</p>}
    <p className="hint">The session resets on backend restart. Research benchmarks should use their original corpus and relevance IDs.</p>
  </section>;
}
