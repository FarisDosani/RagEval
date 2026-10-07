import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import test from 'node:test';
import assert from 'node:assert/strict';
import ts from 'typescript';
import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';

// Compile the real TS/TSX components inside this test process; no extra test framework.
const loadModule = createRequire(import.meta.url);
for (const extension of ['.ts', '.tsx']) {
  loadModule.extensions[extension] = (module, filename) => {
    const { outputText } = ts.transpileModule(readFileSync(filename, 'utf8'), {
      compilerOptions: { module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022 },
    });
    module._compile(outputText, filename);
  };
}
const { Workspace } = loadModule('../components/workspace.tsx');
const { Answer } = loadModule('../components/results.tsx');
const { api } = loadModule('../lib/api.ts');

test('query and experiment forms offer LLM Only without removing existing strategies', () => {
  for (const mode of ['query', 'experiments']) {
    const html = renderToStaticMarkup(React.createElement(Workspace, { mode }));
    assert.match(html, /value="llm_only">LLM Only<\/option>/);
    for (const strategy of ['dense', 'bm25', 'hybrid']) assert.ok(html.includes(`value="${strategy}"`));
  }
});

test('baseline answer renders no retrieval and preserves usage display', () => {
  const html = renderToStaticMarkup(React.createElement(Answer, { result: {
    question: 'Q', answer: 'Direct answer', model: 'test-model', retrieval_strategy: 'llm_only',
    citations: [], retrieved_chunks: [], latency_ms: 12, usage: { total_tokens: 14, cost: null },
  } }));
  assert.match(html, /No retrieval used/);
  assert.match(html, /Direct answer/);
  assert.match(html, /14/);
  assert.match(html, /Not available/);
});

test('baseline request omits ignored top-k; RAG requests preserve it', async () => {
  const original = globalThis.fetch;
  const bodies = [];
  globalThis.fetch = async (url, options) => {
    assert.equal(url, '/backend/query');
    bodies.push(JSON.parse(options.body));
    return { ok: true, json: async () => ({}) };
  };
  try {
    await api.query('Q', Number.NaN, 'llm_only');
    await api.query('Q', 3, 'dense');
    assert.deepEqual(bodies, [{ question: 'Q', retrieval_strategy: 'llm_only' },
      { question: 'Q', top_k: 3, retrieval_strategy: 'dense' }]);
  } finally { globalThis.fetch = original; }
});
