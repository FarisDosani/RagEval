from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from app import runtime
from app.main import create_app
from app.schemas.embedding import EmbeddedChunk
from app.schemas.generation import LLMGenerationResult


@pytest.fixture
def local_runtime(tmp_path, monkeypatch):
    corpus = tmp_path / "notes.txt"
    corpus.write_text("Power is energy per unit time.", encoding="utf-8")
    monkeypatch.setenv("RAGEVAL_CORPUS_PATH", str(corpus))
    llm = Mock(model="test-model", provider="gemini")
    llm.generate_result.return_value = LLMGenerationResult(text="Power is energy per unit time.", model="test-model", latency_ms=1)
    monkeypatch.setattr(runtime, "LLMService", Mock(return_value=llm))
    embeddings = Mock()
    embeddings.embed_chunks.side_effect = lambda chunks: [EmbeddedChunk(**c.model_dump(), embedding=[1., 0.]) for c in chunks]
    embeddings.embed_text.return_value = [1., 0.]
    monkeypatch.setattr(runtime, "EmbeddingService", Mock(return_value=embeddings))
    return llm, embeddings


def test_bootstrap_query_and_experiments_once(local_runtime, tmp_path):
    llm, embeddings = local_runtime
    app = create_app(auto_init=True, results_dir=tmp_path / "results")
    with TestClient(app) as client:
        assert client.get('/health').status_code == 200
        for _ in range(2):
            response = client.post('/query', json={'question': 'What is power?'})
            assert response.status_code == 200
            assert response.json()['answer'] == 'Power is energy per unit time.'
        assert client.post('/experiments/run', json={'benchmark': [], 'config': {
            'experiment_name': 'runtime-test', 'benchmark_name': 'empty',
            'retrieval_strategy': 'hybrid', 'model': 'test-model'}}).status_code == 200
        embeddings.embed_chunks.assert_called_once()
        assert set(app.state.experiment_runner.retrievers) == {'dense', 'bm25', 'hybrid'}
        app.state.experiment_runner.retrievers['dense'].search_text('power')
        assert embeddings.embed_text.call_count == 3
    llm.close.assert_called_once()


def test_disabled_and_injected_never_build(monkeypatch):
    builder = Mock(side_effect=AssertionError('must not build'))
    monkeypatch.setattr(runtime, 'build_runtime', builder)
    monkeypatch.setenv('RAGEVAL_AUTO_INIT', 'true')
    with TestClient(create_app()) as client:
        assert client.get('/health').status_code == 200
        assert client.post('/query', json={'question': 'Power?'}).status_code == 503
    with TestClient(create_app(auto_init=True, pipeline=Mock())) as client:
        assert client.get('/health').status_code == 200
    builder.assert_not_called()


def test_environment_opt_in(local_runtime, monkeypatch):
    monkeypatch.setenv('RAGEVAL_AUTO_INIT', 'true')
    with TestClient(create_app(auto_init=None)) as client:
        assert client.post('/query', json={'question': 'Power?'}).status_code == 200


def test_missing_corpus_fails_without_provider(monkeypatch, tmp_path):
    monkeypatch.setenv('RAGEVAL_CORPUS_PATH', str(tmp_path/'missing.pdf'))
    llm = Mock()
    monkeypatch.setattr(runtime, 'LLMService', llm)
    with pytest.raises(ValueError, match='existing document'):
        runtime.build_runtime()
    llm.assert_not_called()


def test_failed_build_closes_client(local_runtime):
    llm, embeddings = local_runtime
    embeddings.embed_chunks.side_effect = RuntimeError('embedding failed')
    with pytest.raises(RuntimeError, match='embedding failed'):
        runtime.build_runtime()
    llm.close.assert_called_once()
