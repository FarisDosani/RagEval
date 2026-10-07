from io import BytesIO
from unittest.mock import Mock

import pytest
from docx import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject
from fastapi.testclient import TestClient
from app import runtime
from app.api import documents
from app.main import create_app
from app.schemas.embedding import EmbeddedChunk
from app.schemas.generation import LLMGenerationResult


@pytest.fixture
def corpus_api(tmp_path, monkeypatch):
    path = tmp_path / 'original.txt'
    path.write_text('The original document describes power and energy.')
    monkeypatch.setenv('RAGEVAL_CORPUS_PATH', str(path))
    monkeypatch.setattr(documents, 'UPLOAD_ROOT', tmp_path / 'uploads')
    llm = Mock(model='test-model', provider='gemini')
    llm.generate_result.return_value = LLMGenerationResult(text='Answer.', model='test-model')
    monkeypatch.setattr(runtime, 'LLMService', Mock(return_value=llm))
    embed = Mock()
    embed.embed_chunks.side_effect = lambda chunks: [EmbeddedChunk(**c.model_dump(), embedding=[1., 0.]) for c in chunks]
    embed.embed_text.return_value = [1., 0.]
    constructor = Mock(return_value=embed)
    monkeypatch.setattr(runtime, 'EmbeddingService', constructor)
    app = create_app(auto_init=True)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, app, embed, constructor, tmp_path


def file_bytes(extension):
    if extension == '.txt':
        return b'New document about reliability and fault tolerance.'
    stream = BytesIO()
    if extension == '.docx':
        doc = Document(); doc.add_paragraph('New document about reliability.'); doc.save(stream)
    else:
        writer = PdfWriter(); page = writer.add_blank_page(width=300, height=300)
        font = DictionaryObject({NameObject('/Type'): NameObject('/Font'), NameObject('/Subtype'): NameObject('/Type1'), NameObject('/BaseFont'): NameObject('/Helvetica')})
        page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
        content = DecodedStreamObject(); content.set_data(b'BT /F1 12 Tf 10 100 Td (Reliability is important.) Tj ET')
        page[NameObject('/Contents')] = writer._add_object(content)
        writer.write(stream)
    return stream.getvalue()


@pytest.mark.parametrize('extension', ['.txt', '.pdf', '.docx'])
def test_upload_builds_all_indexes_and_lists(corpus_api, extension):
    client, app, embed, constructor, root = corpus_api
    old = app.state.runtime.snapshot
    response = client.post('/documents/upload', params={'filename': 'new' + extension}, content=file_bytes(extension))
    assert response.status_code == 200, response.text
    info = response.json()
    assert info['name'] == 'new' + extension
    assert info['extracted_unit_count'] == info['chunk_count'] == 1
    assert info['index_status'] == 'ready'
    assert len(client.get('/documents').json()) == 2
    new = app.state.runtime.snapshot
    assert new is not old
    assert len(old.documents) == 1  # In-flight readers retain an intact snapshot.
    assert new.experiment_runner.generator is new.pipeline
    assert new.pipeline.llm is old.pipeline.llm
    for name in ('dense', 'bm25', 'hybrid'):
        assert new.experiment_runner.retrievers[name] is not old.experiment_runner.retrievers[name]
        result = client.post('/query', json={'question': 'reliability', 'retrieval_strategy': name, 'top_k': 10})
        assert result.status_code == 200, result.text
        assert info['document_id'] in {c['document_id'] for c in result.json()['retrieved_chunks']}
        assert result.json()['retrieval_strategy'] == name
    constructor.assert_called_once()
    assert embed.embed_chunks.call_count == 2  # No query rebuilds.
    assert len(list((root / 'uploads').glob('*/*'))) == 1


@pytest.mark.parametrize('filename,body,status', [
    ('bad.exe', b'content', 400), ('empty.txt', b'', 400), ('blank.txt', b'  ', 400),
    ('../escape.txt', b'data', 400), ('..\\escape.txt', b'data', 400),
    ('C:\\secret.txt', b'data', 400), ('CON.txt', b'data', 400),
    ('broken.pdf', b'not a pdf', 400), ('broken.docx', b'not docx', 400),
])
def test_rejected_upload_keeps_corpus(corpus_api, filename, body, status):
    client, app, _, _, root = corpus_api
    before = app.state.runtime.snapshot
    response = client.post('/documents/upload', params={'filename': filename}, content=body)
    assert response.status_code == status
    assert app.state.runtime.snapshot is before
    assert not list((root / 'uploads').glob('*/*'))


def test_limit_and_failed_embedding_are_atomic(corpus_api, monkeypatch):
    client, app, embed, _, root = corpus_api
    monkeypatch.setenv('RAGEVAL_MAX_UPLOAD_BYTES', '4')
    before = app.state.runtime.snapshot
    assert client.post('/documents/upload?filename=big.txt', content=b'12345').status_code == 413
    monkeypatch.setenv('RAGEVAL_MAX_UPLOAD_BYTES', '100')
    embed.embed_chunks.side_effect = RuntimeError('private failure')
    response = client.post('/documents/upload?filename=notes.txt', content=b'new text')
    assert response.status_code == 500
    assert 'private failure' not in response.text
    assert app.state.runtime.snapshot is before
    assert not list((root / 'uploads').glob('*/*'))


def test_duplicate_is_idempotent(corpus_api):
    client, app, embed, _, _ = corpus_api
    a = client.post('/documents/upload?filename=notes.txt', content=b'new text').json()
    before = app.state.runtime.snapshot
    b = client.post('/documents/upload?filename=notes.txt', content=b'new text').json()
    assert a == b
    assert app.state.runtime.snapshot is before
    assert embed.embed_chunks.call_count == 2


@pytest.mark.parametrize('strategy', ['dense', 'bm25', 'hybrid', None])
def test_strategy_and_top_k_dispatch(corpus_api, strategy):
    client, app, _, _, _ = corpus_api
    selected = strategy or 'hybrid'
    retriever = Mock(**{'search.return_value': [], 'search_text.return_value': []})
    app.state.runtime.experiment_runner.retrievers[selected] = retriever
    body = {'question': 'power', 'top_k': 7}
    if strategy is not None: body['retrieval_strategy'] = strategy
    response = client.post('/query', json=body)
    assert response.status_code == 200
    assert response.json()['retrieval_strategy'] == selected
    if selected == 'dense': retriever.search_text.assert_called_once_with('power', top_k=7)
    elif selected == 'bm25': retriever.search.assert_called_once_with('power', top_k=7)
    else: retriever.search.assert_called_once_with('power', candidate_k=20, final_top_k=7)


def test_invalid_strategy(corpus_api):
    assert corpus_api[0].post('/query', json={'question': 'power', 'retrieval_strategy': 'other'}).status_code == 400
