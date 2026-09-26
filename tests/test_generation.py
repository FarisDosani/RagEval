import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dotenv import dotenv_values
from openai import OpenAIError

from app.embeddings import embedding_service
from app.generation import LLMService, RAGPipeline
from app.generation import llm_service
from app.generation.llm_service import LLMConfig, LLMError
from app.generation.prompts import INSUFFICIENT_INFORMATION, SYSTEM_PROMPT
from app.schemas.generation import RAGResponse
from app.schemas.retrieval import RetrievalResult

CONFIG = {
    "LLM_BASE_URL": "http://localhost:20128/v1",
    "LLM_API_KEY": "test-key-not-real",
    "LLM_MODEL": "kr/claude-sonnet-4.5",
}


def completion(content="Supported answer.", finish_reason="stop"):
    return SimpleNamespace(choices=[SimpleNamespace(
        message=SimpleNamespace(content=content), finish_reason=finish_reason
    )])


@pytest.fixture(autouse=True)
def external_boundaries(monkeypatch):
    for name, value in CONFIG.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(llm_service, "dotenv_values", Mock(return_value={}))
    constructor = Mock()
    constructor.return_value.chat.completions.create.return_value = completion()
    monkeypatch.setattr(llm_service, "OpenAI", constructor)
    monkeypatch.setattr(embedding_service, "_load_model", Mock(
        side_effect=AssertionError("Tests must not load an embedding model")
    ))
    return constructor


def retrieved(index=0, page=2):
    return RetrievalResult(
        chunk_id=f"doc_chunk_{index}", document_id="doc", source="notes.pdf",
        page_number=page, chunk_index=index, text=f"Evidence {index}.", score=0.9,
    )


def test_environment_config():
    config = LLMConfig.from_env()
    assert config.base_url == CONFIG["LLM_BASE_URL"]
    assert config.model == CONFIG["LLM_MODEL"]
    assert config.api_key == CONFIG["LLM_API_KEY"]
    assert config.api_key not in repr(config)


def test_dotenv_loading_and_environment_precedence(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("\n".join(f"{name}={value}" for name, value in CONFIG.items()), encoding="utf-8")
    monkeypatch.setattr(llm_service, "dotenv_values", dotenv_values)
    for name in CONFIG:
        monkeypatch.delenv(name)
    assert LLMConfig.from_env(path).model == CONFIG["LLM_MODEL"]
    assert all(name not in os.environ for name in CONFIG)
    monkeypatch.setenv("LLM_MODEL", "environment-model")
    assert LLMConfig.from_env(path).model == "environment-model"


@pytest.mark.parametrize("name", list(CONFIG))
@pytest.mark.parametrize("value", [None, "  "])
def test_missing_configuration(name, value, monkeypatch, external_boundaries):
    if value is None:
        monkeypatch.delenv(name)
    else:
        monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        LLMService()
    external_boundaries.assert_not_called()


def test_client_creation_reuse_and_nonstreaming(external_boundaries):
    service = LLMService()
    external_boundaries.assert_called_once_with(
        base_url=CONFIG["LLM_BASE_URL"], api_key=CONFIG["LLM_API_KEY"],
        timeout=60.0, max_retries=0,
    )
    for _ in range(2):
        assert service.generate("system", "user") == "Supported answer."
    external_boundaries.assert_called_once()
    create = external_boundaries.return_value.chat.completions.create
    assert create.call_count == 2
    create.assert_called_with(
        model=CONFIG["LLM_MODEL"], stream=False,
        messages=[{"role": "system", "content": "system"}, {"role": "user", "content": "user"}],
    )
    service.close()
    external_boundaries.return_value.close.assert_called_once()


def test_provider_errors_are_chained(external_boundaries):
    error = OpenAIError("Gateway failure")
    external_boundaries.return_value.chat.completions.create.side_effect = error
    with pytest.raises(LLMError, match="request failed") as caught:
        LLMService().generate("system", "user")
    assert caught.value.__cause__ is error


@pytest.mark.parametrize("response", [
    SimpleNamespace(choices=[]), completion(None), completion("  "),
    completion("Truncated", "length"),
])
def test_invalid_provider_responses(external_boundaries, response):
    external_boundaries.return_value.chat.completions.create.return_value = response
    with pytest.raises(LLMError):
        LLMService().generate("system", "user")


def test_pipeline_response_context_citations_and_top_k(external_boundaries):
    store = Mock()
    chunks = [retrieved(), retrieved(1, None)]
    store.search_text.return_value = chunks
    embeddings = Mock()
    pipeline = RAGPipeline(store, embedding_service=embeddings)

    response = pipeline.ask("What is supported?", top_k=2)

    assert isinstance(response, RAGResponse)
    assert response.question == "What is supported?"
    assert response.answer == "Supported answer."
    assert response.model == CONFIG["LLM_MODEL"]
    assert response.latency_ms >= 0
    assert response.retrieved_chunks == chunks
    assert [citation.model_dump() for citation in response.citations] == [
        {"chunk_id": "doc_chunk_0", "source": "notes.pdf", "page_number": 2},
        {"chunk_id": "doc_chunk_1", "source": "notes.pdf", "page_number": None},
    ]
    store.search_text.assert_called_once_with("What is supported?", top_k=2, service=embeddings)
    messages = external_boundaries.return_value.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert INSUFFICIENT_INFORMATION in SYSTEM_PROMPT
    assert "only from" in SYSTEM_PROMPT
    payload = json.loads(messages[1]["content"])
    assert payload["question"] == response.question
    assert payload["retrieved_context"] == [
        {"chunk_id": item.chunk_id, "source": item.source, "page_number": item.page_number, "text": item.text}
        for item in chunks
    ]


def test_no_context_does_not_call_provider(external_boundaries):
    store = Mock()
    store.search_text.return_value = []
    response = RAGPipeline(store).ask("Question")
    assert response.answer == INSUFFICIENT_INFORMATION
    assert response.retrieved_chunks == []
    assert response.citations == []
    assert response.model == CONFIG["LLM_MODEL"]
    assert response.latency_ms >= 0
    external_boundaries.return_value.chat.completions.create.assert_not_called()


def test_insufficient_answer_has_no_citations(external_boundaries):
    store = Mock()
    store.search_text.return_value = [retrieved()]
    external_boundaries.return_value.chat.completions.create.return_value = completion(INSUFFICIENT_INFORMATION)
    response = RAGPipeline(store).ask("Question")
    assert response.answer == INSUFFICIENT_INFORMATION
    assert response.citations == []
    assert response.retrieved_chunks == [retrieved()]


@pytest.mark.parametrize("question", ["", " \n\t", None, 123])
def test_invalid_question(question, external_boundaries):
    store = Mock()
    with pytest.raises(ValueError, match="question"):
        RAGPipeline(store).ask(question)
    store.search_text.assert_not_called()
    external_boundaries.return_value.chat.completions.create.assert_not_called()


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5, "2"])
def test_invalid_top_k(top_k, external_boundaries):
    store = Mock()
    with pytest.raises(ValueError, match="top_k"):
        RAGPipeline(store).ask("Question", top_k)
    store.search_text.assert_not_called()
    external_boundaries.return_value.chat.completions.create.assert_not_called()


def test_pipeline_propagates_llm_errors():
    store = Mock()
    store.search_text.return_value = [retrieved()]
    llm = Mock()
    llm.generate.side_effect = LLMError("request failed")
    with pytest.raises(LLMError, match="request failed"):
        RAGPipeline(store, llm_service=llm).ask("Question")
