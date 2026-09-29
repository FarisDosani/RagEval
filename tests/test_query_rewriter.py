import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.generation import QueryRewriter
from app.generation import llm_service
from app.generation.llm_service import LLMConfig, LLMEmptyResponseError, LLMError, LLMService
from app.generation.prompts import build_user_prompt
from app.generation.query_rewriter import REWRITE_SYSTEM_PROMPT
from app.schemas.retrieval import RetrievalResult


@pytest.fixture(autouse=True)
def block_real_client(monkeypatch):
    constructor = Mock(side_effect=AssertionError("No real API client in rewrite tests"))
    monkeypatch.setattr(llm_service, "OpenAI", constructor)
    return constructor


def test_success_prompt_and_input_preservation(block_real_client):
    llm = Mock(spec=LLMService)
    llm.generate.return_value = "  solar panel energy conversion efficiency\n"
    rewriter = QueryRewriter(llm)
    original = "  How efficiently do solar panels convert energy?  "
    assert rewriter.rewrite_query(original) == "solar panel energy conversion efficiency"
    llm.generate.assert_called_once_with(REWRITE_SYSTEM_PROMPT, original)
    assert "Do not answer the question" in REWRITE_SYSTEM_PROMPT
    assert "Preserve the original intent and meaning" in REWRITE_SYSTEM_PROMPT
    assert "Do not add unsupported facts" in REWRITE_SYSTEM_PROMPT
    assert "no explanation" in REWRITE_SYSTEM_PROMPT
    assert "no prefix" in REWRITE_SYSTEM_PROMPT
    assert rewriter.llm is llm
    block_real_client.assert_not_called()


@pytest.mark.parametrize("output", ["", " \n\t"])
def test_blank_output_falls_back(output):
    llm = Mock(spec=LLMService)
    llm.generate.return_value = output
    assert QueryRewriter(llm).rewrite_query("  Original question?  ") == "Original question?"


def test_specific_empty_response_error_falls_back():
    llm = Mock(spec=LLMService)
    llm.generate.side_effect = LLMEmptyResponseError("empty text")
    assert QueryRewriter(llm).rewrite_query("Question?") == "Question?"


@pytest.mark.parametrize("query", ["", " \t\n", None, 42])
def test_invalid_input(query):
    llm = Mock(spec=LLMService)
    with pytest.raises(ValueError, match="query"):
        QueryRewriter(llm).rewrite_query(query)
    llm.generate.assert_not_called()


def test_provider_errors_propagate_and_stop_retrieval():
    llm = Mock(spec=LLMService)
    error = LLMError("provider failed")
    llm.generate.side_effect = error
    search = Mock()
    with pytest.raises(LLMError) as caught:
        QueryRewriter(llm).rewrite_and_retrieve("Question?", search)
    assert caught.value is error
    search.assert_not_called()


def test_reuses_injected_service_for_multiple_queries():
    llm = Mock(spec=LLMService)
    llm.generate.side_effect = ["first rewrite", "second rewrite"]
    rewriter = QueryRewriter(llm)
    assert rewriter.rewrite_query("First?") == "first rewrite"
    assert rewriter.rewrite_query("Second?") == "second rewrite"
    assert llm.generate.call_count == 2


def test_retrieval_uses_rewrite_generation_keeps_original():
    llm = Mock(spec=LLMService)
    llm.generate.return_value = "solar panel efficiency"
    chunks = [RetrievalResult(
        chunk_id="c1", document_id="d1", source="example.txt", page_number=None,
        chunk_index=0, text="Example context.", score=0.9,
    )]
    search = Mock(return_value=chunks)
    original = "How efficient are solar panels?"
    result = QueryRewriter(llm).rewrite_and_retrieve(original, search)
    search.assert_called_once_with("solar panel efficiency")
    assert result.original_query == original
    assert result.rewritten_query == "solar panel efficiency"
    assert result.retrieved_chunks == chunks
    generation_prompt = json.loads(build_user_prompt(result.original_query, result.retrieved_chunks))
    assert generation_prompt["question"] == original


def test_helper_empty_retrieval():
    llm = Mock(spec=LLMService)
    llm.generate.return_value = "rewritten"
    result = QueryRewriter(llm).rewrite_and_retrieve("original", Mock(return_value=[]))
    assert result.original_query == "original"
    assert result.rewritten_query == "rewritten"
    assert result.retrieved_chunks == []


@pytest.mark.parametrize("content", ["", " \n"])
def test_real_service_blank_response_contract_with_mock_client(monkeypatch, content):
    # Exercise the service boundary as well as the mock-return fallback path.
    monkeypatch.setattr(LLMConfig, "from_env", Mock(return_value=LLMConfig(
        "http://localhost:20128/v1", "fake-test-key", "test-model"
    )))
    constructor = Mock()
    constructor.return_value.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))]
    )
    monkeypatch.setattr(llm_service, "OpenAI", constructor)
    service = LLMService()
    rewriter = QueryRewriter(service)
    assert rewriter.rewrite_query("Original?") == "Original?"
    assert rewriter.rewrite_query("Another?") == "Another?"
    constructor.assert_called_once()
    assert constructor.return_value.chat.completions.create.call_args.kwargs["stream"] is False


def test_search_error_propagates():
    llm = Mock(spec=LLMService)
    llm.generate.return_value = "rewritten"
    with pytest.raises(RuntimeError, match="search failed"):
        QueryRewriter(llm).rewrite_and_retrieve(
            "original", Mock(side_effect=RuntimeError("search failed"))
        )
