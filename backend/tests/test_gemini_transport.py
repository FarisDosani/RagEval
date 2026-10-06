import json
from unittest.mock import Mock

import httpx
import pytest

from app.generation import llm_service, gemini_transport
from app.generation.llm_service import LLMService, LLMConfig, LLMError


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("GEMINI_API_KEY", "test-secret")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
    monkeypatch.setattr(llm_service, "dotenv_values", Mock(return_value={}))
    monkeypatch.setattr(llm_service, "OpenAI", Mock(side_effect=AssertionError("Wrong provider")))
    constructor = Mock()
    constructor.return_value.post.return_value.json.return_value = {
        "candidates": [{"finishReason":"STOP", "content":{"parts":[
            {"thought":True,"text":"hidden"},{"text":"OK"}]}}],
        "usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1,"totalTokenCount":10},
        "modelVersion":"reported-version"}
    monkeypatch.setattr(gemini_transport.httpx, "Client", constructor)
    return constructor


def test_native_mapping_usage_and_reuse(client, monkeypatch):
    monkeypatch.setattr(gemini_transport, "perf_counter", Mock(side_effect=[1,1.1,2,2.1]))
    service=LLMService()
    result=service.generate_result("instructions","question")
    assert result.text=="OK" and result.model=="reported-version"
    assert result.latency_ms==pytest.approx(100)
    assert (result.prompt_tokens,result.completion_tokens,result.total_tokens,result.cost)==(5,1,10,None)
    assert service.generate("instructions","question")=="OK"
    client.assert_called_once()
    client.return_value.post.assert_called_with("models/gemini-3.5-flash-lite:generateContent",json={
        "systemInstruction":{"parts":[{"text":"instructions"}]},
        "contents":[{"role":"user","parts":[{"text":"question"}]}]})
    service.close();client.return_value.close.assert_called_once()


def test_missing_usage(client):
    data=client.return_value.post.return_value.json.return_value
    del data['usageMetadata'];del data['modelVersion']
    result=LLMService().generate_result('s','u')
    assert result.model=='gemini-3.5-flash-lite'
    assert result.prompt_tokens is result.completion_tokens is result.total_tokens is result.cost is None


@pytest.mark.parametrize('data',[None,{}, {'candidates':[]}, {'candidates':[{'finishReason':'MAX_TOKENS'}]},
    {'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':' '}]}}]},
    {'candidates':[{'finishReason':'STOP','content':{'parts':[{'thought':True,'text':'hidden'}]}}]}])
def test_invalid_response(client,data):
    client.return_value.post.return_value.json.return_value=data
    with pytest.raises(LLMError):LLMService().generate('s','u')


def test_http_error_details_and_redaction(client):
    request=httpx.Request('POST','https://generativelanguage.googleapis.com/v1beta/models/test:generateContent')
    response=httpx.Response(429,request=request,json={'error':{'code':429,'status':'RESOURCE_EXHAUSTED',
        'message':'Quota test-secret Bearer hidden-token','headers':{'authorization':'secret'}}})
    error=httpx.HTTPStatusError('provider error',request=request,response=response)
    client.return_value.post.return_value.raise_for_status.side_effect=error
    with pytest.raises(LLMError) as caught:LLMService().generate('s','u')
    assert caught.value.__cause__ is error
    assert caught.value.diagnostics['http_status']==429
    encoded=json.dumps(caught.value.diagnostics)
    assert 'test-secret' not in encoded and 'hidden-token' not in encoded and 'authorization' not in encoded
    client.return_value.post.assert_called_once()


@pytest.mark.parametrize('name',['GEMINI_API_KEY','GEMINI_MODEL'])
def test_required_gemini_config(client,monkeypatch,name):
    monkeypatch.delenv(name)
    with pytest.raises(ValueError,match=name):LLMService()
    client.assert_not_called()


def test_config_secret_repr(client):
    assert 'test-secret' not in repr(LLMConfig.from_env())
