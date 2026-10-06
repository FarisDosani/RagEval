import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.experiments import phase25_resume as module
from app.experiments.phase25 import save
from app.generation.llm_service import LLMError
from app.schemas.generation import LLMGenerationResult


def service():
    model=Mock()
    model.model='gemini-3.5-flash-lite'
    model.generate_result.return_value=LLMGenerationResult(text='ok',model=model.model,latency_ms=10,prompt_tokens=2,completion_tokens=1,total_tokens=3,cost=None)
    return model


def test_pacing_and_replay_dont_add_usage(tmp_path,monkeypatch):
    now=[100.]
    def sleep(seconds): now[0]+=seconds
    monkeypatch.setattr(module,'time',SimpleNamespace(time=lambda:now[0],perf_counter=lambda:now[0],sleep=sleep))
    backend=service()
    llm=module.PacedRecordedLLM(backend,tmp_path/'run/calls',tmp_path,tmp_path/'pace.json')
    first=llm.generate_result('system','first')
    llm.generate_result('system','second')
    assert now[0]==105.5
    assert llm.wait_ms==5500.
    assert llm.generate_result('system','first')==first
    assert backend.generate_result.call_count==2
    assert llm.usage()['successful_calls']==2
    assert llm.replayed_calls==1
    resumed=module.PacedRecordedLLM(backend,tmp_path/'next/calls',tmp_path,tmp_path/'pace.json')
    resumed.generate_result('system','first')
    assert backend.generate_result.call_count==2
    resumed.context={'question_id':'different'}
    resumed.generate_result('system','first')
    assert backend.generate_result.call_count==3


def test_429_preserves_cooldown_and_stops_without_retry(tmp_path,monkeypatch):
    monkeypatch.setattr(module,'time',SimpleNamespace(time=lambda:100.,perf_counter=lambda:100.,sleep=Mock()))
    backend=service()
    error=LLMError('quota',diagnostics={'http_status':429,'provider_error':{'message':'Please retry in 49s.'}})
    backend.generate_result.side_effect=error
    llm=module.PacedRecordedLLM(backend,tmp_path/'run/calls',tmp_path,tmp_path/'pace.json')
    with pytest.raises(LLMError) as caught:
        llm.generate_result('system','user')
    assert caught.value is error
    backend.generate_result.assert_called_once()
    state=json.loads((tmp_path/'pace.json').read_text())
    assert state['next_request_at']==150.
    assert llm.usage()['successful_calls']==0
    assert llm.records[0]['diagnostics']['http_status']==429


def test_failed_response_never_replayed(tmp_path):
    save(tmp_path/'old/calls/failure.json',{'success':False})
    llm=module.PacedRecordedLLM(service(),tmp_path/'new/calls',tmp_path,tmp_path/'pace.json')
    llm.generate_result('system','user')
    assert llm.replayed_calls==0
