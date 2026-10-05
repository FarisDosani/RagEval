import json
from unittest.mock import Mock

import pytest

from app.experiments.phase25 import DATA
from app.experiments.phase25_benchmark import build_benchmark, validate_benchmark
from app.experiments.phase25_execute import TracedComponent, SMOKE, SUBSET
from app.schemas.chunk import DocumentChunk


def corpus():
    return [DocumentChunk.model_validate(c) for c in json.loads((DATA/'chunks_512.json').read_text(encoding='utf-8'))]


def test_fresh_benchmark_references_actual_source_chunks():
    chunks=corpus()
    items=build_benchmark(chunks)
    assert len(items)==30
    assert sum(i.answerable for i in items)==24
    assert {i.category for i in items}=={'factual','definition','comparison','multi_hop','reasoning','unanswerable'}
    assert {i.difficulty for i in items}=={'easy','medium','hard'}
    assert len({i.question for i in items})==30
    assert all(i.ground_truth for i in items)
    assert all(not i.relevant_chunk_ids for i in items if not i.answerable)
    assert any(len(i.relevant_chunk_ids)>1 for i in items)
    by_id={c.chunk_id:c for c in chunks}
    assert all(by_id[id].source=='Week # 02 Slides.pdf' for i in items for id in i.relevant_chunk_ids)
    assert len(SMOKE)==3 and len(SUBSET)==10 and set(SMOKE)<=set(SUBSET)
    assert any(not items[n-1].answerable for n in SMOKE)


def test_unknown_relevance_id_rejected():
    chunks=corpus();items=build_benchmark(chunks)
    items[0]=items[0].model_copy(update={'relevant_chunk_ids':['stale-id']})
    with pytest.raises(ValueError,match='Unknown relevance'):
        validate_benchmark(items,chunks)


def test_split_evidence_page_requires_review():
    chunks=corpus();page3=next(c for c in chunks if c.page_number==3)
    with pytest.raises(ValueError,match='split'):
        build_benchmark(chunks+[page3.model_copy(update={'chunk_id':'extra'})])


def test_trace_preserves_exact_error_and_question_context(tmp_path):
    component=Mock();error=ValueError('strict output rejected')
    component.evaluate.side_effect=error
    llm=Mock();llm.context={'stage':'smoke'}
    traced=TracedComponent(component,'groundedness',llm,{'question':'q1'},tmp_path)
    with pytest.raises(ValueError) as caught:
        traced.evaluate('question','answer',[])
    assert caught.value is error
    assert llm.context=={'stage':'smoke','question_id':'q1','evaluator':'groundedness'}
    component.evaluate.assert_called_once_with('question','answer',[])
