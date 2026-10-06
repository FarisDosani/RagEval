from unittest.mock import Mock
import json
import pytest
from app.evaluation.answer_correctness import AnswerCorrectnessEvaluator, CorrectnessParsingError, CORRECTNESS_PROMPT
from app.evaluation.groundedness import GROUNDEDNESS_PROMPT
from app.evaluation.citation_accuracy import CITATION_SUPPORT_PROMPT
from app.evaluation.hallucination_refusal import HALLUCINATION_PROMPT, HallucinationRefusalEvaluator, HallucinationParsingError
from app.schemas.retrieval import RetrievalResult


@pytest.mark.parametrize('prompt',[CORRECTNESS_PROMPT,GROUNDEDNESS_PROMPT,CITATION_SUPPORT_PROMPT,HALLUCINATION_PROMPT])
def test_exact_contract_guidance(prompt):
    assert 'only permitted keys' in prompt
    assert 'reason MUST be a nonempty' in prompt
    assert 'No Markdown' in prompt


def test_correctness_still_rejects_extra_field():
    llm=Mock();llm.generate.return_value=json.dumps({'score':1.0,'label':'correct','reason':'Matches.', 'uses_outside_knowledge':False})
    with pytest.raises(CorrectnessParsingError):AnswerCorrectnessEvaluator(llm).evaluate('q','truth','answer')


@pytest.mark.parametrize('reason',['',' ','Supported by the evidence.'])
def test_hallucination_nonempty_reason(reason):
    llm=Mock();llm.generate.return_value=json.dumps({'hallucinated':False,'reason':reason,'unsupported_claims':[]})
    evidence=[RetrievalResult(chunk_id='c',document_id='d',source='s',page_number=1,chunk_index=0,text='Evidence',score=1)]
    evaluator=HallucinationRefusalEvaluator(llm)
    if reason.strip():assert evaluator.evaluate(True,'Answer',evidence).reason==reason
    else:
        with pytest.raises(HallucinationParsingError):evaluator.evaluate(True,'Answer',evidence)
