"""Fresh Phase 25 artifacts and gated native Gemini compatibility checks.

No live calls at import. Request/response records allow exact diagnostic replay.
No retry or judge-output repair is performed.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from pypdf import PdfReader
from app.ingestion.loaders import load_document
from app.ingestion.chunker import chunk_document
from app.generation.llm_service import LLMService, LLMConfig
from app.generation.rag_pipeline import RAGPipeline
from app.generation.query_rewriter import QueryRewriter
from app.schemas.retrieval import RetrievalResult
from app.evaluation.answer_correctness import AnswerCorrectnessEvaluator
from app.evaluation.groundedness import GroundednessEvaluator
from app.evaluation.hallucination_refusal import HallucinationRefusalEvaluator

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'data/corpus/Week # 02 Slides.pdf'
DATA = ROOT / 'data/processed/phase25'
SUMMARY = ROOT / 'results/final/ca_small_gemini_final_summary.json'
USAGE_LEDGER = ROOT / 'results/final/usage_ledger.json'


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def cumulative_usage(ledger_root, archived_ledger):
    """Combine compact historical accounting and live records, once per call ID.

    Finished request bodies may be pruned after their accounting is archived.
    Active request/response records remain available for interrupted-call replay.
    """
    records = {}
    if archived_ledger.exists():
        archive = json.loads(archived_ledger.read_text(encoding='utf-8'))
        records.update({record['id']: record for record in archive['records']})
    for path in ledger_root.glob('*/calls/*.json'):
        record = json.loads(path.read_text(encoding='utf-8'))
        records[record['id']] = record
    successful = [r['response'] for r in records.values() if r['success']]
    usage = {'attempted_calls':len(records), 'successful_calls':len(successful)}
    for field in ('prompt_tokens','completion_tokens','total_tokens','cost'):
        values = [r.get(field) for r in successful]
        usage[field] = sum(values) if values and all(v is not None for v in values) else None
    return usage


def ingest():
    digest=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    units=[u.model_copy(update={'document_id':digest}) for u in load_document(SOURCE)]
    chunks=chunk_document([u.model_copy(update={'document_id':digest+'_s512_o50'}) for u in units],512,50)
    manifest={'source':SOURCE.name,'corpus_sha256':digest,'document_count':1,
        'page_count':len(PdfReader(SOURCE).pages),'nonempty_units':len(units),
        'word_count':sum(len(u.text.split()) for u in units),'chunk_count':len(chunks),
        'chunk_size':512,'chunk_overlap':50}
    for path,data in [('units.json',[u.model_dump() for u in units]),('chunks_512.json',[c.model_dump() for c in chunks]),('manifest.json',manifest)]:
        target=DATA/path
        if target.exists():raise FileExistsError('Fresh ingestion would overwrite an existing artifact: '+str(target))
        save(target,data)
    return chunks,manifest


class RecordedLLM:
    def __init__(self, service, directory):
        self.service=service;self.model=service.model;self.directory=directory
        self.context={};self.records=[]

    def generate_result(self, system_prompt, user_prompt):
        record={'id':str(uuid4()),'timestamp':datetime.now(timezone.utc).isoformat(),
                'provider':'gemini','model':self.model,**self.context,
                'system_prompt':system_prompt,'user_prompt':user_prompt}
        started=perf_counter()
        try:
            result=self.service.generate_result(system_prompt,user_prompt)
            record.update(success=True,response=result.model_dump(mode='json'))
            return result
        except Exception as exc:
            record.update(success=False,error_type=type(exc).__name__,diagnostics=getattr(exc,'diagnostics',{}))
            raise
        finally:
            record['wall_clock_ms']=(perf_counter()-started)*1000
            save(self.directory/(record['id']+'.json'),record)
            self.records.append(record)
            print(json.dumps({'context':self.context,'success':record['success'],
                'latency_ms':record['wall_clock_ms'],'usage':record.get('response',{}).get('total_tokens'),
                'diagnostics':record.get('diagnostics')}),flush=True)

    def generate(self, system_prompt, user_prompt):
        return self.generate_result(system_prompt,user_prompt).text

    def usage(self):
        successful=[r['response'] for r in self.records if r['success']]
        result={'successful_calls':len(successful),'attempted_calls':len(self.records)}
        for field in ('prompt_tokens','completion_tokens','total_tokens','cost'):
            values=[r[field] for r in successful]
            result[field]=sum(values) if values and all(v is not None for v in values) else None
        return result


def compatibility(chunks,manifest):
    cfg=LLMConfig.from_env()
    if cfg.provider!='gemini' or cfg.model!='gemini-3.5-flash-lite':
        raise ValueError('Requires native gemini and gemini-3.5-flash-lite')
    run_id=str(uuid4());directory=ROOT/'results/phase25'/run_id
    service=LLMService();llm=RecordedLLM(service,directory/'calls')
    question='According to the slides, how often does the number of transistors on a chip double, and what happens to cost?'
    truth='The number of transistors on a chip doubles roughly every two years, with a minimal increase in cost.'
    selected=[RetrievalResult(**c.model_dump(),score=1.0) for c in chunks if c.page_number==3]
    assert selected and 'every two years' in selected[0].text
    report={'run_id':run_id,'provider':'gemini','model':cfg.model,'corpus':manifest,
        'compatibility':[],'complete':False,'benchmark':None,
        'limitations':['Compatibility checks are not benchmark results.','API cost unavailable is recorded as null.']}
    def check(name,fn):
        llm.context={'stage':'compatibility','question_id':'compatibility_slide_3','evaluator':name}
        print('CHECK',name,flush=True)
        result=fn()
        report['compatibility'].append({'check':name,'success':True,
            'result':result.model_dump(mode='json') if hasattr(result,'model_dump') else result})
        save(directory/'compatibility.json',report)
        return result
    try:
        check('simple',lambda:llm.generate('You are a concise assistant.','Reply with OK.'))
        rag=check('rag',lambda:RAGPipeline(None,llm).generate_from_chunks(question,selected))
        check('correctness',lambda:AnswerCorrectnessEvaluator(llm).evaluate(question,truth,rag.answer))
        check('groundedness',lambda:GroundednessEvaluator(llm).evaluate(question,rag.answer,selected))
        check('hallucination_refusal',lambda:HallucinationRefusalEvaluator(llm).evaluate(True,rag.answer,selected))
        check('query_rewrite',lambda:QueryRewriter(llm).rewrite_query(question))
        report['compatibility_passed']=True
    except Exception as exc:
        report['blocker']={**llm.context,'error_type':type(exc).__name__,
                           'diagnostics':getattr(exc,'diagnostics',{}),'response_record':llm.records[-1]['id'] if llm.records else None}
        report['compatibility_passed']=False
    finally:
        service.close();report['usage']=llm.usage()
        save(directory/'compatibility.json',report);save(SUMMARY,report)
        print(json.dumps({'compatibility_passed':report['compatibility_passed'],'usage':report['usage'],
                          'blocker':report.get('blocker'),'directory':str(directory)}),flush=True)
    return report


if __name__=='__main__':
    chunks,manifest=ingest()
    print(json.dumps(manifest),flush=True)
    compatibility(chunks,manifest)
