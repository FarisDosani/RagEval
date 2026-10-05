"""Run gated fresh Phase 25 stages through the existing research components.

Every request is recorded before any judge parses its response. Completed
experiments are reused only with exact config and benchmark hash matches.
No retries, score repair, or automatic advancement after a failure.
"""
import argparse
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

from app.experiments.phase25 import ROOT, DATA, SUMMARY, RecordedLLM, save
from app.experiments.phase25_benchmark import build_benchmark, validate_benchmark, QUESTIONS
from app.experiments.phase25_resume import PacedRecordedLLM
from app.experiments.experiment_runner import ExperimentRunner, _digest
from app.experiments.comparison_runner import ComparisonRunner
from app.experiments.models import ExperimentConfig, ExperimentResult
from app.schemas.chunk import DocumentChunk
from app.schemas.benchmark import BenchmarkItem
from app.schemas.embedding import EmbeddedChunk
from app.embeddings import embedding_service
from app.retrieval.vector_store import VectorStore
from app.retrieval.bm25_retriever import BM25Retriever
from app.retrieval.hybrid_retriever import HybridRetriever
from app.generation.llm_service import LLMService, LLMConfig
from app.generation.rag_pipeline import RAGPipeline
from app.evaluation.answer_correctness import AnswerCorrectnessEvaluator
from app.evaluation.groundedness import GroundednessEvaluator
from app.evaluation.citation_accuracy import CitationAccuracyEvaluator
from app.evaluation.hallucination_refusal import HallucinationRefusalEvaluator
from app.evaluation.failure_analysis import analyze_experiment
from app.evaluation.retrieval_metrics import evaluate_retrieval

SMOKE = (1, 18, 25)
SUBSET = (1, 3, 8, 12, 15, 18, 23, 24, 25, 29)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


class TracedComponent:
    """Attach existing method/question context without altering component logic."""
    def __init__(self, component, name, llm, questions, directory):
        self.component=component; self.name=name; self.llm=llm
        self.questions=questions; self.directory=directory

    def __getattr__(self, name):
        method=getattr(self.component,name)
        if not callable(method):
            return method
        def call(*args, **kwargs):
            self.llm.context['evaluator']=self.name
            if args and isinstance(args[0],str) and args[0] in self.questions:
                self.llm.context['question_id']=self.questions[args[0]]
            wait_before=self.llm.wait_ms if isinstance(self.llm,PacedRecordedLLM) else 0.0
            value=method(*args, **kwargs)
            if name=='generate_from_chunks':
                if isinstance(self.llm,PacedRecordedLLM):
                    value=value.model_copy(update={'latency_ms':max(value.usage.latency_ms or 0,
                        value.latency_ms-(self.llm.wait_ms-wait_before))})
                save(self.directory/'generations'/(str(uuid4())+'.json'), {
                    **self.llm.context,'result':value.model_dump(mode='json')})
            return value
        return call


class SavedRunner:
    """Delegate execution and preserve completed experiments before next config."""
    def __init__(self, runner, llm, report, stage):
        self.runner=runner; self.llm=llm; self.report=report; self.stage=stage

    def run(self, benchmark, config, **kwargs):
        items=list(benchmark)
        records=self.report.setdefault('stages',{}).setdefault(self.stage,{'experiments':[]})
        records.setdefault('experiments',[])
        for saved in records['experiments']:
            result=ExperimentResult.model_validate(read(ROOT/'results/experiments'/(saved['experiment_id']+'.json')))
            if result.config==config and result.benchmark_sha256==_digest([i.model_dump(mode='json') for i in items]):
                return result
        self.llm.context={'stage':self.stage,'strategy':config.experiment_name}
        checkpoint=ROOT/'results/phase25/checkpoints'/(_digest({
            'config':config.model_dump(mode='json'),
            'benchmark':[i.model_dump(mode='json') for i in items]})+'.json')
        self.report['active_checkpoint']=str(checkpoint.relative_to(ROOT))
        save(SUMMARY,self.report)
        result=self.runner.run(items,config,checkpoint_path=checkpoint,
            excluded_wait_ms=lambda:self.llm.wait_ms,**kwargs)
        analysis=analyze_experiment(result)
        save(ROOT/'results/failure_analysis'/(str(result.experiment_id)+'.json'),analysis.model_dump(mode='json'))
        records['experiments'].append({'experiment_id':str(result.experiment_id),
            'config':config.model_dump(mode='json'),'failure_analysis':analysis.summary.model_dump(mode='json')})
        save(SUMMARY,self.report)
        return result


def prepare():
    chunks=[DocumentChunk.model_validate(c) for c in read(DATA/'chunks_512.json')]
    items=build_benchmark(chunks)
    path=ROOT/'data/benchmarks/ca_small_benchmark.json'
    serialized=[i.model_dump(mode='json') for i in items]
    if path.exists() and read(path)!=serialized:
        raise ValueError('Existing fresh benchmark differs; refuse to overwrite')
    save(path,serialized)
    save(DATA/'benchmark_provenance.json',[
        {'question_id':i.question_id,'support_pages':q[2], 'relevant_chunk_ids':i.relevant_chunk_ids}
        for i,q in zip(items,QUESTIONS,strict=True)])
    return chunks,items


def retrieval_only(hybrid, selected, report):
    """One real hybrid ranking per question; evaluate prefixes without an LLM."""
    path=ROOT/'results/phase25/retrieval_top_k.json'
    identity=_digest([i.model_dump(mode='json') for i in selected])
    if path.exists():
        artifact=read(path)
        if artifact['benchmark_sha256']!=identity:
            raise ValueError('Retrieval-only benchmark identity mismatch')
    else:
        rankings={i.question_id:hybrid.search(i.question,candidate_k=20,final_top_k=5) for i in selected}
        artifact={'benchmark_sha256':identity,'question_ids':[i.question_id for i in selected],
            'candidate_k':20,'api_calls':0,'results':[],
            'rankings':{q:[c.model_dump(mode='json') for c in chunks] for q,chunks in rankings.items()}}
        for k in (1,3,5):
            metrics=evaluate_retrieval(selected,{q:[c.chunk_id for c in chunks[:k]] for q,chunks in rankings.items()},(k,))
            artifact['results'].append({'top_k':k,**asdict(metrics)})
        save(path,artifact)
    report['retrieval_only_top_k']={'path':str(path.relative_to(ROOT)),**artifact}


def run_stage(stage, *, resume_rate_limit=False):
    report=read(SUMMARY)
    if not report.get('compatibility_passed'):
        raise ValueError('Compatibility gate has not passed')
    previous={'subset':'smoke'}.get(stage)
    if previous and not report.get('stages',{}).get(previous,{}).get('comparison_id'):
        raise ValueError('Previous stage must complete first')
    if report.get('blocker'):
        if not resume_rate_limit or report['blocker'].get('diagnostics',{}).get('http_status')!=429:
            raise ValueError('Recorded failure requires explicit diagnosis before resuming')
        report.setdefault('rate_limit_interruptions',[]).append(report.pop('blocker'))
    if report.get('stages',{}).get(stage,{}).get('comparison_id'):
        print('Stage already complete; no calls made',flush=True)
        return True
    chunks=[DocumentChunk.model_validate(c) for c in read(DATA/'chunks_512.json')]
    items=[BenchmarkItem.model_validate(i) for i in read(ROOT/'data/benchmarks/ca_small_benchmark.json')]
    validate_benchmark(items,chunks)
    numbers=SMOKE if stage=='smoke' else SUBSET
    selected=[items[n-1] for n in numbers]
    subset=[items[n-1] for n in SUBSET]
    save(DATA/'evaluation_subset.json',{'question_ids':[i.question_id for i in subset],
        'benchmark_sha256':_digest([i.model_dump(mode='json') for i in items]),
        'selection':'Fixed IDs: eight answerable, two unanswerable; multiple categories and all difficulty levels.'})
    report['scope']='simplified_phase25'
    report['selected_subset_ids']=[i.question_id for i in subset]
    max_words=max(len(c.text.split()) for c in chunks)
    if max_words>=256:
        raise ValueError('Short-slide chunk-size limitation no longer holds')
    report['skipped_experiments']={
        'full_30_question_runs':'Replaced by a controlled 10-question subset to limit free-tier usage.',
        'llm_only_reranker_query_rewrite':'Excluded from the simplified scope.',
        'generation_top_k_sweep':'Replaced by retrieval-only top_k 1, 3, 5; no top_k=10 run.',
        'chunk_size_sweep':f'All page units are shorter than 256 words (maximum {max_words}); page-preserving chunking has identical boundaries at 256/512/1024.'}
    for skipped in ('full','top_k_sweep','chunk_size_sweep'):
        report.setdefault('stages',{})[skipped]={'status':'skipped','reason':report['skipped_experiments'][{'full':'full_30_question_runs','top_k_sweep':'generation_top_k_sweep','chunk_size_sweep':'chunk_size_sweep'}[skipped]]}
    report['benchmark']={'name':'ca_small_benchmark','questions':len(items),
        'answerable':sum(i.answerable for i in items),'unanswerable':sum(not i.answerable for i in items),
        'sha256':_digest([i.model_dump(mode='json') for i in items]),
        'smoke_question_ids':[items[n-1].question_id for n in SMOKE],
        'subset_question_ids':[items[n-1].question_id for n in SUBSET]}
    cfg=LLMConfig.from_env()
    if cfg.provider!='gemini' or cfg.model!='gemini-3.5-flash-lite':
        raise ValueError('Provider/model identity mismatch')
    directory=ROOT/'results/phase25'/str(uuid4())
    service=LLMService(); llm=PacedRecordedLLM(service,directory/'calls',ROOT/'results/phase25',ROOT/'results/phase25/pacing.json')
    success=False
    try:
        print('Loading preserved embeddings into dense, BM25 and hybrid resources',flush=True)
        embedded=[EmbeddedChunk.model_validate(c) for c in read(DATA/'embeddings_512.json')]
        if [c.model_dump(exclude={'embedding'}) for c in embedded]!=[c.model_dump() for c in chunks]:
            raise ValueError('Saved embeddings do not match current chunks')
        dense=VectorStore(embedded);bm25=BM25Retriever(chunks)
        hybrid=HybridRetriever(dense,bm25)
        retrieval_only(hybrid,subset,report)
        questions={i.question:i.question_id for i in items}
        def trace(component,name):
            return TracedComponent(component,name,llm,questions,directory)
        runner=ExperimentRunner(retrievers={'dense':dense,'bm25':bm25,'hybrid':hybrid},
            generator=trace(RAGPipeline(dense,llm),'generation'),
            correctness=trace(AnswerCorrectnessEvaluator(llm),'correctness'),
            groundedness=trace(GroundednessEvaluator(llm),'groundedness'),
            citation_accuracy=trace(CitationAccuracyEvaluator(llm),'citation_accuracy'),
            hallucination_refusal=trace(HallucinationRefusalEvaluator(llm),'hallucination_refusal'))
        corpus=report['corpus']['corpus_sha256']
        configs=[ExperimentConfig(experiment_name='ca_small_gemini_'+stage+'_'+strategy,
            retrieval_strategy=strategy,top_k=5,model=cfg.model,benchmark_name='ca_small_benchmark',
            chunk_size=512,component_metadata={'provider':'gemini','corpus_id':corpus,
                'index_id':_digest([c.model_dump() for c in chunks]),
                'embedding_model':embedding_service.DEFAULT_MODEL_NAME,'chunk_overlap':'50'})
            for strategy in ('dense','bm25','hybrid')]
        comparison=ComparisonRunner(SavedRunner(runner,llm,report,stage)).run(selected,configs)
        report['stages'][stage]['comparison_id']=str(comparison.comparison_id)
        report['stages'][stage]['status']='completed'
        report['stages'][stage]['strategy_status']={s:'completed' for s in ('dense','bm25','hybrid')}
        report.pop('active_checkpoint',None)
        success=True
        print('COMPLETED',stage,str(comparison.comparison_id),flush=True)
    except Exception as exc:
        report['blocker']={**llm.context,'error_type':type(exc).__name__,
            'message':str(exc) if not getattr(exc,'diagnostics',None) else 'Provider request failed',
            'diagnostics':getattr(exc,'diagnostics',{}),
            'response_record':llm.records[-1]['id'] if llm.records else None}
        # Pydantic/schema errors are local and useful for explaining strict failures.
        if isinstance(exc,ValueError) and exc.__cause__ is not None:
            report['blocker']['validation_detail']=str(exc.__cause__)
        report.setdefault('stages',{}).setdefault(stage,{})['status']='blocked'
        print(json.dumps({'stopped':report['blocker']}),flush=True)
    finally:
        service.close()
        report.setdefault('execution_sessions',[]).append({'stage':stage,'directory':str(directory.relative_to(ROOT)), 'usage':llm.usage(),
            'replayed_calls':llm.replayed_calls,'quota_wait_ms':llm.wait_ms})
        successful=[];attempts=0
        for path in (ROOT/'results/phase25').glob('*/calls/*.json'):
            record=read(path);attempts+=1
            if record['success']:successful.append(record['response'])
        report['cumulative_gemini_usage']={'attempted_calls':attempts,'successful_calls':len(successful)}
        for field in ('prompt_tokens','completion_tokens','total_tokens','cost'):
            values=[r[field] for r in successful]
            report['cumulative_gemini_usage'][field]=sum(values) if values and all(v is not None for v in values) else None
        report['complete']=bool(report.get('stages',{}).get('subset',{}).get('comparison_id')) and not report.get('blocker')
        report['limitations']=[
            'Small controlled subset of 10 questions (8 answerable, 2 unanswerable); no broad statistical conclusions.',
            'Gemini free-tier quota limited experiment scale. Request pacing targets 5.5-second intervals; no automatic retries.',
            'Judge and generation calls use the same model; automated judge scores are not independent human validation.',
            'API cost is unavailable (null). Usage ledger includes compatibility and interrupted calls with successful responses.',
            'Experiment usage totals cover answer generation only; cumulative Gemini totals include all judges.',
            'New experiment latency excludes deliberate quota waits; individual provider latency is retained in generation usage.',
            'Recall@K is any-relevant-hit recall, not completeness of all multi-chunk evidence; MRR uses the returned ranking.',
            'Chunk-size sweep skipped: page units are shorter than 256 words, so chunk boundaries are identical.',
            'Evidence is extracted slide text; image-only equations and diagrams are not evaluated.']
        save(SUMMARY,report)
    return success


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','smoke','subset','minimal'))
    parser.add_argument('--resume-rate-limit',action='store_true',help='Resume a preserved 429 interruption after the recorded reset time; no retry loop.')
    args=parser.parse_args()
    if args.stage=='prepare':
        chunks,items=prepare()
        print(json.dumps({'chunks':len(chunks),'questions':len(items),'answerable':sum(i.answerable for i in items)}))
    elif args.stage=='minimal':
        if run_stage('smoke',resume_rate_limit=args.resume_rate_limit):
            run_stage('subset',resume_rate_limit=args.resume_rate_limit)
    else:
        run_stage(args.stage,resume_rate_limit=args.resume_rate_limit)
