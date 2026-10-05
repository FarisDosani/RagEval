"""Conservative quota pacing and exact request replay for Phase 25 only."""
import json
import re
import time
from datetime import datetime, timezone

from app.experiments.experiment_runner import _digest
from app.experiments.phase25 import RecordedLLM, save
from app.schemas.generation import LLMGenerationResult


class PacedRecordedLLM(RecordedLLM):
    """Pace new requests toward 5.5-second intervals; never retry a failure.

Successful identical requests from this fresh run may be replayed locally when
resuming an interrupted question. Provider/model, strategy, stage, question,
evaluator and both exact prompts must match. Every replay is recorded separately
and does not inflate the API usage ledger. Run only one executor at a time.
"""
    def __init__(self, service, directory, ledger_root, pace_path, interval=5.5):
        super().__init__(service, directory)
        self.interval = interval
        self.pace_path = pace_path
        self.wait_ms = 0.0
        self.replayed_calls = 0
        self.cache = {}
        for path in sorted(ledger_root.glob('*/calls/*.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            if record.get('success') and record.get('provider') == 'gemini' and record.get('model') == self.model:
                self.cache.setdefault(self.key(record), (record['id'], record['response']))

    @staticmethod
    def key(record):
        return _digest({key: record.get(key) for key in (
            'provider', 'model', 'stage', 'strategy', 'question_id', 'evaluator',
            'system_prompt', 'user_prompt',
        )})

    def generate_result(self, system_prompt, user_prompt):
        key = self.key({'provider':'gemini', 'model':self.model, **self.context,
                        'system_prompt':system_prompt, 'user_prompt':user_prompt})
        if key in self.cache:
            record_id, response = self.cache[key]
            self.replayed_calls += 1
            save(self.directory.parent/'replays'/f'{self.replayed_calls}.json', {
                **self.context, 'original_record_id':record_id, 'request_sha256':key})
            print(json.dumps({'replayed':record_id, 'context':self.context}), flush=True)
            return LLMGenerationResult.model_validate(response)
        state = json.loads(self.pace_path.read_text(encoding='utf-8')) if self.pace_path.exists() else {}
        delay = max(0.0, state.get('next_request_at', 0.0) - time.time())
        started = time.perf_counter()
        while delay > 0:
            time.sleep(min(delay, 30.0))
            delay = max(0.0, state['next_request_at'] - time.time())
        self.wait_ms += (time.perf_counter() - started) * 1000
        save(self.pace_path, {'next_request_at':time.time()+self.interval})
        try:
            response = super().generate_result(system_prompt, user_prompt)
        except Exception as exc:
            diagnostics = getattr(exc, 'diagnostics', {})
            if diagnostics.get('http_status') == 429:
                message = diagnostics.get('provider_error', {}).get('message', '')
                match = re.search(r'retry in\s+([\d.]+)s', message, re.IGNORECASE)
                cooldown = max(self.interval, float(match.group(1))+1 if match else 60.0)
                save(self.pace_path, {'next_request_at':time.time()+cooldown,
                    'interrupted_at':datetime.now(timezone.utc).isoformat(),
                    'http_status':429, 'retry_delay_seconds':cooldown,
                    'note':'Stopped without retry. Resume is an explicit later invocation.'})
            raise
        self.cache[key] = (self.records[-1]['id'], response.model_dump(mode='json'))
        return response
