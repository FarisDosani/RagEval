"""Keep published research references and accounting valid after cleanup."""
import json
from pathlib import Path

from app.experiments.models import ExperimentResult
from app.experiments.comparison_runner import ComparisonResult
from app.experiments.phase25 import SUMMARY, USAGE_LEDGER, ROOT, cumulative_usage


def test_published_results_and_comparisons_remain_consistent():
    summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
    assert summary['complete']
    for stage, count in (('smoke', 3), ('subset', 10)):
        saved = summary['stages'][stage]
        comparison = ComparisonResult.model_validate_json(
            (ROOT/'results/comparisons'/f"{saved['comparison_id']}.json").read_text(encoding='utf-8'))
        assert [str(i) for i in comparison.experiment_ids] == [e['experiment_id'] for e in saved['experiments']]
        for entry in saved['experiments']:
            result = ExperimentResult.model_validate_json(
                (ROOT/'results/experiments'/f"{entry['experiment_id']}.json").read_text(encoding='utf-8'))
            assert len(result.questions) == count
            if stage == 'subset':
                assert [q.question_id for q in result.questions] == summary['selected_subset_ids']
            assert (ROOT/'results/failure_analysis'/f"{entry['experiment_id']}.json").is_file()
    assert (ROOT/summary['retrieval_only_top_k']['path']).is_file()


def test_usage_claims_survive_raw_diagnostic_removal(tmp_path):
    summary = json.loads(SUMMARY.read_text(encoding='utf-8'))
    assert cumulative_usage(tmp_path, USAGE_LEDGER) == summary['cumulative_gemini_usage']
