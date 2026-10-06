from unittest.mock import Mock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.experiments.experiment_runner import ExperimentRunner
from app.generation.llm_service import LLMError, LLMService
from app.main import create_app
from app.schemas.generation import RAGResponse


def config(strategy="dense", **updates):
    return {"experiment_name": "api-test", "retrieval_strategy": strategy,
            "benchmark_name": "synthetic", "model": "test-model", **updates}


@pytest.fixture
def api(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("External service must not be constructed")
    monkeypatch.setattr(LLMService, "__init__", forbidden)
    pipeline = Mock()
    pipeline.llm.model = "test-model"
    pipeline.ask.return_value = RAGResponse(question="Question?", answer="Answer", citations=[],
                                           retrieved_chunks=[], model="test-model", latency_ms=2)
    # Empty synthetic benchmarks exercise real runner serialization without
    # executing retrieval, generation or judges. All components are mocks.
    real_runner = ExperimentRunner(retrievers={"dense": Mock(), "bm25": Mock()}, generator=pipeline,
                                   correctness=Mock(), groundedness=Mock(), citation_accuracy=Mock(),
                                   hallucination_refusal=Mock())
    runner = Mock(wraps=real_runner)
    runner.retrievers = real_runner.retrievers
    app = create_app(pipeline=pipeline, experiment_runner=runner, results_dir=tmp_path)
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, pipeline, runner, tmp_path


def test_health_root_and_query(api):
    client, pipeline, _, _ = api
    assert client.get("/health").json() == {"status": "healthy"}
    assert client.get("/").json() == {"name": "RAGEval", "status": "running"}
    response = client.post("/query", json={"question": "Question?", "top_k": 3})
    assert response.status_code == 200
    assert response.json()["answer"] == "Answer"
    pipeline.ask.assert_called_once_with("Question?", top_k=3)


@pytest.mark.parametrize("payload", [{"question": " "}, {"question": "Q", "top_k": 0},
                                     {"question": "Q", "top_k": True}, {}])
def test_invalid_query(api, payload):
    client, pipeline, _, _ = api
    assert client.post("/query", json=payload).status_code == 400
    pipeline.ask.assert_not_called()


def test_experiment_saved_read_and_analysis(api):
    client, _, runner, root = api
    response = client.post("/experiments/run", json={"benchmark": [], "config": config()})
    assert response.status_code == 200
    result = response.json()
    identifier = result["experiment_id"]
    assert client.get(f"/experiments/{identifier}").json() == result
    assert runner.run.call_args.kwargs == {"output_dir": root / "experiments"}
    analysis = client.post("/failure-analysis", json={"experiment_id": identifier})
    assert analysis.status_code == 200
    assert analysis.json()["summary"]["total_questions_analyzed"] == 0
    assert runner.run.call_count == 1  # Reads and analysis never execute research.


def test_comparison_delegates_and_persists(api):
    client, _, runner, _ = api
    response = client.post("/comparisons/run", json={"benchmark": [], "configs": [config(), config("bm25")]})
    assert response.status_code == 200
    result = response.json()
    assert runner.run.call_count == 2
    assert [call.args[1].retrieval_strategy for call in runner.run.call_args_list] == ["dense", "bm25"]
    assert client.get(f"/comparisons/{result['comparison_id']}").json() == result


def test_top_k_sweep_delegates_and_persists(api):
    client, _, runner, _ = api
    response = client.post("/sweeps/run", json={"benchmark": [], "base_config": config(), "sweep_type": "top_k", "values": [1, 3]})
    assert response.status_code == 200
    result = response.json()
    assert [call.args[1].top_k for call in runner.run.call_args_list] == [1, 3]
    assert client.get(f"/sweeps/{result['sweep_id']}").json() == result


def test_chunk_size_is_explicitly_unsupported(api):
    client, _, runner, _ = api
    response = client.post("/sweeps/run", json={"benchmark": [], "base_config": config(), "sweep_type": "chunk_size", "values": [256]})
    assert response.status_code == 501 and "rebuild hook" in response.json()["detail"]
    runner.run.assert_not_called()


def test_invalid_research_configuration(api):
    client, _, runner, _ = api
    assert client.post("/experiments/run", json={"benchmark": [], "config": config(top_k=0)}).status_code == 400
    assert client.post("/comparisons/run", json={"benchmark": [], "configs": [config(), config(model="other")]}).status_code == 400
    assert client.post("/sweeps/run", json={"benchmark": [], "base_config": config(), "sweep_type": "top_k", "values": [1, 1]}).status_code == 400
    runner.run.assert_not_called()


@pytest.mark.parametrize("kind", ["experiments", "comparisons", "sweeps"])
def test_missing_invalid_and_traversal_ids(api, kind):
    client = api[0]
    assert client.get(f"/{kind}/{uuid4()}").status_code == 404
    for identifier in ("not-a-uuid", "..%5Csecrets", "%2E%2E%2Fsecrets"):
        assert client.get(f"/{kind}/{identifier}").status_code in (400, 404)
    assert client.post("/failure-analysis", json={"experiment_id": "../secrets"}).status_code == 400
    assert client.post("/failure-analysis", json={"experiment_id": str(uuid4())}).status_code == 404


@pytest.mark.parametrize("error,status", [(LLMError("secret-key"), 502), (RuntimeError("secret-key"), 500),
                                         (ValueError("secret-key"), 400)])
def test_service_errors_are_sanitized(api, error, status):
    client, pipeline, _, _ = api
    pipeline.ask.side_effect = error
    response = client.post("/query", json={"question": "Q"})
    assert response.status_code == status
    assert "secret-key" not in response.text


def test_corrupt_saved_result_is_internal_failure(api):
    client, _, _, root = api
    identifier = uuid4()
    (root / "experiments").mkdir()
    (root / "experiments" / f"{identifier}.json").write_text('{"secret": "key"}')
    response = client.get(f"/experiments/{identifier}")
    assert response.status_code == 500
    assert "secret" not in response.text


def test_unconfigured_startup_is_lazy(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("No provider initialization")
    monkeypatch.setattr(LLMService, "__init__", forbidden)
    with TestClient(create_app(results_dir=tmp_path)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/query", json={"question": "Q"}).status_code == 503
        assert client.post("/experiments/run", json={"benchmark": [], "config": config()}).status_code == 503
        assert client.get(f"/experiments/{uuid4()}").status_code == 404
