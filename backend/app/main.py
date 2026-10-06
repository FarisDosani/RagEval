from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.generation.llm_service import LLMError


def create_app(*, pipeline=None, experiment_runner=None, results_dir: str | Path | None = None, auto_init=False) -> FastAPI:
    """Injected services are caller-owned. Opt-in startup owns its local runtime."""
    @asynccontextmanager
    async def lifespan(app):
        from app.runtime import auto_init_enabled, build_runtime
        runtime = None
        enabled = auto_init_enabled() if auto_init is None else auto_init
        # Explicit injection always wins, including intentionally partial setups.
        if enabled and pipeline is None and experiment_runner is None:
            runtime = build_runtime()
            app.state.pipeline = runtime.pipeline
            app.state.experiment_runner = runtime.experiment_runner
        try:
            yield
        finally:
            if runtime is not None:
                runtime.close()
                app.state.pipeline = None
                app.state.experiment_runner = None

    app = FastAPI(title="RAGEval", description="Evaluation framework for Retrieval-Augmented Generation systems", version="0.1.0", lifespan=lifespan)
    app.state.pipeline = pipeline
    app.state.experiment_runner = experiment_runner
    app.state.results_dir = Path(results_dir) if results_dir is not None else Path(__file__).resolve().parents[1] / "results"

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        # Never echo submitted data, exception messages or provider credentials.
        return JSONResponse(status_code=400, content={"detail": "Invalid request fields or identifier"})

    @app.exception_handler(ValueError)
    async def invalid_config(request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"detail": "Invalid research request or configuration"})

    @app.exception_handler(LLMError)
    async def provider_error(request: Request, exc: LLMError):
        return JSONResponse(status_code=502, content={"detail": "LLM provider request failed"})

    @app.exception_handler(Exception)
    async def internal_error(request: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": "Internal research service error"})

    @app.get("/")
    def root():
        return {"name": "RAGEval", "status": "running"}

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    app.include_router(router)
    return app


app = create_app(auto_init=None)
