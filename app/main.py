from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.generation.llm_service import LLMError


def create_app(*, pipeline=None, experiment_runner=None, results_dir: str | Path | None = None) -> FastAPI:
    """Inject prebuilt components; startup never constructs clients or models.

    The caller owns component initialization/lifecycle and corpus selection.
    Without components, saved-result reads remain usable; execution returns 503.
    All runner persistence is directed to the same server-owned results root.
    """
    app = FastAPI(title="RAGEval", description="Evaluation framework for Retrieval-Augmented Generation systems", version="0.1.0")
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


app = create_app()
