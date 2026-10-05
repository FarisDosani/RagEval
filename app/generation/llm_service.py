import os
from math import isfinite
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError

from app.schemas.generation import LLMGenerationResult


def _field(value, name):
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _tokens(usage, name: str) -> int | None:
    value = _field(usage, name)
    return value if type(value) is int and value >= 0 else None


def _cost(value) -> float | None:
    if type(value) in (int, float) and isfinite(value) and value >= 0:
        return float(value)
    return None


class LLMError(RuntimeError):
    """The configured LLM failed to return a complete text response."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


class LLMEmptyResponseError(LLMError):
    """A completed response contained only empty or whitespace text."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str
    provider: str = "openai"

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "LLMConfig":
        """Read project .env without changing process state; environment wins."""
        path = Path(env_path) if env_path is not None else Path(__file__).resolve().parents[2] / ".env"
        values = dotenv_values(path, interpolate=False)
        provider = (os.environ.get("LLM_PROVIDER", values.get("LLM_PROVIDER")) or "openai").strip().lower()
        if provider not in ("openai", "gemini"):
            raise ValueError("LLM_PROVIDER must be openai or gemini")
        names = ("GEMINI_API_KEY", "GEMINI_MODEL") if provider == "gemini" else ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
        settings = {name: (os.environ.get(name, values.get(name)) or "").strip() for name in names}
        missing = [name for name, value in settings.items() if not value]
        if missing:
            raise ValueError("Missing LLM configuration: " + ", ".join(missing))
        if provider == "gemini":
            return cls("https://generativelanguage.googleapis.com/v1beta/", settings[names[0]], settings[names[1]], provider)
        return cls(*(settings[name] for name in names))


class LLMService:
    """Provider-neutral interface over reusable, non-streaming transports."""

    def __init__(self):
        config = LLMConfig.from_env()
        self.model = config.model
        self.provider = config.provider
        if config.provider == "gemini":
            from app.generation.gemini_transport import GeminiTransport
            self._client = GeminiTransport(config)
            return
        self._client = OpenAI(
            base_url=config.base_url, api_key=config.api_key, timeout=60.0, max_retries=0
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Backward-compatible text-only interface."""
        return self.generate_result(system_prompt, user_prompt).text

    def generate_result(self, system_prompt: str, user_prompt: str) -> LLMGenerationResult:
        """Return text and per-call metadata without shared last-response state.

        Model uses the parsed response name when present, otherwise the configured
        name. Cost reads only numeric parsed usage.cost or response.cost (in that
        order). Missing/invalid metadata stays None; totals are never inferred.
        """
        if self.provider == "gemini":
            return self._client.generate_result(system_prompt, user_prompt)
        started = perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                stream=False,
            )
        except OpenAIError as exc:
            raise LLMError("LLM chat completion request failed") from exc
        elapsed_ms = (perf_counter() - started) * 1000
        if not response.choices:
            raise LLMError("LLM response contained no choices")
        choice = response.choices[0]
        content = choice.message.content
        if choice.finish_reason != "stop":
            raise LLMError("LLM response did not finish with a complete text answer")
        if not isinstance(content, str):
            raise LLMError("LLM response contained no assistant text")
        if not content.strip():
            raise LLMEmptyResponseError("LLM response contained no assistant text")
        usage = _field(response, "usage")
        cost = _cost(_field(usage, "cost"))
        if cost is None:
            cost = _cost(_field(response, "cost"))
        model = _field(response, "model")
        return LLMGenerationResult(
            text=content.strip(), model=model if isinstance(model, str) and model.strip() else self.model,
            latency_ms=elapsed_ms, prompt_tokens=_tokens(usage, "prompt_tokens"),
            completion_tokens=_tokens(usage, "completion_tokens"), total_tokens=_tokens(usage, "total_tokens"),
            cost=cost,
        )

    def close(self) -> None:
        self._client.close()
