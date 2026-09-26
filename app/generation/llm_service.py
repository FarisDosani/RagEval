import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values
from openai import OpenAI, OpenAIError


class LLMError(RuntimeError):
    """The configured LLM failed to return a complete text response."""


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    api_key: str = field(repr=False)
    model: str

    @classmethod
    def from_env(cls, env_path: str | Path | None = None) -> "LLMConfig":
        """Read project .env without changing process state; environment wins."""
        path = Path(env_path) if env_path is not None else Path(__file__).resolve().parents[2] / ".env"
        values = dotenv_values(path, interpolate=False)
        names = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
        settings = {name: (os.environ.get(name, values.get(name)) or "").strip() for name in names}
        missing = [name for name, value in settings.items() if not value]
        if missing:
            raise ValueError("Missing LLM configuration: " + ", ".join(missing))
        return cls(*(settings[name] for name in names))


class LLMService:
    """One reusable OpenAI-compatible client, configured for the local gateway."""

    def __init__(self):
        config = LLMConfig.from_env()
        self.model = config.model
        self._client = OpenAI(
            base_url=config.base_url, api_key=config.api_key, timeout=60.0, max_retries=0
        )

    def generate(self, system_prompt: str, user_prompt: str) -> str:
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
        if not response.choices:
            raise LLMError("LLM response contained no choices")
        choice = response.choices[0]
        content = choice.message.content
        if choice.finish_reason != "stop":
            raise LLMError("LLM response did not finish with a complete text answer")
        if not isinstance(content, str) or not content.strip():
            raise LLMError("LLM response contained no assistant text")
        return content.strip()

    def close(self) -> None:
        self._client.close()
