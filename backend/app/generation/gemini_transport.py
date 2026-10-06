"""Native non-streaming Gemini transport. No generation or judge policy here."""
import re
from time import perf_counter
from urllib.parse import quote

import httpx

from app.generation.llm_service import LLMError, LLMEmptyResponseError, _tokens
from app.schemas.generation import LLMGenerationResult


class GeminiTransport:
    def __init__(self, config):
        self.model = config.model
        self._key = config.api_key
        self.client = httpx.Client(base_url=config.base_url, timeout=60.0,
                                  headers={"x-goog-api-key": config.api_key})

    def _diagnostics(self, exc):
        response = getattr(exc, "response", None)
        result = {"provider": "gemini", "error_type": type(exc).__name__,
                  "http_status": response.status_code if response is not None else None}
        if response is None:
            return result
        try:
            data = response.json()
        except ValueError:
            result["provider_error"] = {"message": "Non-JSON error body omitted"}
            return result
        error = data.get("error") if isinstance(data, dict) else None
        if isinstance(error, dict):
            safe = {}
            for field in ("code", "status", "message"):
                if field in error:
                    text = str(error[field]).replace(self._key, "[REDACTED]")
                    text = re.sub(r"(?i)(?:bearer\s+\S+|AIza[\w-]+|sk-[\w-]+)", "[REDACTED]", text)
                    safe[field] = text[:2000]
            result["provider_error"] = safe
        return result

    def generate_result(self, system_prompt, user_prompt):
        started = perf_counter()
        try:
            response = self.client.post(
                "models/" + quote(self.model.removeprefix("models/"), safe="") + ":generateContent",
                json={"systemInstruction": {"parts": [{"text": system_prompt}]},
                      "contents": [{"role": "user", "parts": [{"text": user_prompt}]}]},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMError("LLM generateContent request failed", diagnostics=self._diagnostics(exc)) from exc
        elapsed = (perf_counter() - started) * 1000
        try:
            data = response.json()
            candidates = data.get("candidates")
            if not candidates:
                raise LLMError("LLM response contained no candidates")
            candidate = candidates[0]
            if candidate.get("finishReason") != "STOP":
                raise LLMError("LLM response did not finish with a complete text answer")
            parts = candidate.get("content", {}).get("parts", [])
            text = "".join(p["text"] for p in parts if not p.get("thought") and isinstance(p.get("text"), str)).strip()
            if not text:
                raise LLMEmptyResponseError("LLM response contained no assistant text")
            usage = data.get("usageMetadata")
            model = data.get("modelVersion")
            # candidatesTokenCount excludes thoughts; totalTokenCount is retained
            # exactly as supplied rather than summing/inventing missing counts.
            return LLMGenerationResult(text=text, model=model if isinstance(model, str) and model.strip() else self.model,
                latency_ms=elapsed, prompt_tokens=_tokens(usage, "promptTokenCount"),
                completion_tokens=_tokens(usage, "candidatesTokenCount"), total_tokens=_tokens(usage, "totalTokenCount"),
                cost=None)  # Native API provides no documented response cost.
        except (ValueError, TypeError, AttributeError, KeyError, IndexError) as exc:
            raise LLMError("LLM returned malformed generateContent response") from exc

    def close(self):
        self.client.close()
