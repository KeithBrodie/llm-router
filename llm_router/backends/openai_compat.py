"""OpenAI-compatible backend — OpenRouter, vLLM, LiteLLM, etc."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import Backend, ChatResponse


class OpenAIBackend(Backend):
    """Backend for any OpenAI-compatible API (OpenRouter, vLLM, etc.)."""

    name = "openai"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        label: str | None = None,
        timeout: int = 600,
        models: list[str] | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.label = label or self.base_url
        self.timeout = timeout
        self._explicit_models = models  # optional static model list

    def __repr__(self) -> str:
        return f"OpenAIBackend({self.label!r})"

    def _headers(self) -> dict:
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "llm-router/0.1",
        }

    def probe(self) -> list[str]:
        """Query /models endpoint or return explicit model list."""
        if self._explicit_models:
            # Verify the endpoint is reachable with a lightweight call
            try:
                req = urllib.request.Request(
                    f"{self.base_url}/models",
                    headers=self._headers(),
                )
                with urllib.request.urlopen(req, timeout=10) as resp:
                    resp.read()
                return list(self._explicit_models)
            except Exception:
                return []

        try:
            req = urllib.request.Request(
                f"{self.base_url}/models",
                headers=self._headers(),
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            models = data.get("data", [])
            return [m["id"] for m in models if isinstance(m, dict) and "id" in m]
        except Exception:
            return []

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=data,
            headers=self._headers(),
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"OpenAI-compatible ({self.label}) unreachable: {e}"
            ) from e

        choices = result.get("choices", [])
        if not choices:
            raise ValueError(f"Empty response from {self.label} for model {model}")

        text = choices[0].get("message", {}).get("content", "")
        usage = result.get("usage")
        return ChatResponse(text=text, model=model, usage=usage)
