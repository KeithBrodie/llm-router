"""Ollama backend — local or remote Ollama instances."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from .base import Backend, ChatResponse


class OllamaBackend(Backend):
    """Backend for Ollama API (local or remote, including RunPod)."""

    name = "ollama"

    def __init__(self, base_url: str, label: str | None = None, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.label = label or self.base_url
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"OllamaBackend({self.label!r})"

    def probe(self) -> list[str]:
        """Query /api/tags to discover available models."""
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                headers={"User-Agent": "llm-router/0.1"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return [m["name"] for m in data.get("models", [])]
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
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "llm-router/0.1",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode())
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Ollama ({self.label}) unreachable: {e}"
            ) from e

        text = result.get("message", {}).get("content", "")
        usage = None
        if "prompt_eval_count" in result:
            usage = {
                "prompt_tokens": result.get("prompt_eval_count", 0),
                "completion_tokens": result.get("eval_count", 0),
            }
        return ChatResponse(text=text, model=model, usage=usage)
