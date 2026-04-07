"""LLMRouter — singleton that discovers backends and routes chat requests."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

from .backends.base import Backend, ChatResponse
from .backends.ollama import OllamaBackend
from .backends.openai_compat import OpenAIBackend
from .config import load_config, resolve_api_key


@dataclass
class ModelInfo:
    """A model available through the router."""
    name: str
    backend: Backend
    backend_label: str


class LLMRouter:
    """Discover LLM backends and route chat requests by model name.

    Usage::

        router = LLMRouter()          # loads config, probes backends
        router = LLMRouter.instance() # singleton access

        # What's available?
        for m in router.models():
            print(m.name, m.backend_label)

        # Chat
        response = router.chat("qwen2.5:7b-instruct-16k", [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ])
        print(response.text)
    """

    _instance: LLMRouter | None = None
    _lock = threading.Lock()

    def __init__(self, config_path: str | None = None, auto_probe: bool = True):
        self._backends: list[Backend] = []
        self._model_map: dict[str, ModelInfo] = {}
        self._config = load_config(config_path)

        self._build_backends()
        if auto_probe:
            self.probe()

    @classmethod
    def instance(cls, config_path: str | None = None) -> LLMRouter:
        """Get or create the singleton router instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config_path=config_path)
        return cls._instance

    @classmethod
    def reset(cls):
        """Reset the singleton (for testing)."""
        with cls._lock:
            cls._instance = None

    def _build_backends(self):
        """Create backend instances from config."""
        for entry in self._config.get("backends", []):
            backend_type = entry.get("type", "")
            enabled = entry.get("enabled", True)
            if not enabled:
                continue

            label = entry.get("label", entry.get("url", ""))

            if backend_type == "ollama":
                url = entry.get("url", "http://localhost:11434")
                timeout = entry.get("timeout", 300)
                self._backends.append(
                    OllamaBackend(base_url=url, label=label, timeout=timeout)
                )

            elif backend_type == "openai":
                url = entry.get("url", "")
                api_key = resolve_api_key(entry.get("api_key"))
                if not url or not api_key:
                    continue
                timeout = entry.get("timeout", 300)
                models = entry.get("models")  # optional explicit list
                self._backends.append(
                    OpenAIBackend(
                        base_url=url, api_key=api_key, label=label,
                        timeout=timeout, models=models,
                    )
                )

    def add_backend(self, backend: Backend):
        """Add a backend programmatically (after construction)."""
        self._backends.append(backend)

    def probe(self) -> dict[str, list[str]]:
        """Probe all backends and build the model catalog.

        Returns dict of {backend_label: [model_names]}.
        """
        self._model_map.clear()
        results = {}

        for backend in self._backends:
            models = backend.probe()
            results[backend.label] = models
            for model_name in models:
                # First backend to claim a model wins
                if model_name not in self._model_map:
                    self._model_map[model_name] = ModelInfo(
                        name=model_name,
                        backend=backend,
                        backend_label=backend.label,
                    )

        return results

    def models(self) -> list[ModelInfo]:
        """Return all available models across all backends."""
        return list(self._model_map.values())

    def model_names(self) -> list[str]:
        """Return just the model name strings."""
        return list(self._model_map.keys())

    def has_model(self, model: str) -> bool:
        """Check if a model is available."""
        return model in self._model_map

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Route a chat request to the appropriate backend.

        Args:
            model: Model name (must be in the catalog from probe()).
            messages: List of {"role": str, "content": str} dicts.
            temperature: Sampling temperature.
            max_tokens: Max response tokens.

        Returns:
            ChatResponse with generated text.

        Raises:
            ValueError: If model is not available.
            ConnectionError: If the backend is unreachable.
        """
        info = self._model_map.get(model)
        if info is None:
            available = ", ".join(sorted(self._model_map.keys())[:10])
            raise ValueError(
                f"Model {model!r} not available. "
                f"Available: {available or '(none — run .probe())'}"
            )

        return info.backend.chat(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
