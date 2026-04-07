"""Base backend interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ChatResponse:
    """Normalized response from any backend."""
    text: str
    model: str
    usage: dict | None = None  # {"prompt_tokens": N, "completion_tokens": N}


class Backend:
    """Abstract backend — subclass for each provider type."""

    name: str = "base"

    def probe(self) -> list[str]:
        """Probe the endpoint and return available model names.

        Returns an empty list if the endpoint is unreachable.
        """
        raise NotImplementedError

    def chat(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """Send a chat completion request.

        Args:
            model: Model name as known to this backend.
            messages: List of {"role": str, "content": str} dicts.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens in the response.

        Returns:
            ChatResponse with the generated text.

        Raises:
            ConnectionError: If the backend is unreachable.
            ValueError: If the model is not available on this backend.
        """
        raise NotImplementedError
