"""Backend implementations for LLM providers."""

from .ollama import OllamaBackend
from .openai_compat import OpenAIBackend

__all__ = ["OllamaBackend", "OpenAIBackend"]
