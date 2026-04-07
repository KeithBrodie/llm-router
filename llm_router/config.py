"""Configuration loading — YAML file + environment variables."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


_DEFAULT_CONFIG_PATHS = [
    Path("~/.config/llm-router/config.yaml").expanduser(),
    Path("llm-router.yaml"),  # CWD fallback
]


def load_config(path: str | Path | None = None) -> dict:
    """Load router config from YAML file.

    Search order:
      1. Explicit path argument
      2. LLM_ROUTER_CONFIG env var
      3. ~/.config/llm-router/config.yaml
      4. ./llm-router.yaml

    Returns empty dict if no config found (backends can still be
    added programmatically).
    """
    candidates = []

    if path:
        candidates.append(Path(path))
    if env := os.environ.get("LLM_ROUTER_CONFIG"):
        candidates.append(Path(env))
    candidates.extend(_DEFAULT_CONFIG_PATHS)

    for p in candidates:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f) or {}

    return {}


def resolve_api_key(key_spec: str | None) -> str | None:
    """Resolve an API key from a config value.

    Supports:
      - Direct string value: "sk-..."
      - Environment variable reference: "env:OPENROUTER_API_KEY"
      - File reference: "file:~/.secrets/openrouter.key"

    Returns None if key_spec is None or resolution fails.
    """
    if not key_spec:
        return None

    if key_spec.startswith("env:"):
        var_name = key_spec[4:].strip()
        return os.environ.get(var_name)

    if key_spec.startswith("file:"):
        path = Path(key_spec[5:].strip()).expanduser()
        try:
            return path.read_text().strip()
        except Exception:
            return None

    return key_spec
