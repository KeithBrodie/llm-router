"""Configuration loading — YAML file + environment variables + .env."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


# Search order: NAS (shared, single source of truth) → home dir → CWD.
# NAS path works for any cluster machine with /mnt/public mounted.
# Home dir fallback for machines without NAS (e.g. Tokyo VPS).
_DEFAULT_CONFIG_PATHS = [
    Path("/mnt/public/Vault/Projects/llm-router/config.yaml"),
    Path("~/.config/llm-router/config.yaml").expanduser(),
    Path("llm-router.yaml"),  # CWD fallback
]

_DEFAULT_DOTENV_PATHS = [
    Path("/mnt/public/Vault/Projects/llm-router/.env"),
    Path(".env"),  # CWD
    Path("~/.config/llm-router/.env").expanduser(),
]


def _load_dotenv():
    """Load .env file into os.environ. Uses python-dotenv if available,
    falls back to a simple key=value parser."""
    for p in _DEFAULT_DOTENV_PATHS:
        if not p.exists():
            continue
        try:
            from dotenv import load_dotenv
            load_dotenv(p, override=False)
            return
        except ImportError:
            pass
        # Simple fallback: parse KEY=VALUE lines
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value
        return


def load_config(path: str | Path | None = None) -> dict:
    """Load router config from YAML file.

    Loads .env first (if found) so environment variable references
    in the config can resolve.

    Search order:
      1. Explicit path argument
      2. LLM_ROUTER_CONFIG env var
      3. /mnt/public/Vault/Projects/llm-router/config.yaml (NAS — shared)
      4. ~/.config/llm-router/config.yaml (per-machine fallback)
      5. ./llm-router.yaml (CWD)

    Returns empty dict if no config found (backends can still be
    added programmatically).
    """
    _load_dotenv()

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
