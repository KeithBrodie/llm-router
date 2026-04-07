# llm-router

Discover LLM backends and route chat requests by model name. One interface, multiple providers.

## What it does

`LLMRouter` probes your available LLM endpoints at startup, builds a catalog of which models are where, and routes `.chat()` calls to the right backend. You don't need to know or care which provider serves a given model.

Supports:
- **Ollama** (local or remote, including RunPod)
- **OpenAI-compatible APIs** (OpenRouter, vLLM, LiteLLM, etc.)

## Quick start

```python
from llm_router import LLMRouter

router = LLMRouter()  # loads config, probes backends

# What's available?
print(router.model_names())

# Chat
response = router.chat("qwen2.5:7b-instruct-16k", [
    {"role": "system", "content": "Reply in one sentence."},
    {"role": "user", "content": "What is 2+2?"},
])
print(response.text)
```

## Configuration

Copy `config.example.yaml` to `~/.config/llm-router/config.yaml`:

```yaml
backends:
  - label: local-ollama
    type: ollama
    url: http://localhost:11434
    enabled: true

  - label: openrouter
    type: openai
    url: https://openrouter.ai/api/v1
    api_key: env:OPENROUTER_API_KEY
    enabled: true
```

API keys can be specified as:
- Direct values: `"sk-abc123..."`
- Environment variables: `"env:OPENROUTER_API_KEY"`
- Files: `"file:~/.secrets/openrouter.key"`

Config search order: `LLM_ROUTER_CONFIG` env var, `~/.config/llm-router/config.yaml`, `./llm-router.yaml`.

## Programmatic usage

You can skip the config file and add backends in code:

```python
from llm_router import LLMRouter
from llm_router.backends import OllamaBackend, OpenAIBackend

router = LLMRouter(auto_probe=False)
router.add_backend(OllamaBackend("http://localhost:11434", label="local"))
router.add_backend(OpenAIBackend(
    "https://openrouter.ai/api/v1",
    api_key="sk-...",
    label="openrouter",
))
router.probe()
```

## Singleton

Use `LLMRouter.instance()` for a process-wide singleton that probes once:

```python
router = LLMRouter.instance()
```

## API

### `LLMRouter`

| Method | Description |
|--------|-------------|
| `LLMRouter(config_path=None, auto_probe=True)` | Create router, optionally probe on init |
| `.instance(config_path=None)` | Get or create singleton |
| `.probe()` | Probe all backends, return `{label: [models]}` |
| `.models()` | List of `ModelInfo(name, backend, backend_label)` |
| `.model_names()` | List of model name strings |
| `.has_model(name)` | Check availability |
| `.chat(model, messages, temperature=0.7, max_tokens=4096)` | Route a chat request |
| `.add_backend(backend)` | Add a backend programmatically |

### `ChatResponse`

| Field | Type | Description |
|-------|------|-------------|
| `text` | `str` | Generated response text |
| `model` | `str` | Model that was used |
| `usage` | `dict \| None` | Token counts (`prompt_tokens`, `completion_tokens`) |

## Install

```bash
pip install -e /path/to/llm-router
```

Or add the project directory to your Python path.

## License

MIT
