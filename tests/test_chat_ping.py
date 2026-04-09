#!/usr/bin/env python3
"""Send a single chat request with no model specified.

Lets the router pick the default model, sends a short prompt,
and reports what model was requested and what OpenRouter actually used.

Usage:
    python3 tests/test_chat_ping.py
"""

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_router import LLMRouter


def main():
    print("Initializing router...", flush=True)
    router = LLMRouter()

    names = router.model_names()
    if not names:
        print("ERROR: no models available")
        return

    print(f"names[0] from probe: {names[0]}")

    # No model specified — use make_llm_client default
    from hl_discovery.llm_client import make_llm_client
    client = make_llm_client()
    model = client.model
    print(f"make_llm_client() chose: {model}")

    messages = [
        {"role": "user", "content": "Say hello in exactly 3 words."},
    ]

    # Use the router's chat — but also capture the raw response
    # to see what model OpenRouter actually used.
    info = router._model_map[model]
    backend = info.backend

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 20,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{backend.base_url}/chat/completions",
        data=data,
        headers=backend._headers(),
    )

    print(f"Sending request to {backend.base_url}...", flush=True)
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())

    response_model = result.get("model", "MISSING")
    text = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    cost = usage.get("cost", "unknown")

    print(f"\nRequested model:  {model}")
    print(f"Response model:   {response_model}")
    print(f"Match:            {'YES' if model == response_model else 'NO — MISMATCH'}")
    print(f"Response text:    {text}")
    print(f"Tokens:           {usage.get('prompt_tokens', '?')} in, "
          f"{usage.get('completion_tokens', '?')} out")
    print(f"Cost:             ${cost}")

    LLMRouter.reset()
    print("\nRouter shut down.")


if __name__ == "__main__":
    main()
