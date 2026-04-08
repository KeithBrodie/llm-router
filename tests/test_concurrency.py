#!/usr/bin/env python3
"""Concurrency test harness for LLM Router.

Tests sequential and concurrent calls to isolate timeout issues.

Usage:
    python tests/test_concurrency.py
"""

import sys
import time
import concurrent.futures
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "HyperLiquid"))

from llm_router import LLMRouter

# A representative full-size system prompt (from our codegen cases)
SYSTEM = """You are a quantitative trading strategy developer. Write Python strategies.

Strategy, np (numpy), pd (pandas), and math are already available. Do NOT write import statements.
Your class MUST subclass Strategy directly.

class BollingerMeanRevert(Strategy):
    def fit(self, bars):
        closes = bars['close']
        self.mu = np.mean(closes)
        self.sigma = max(np.std(closes), 1e-10)

    def signal(self, bars):
        z = (bars['close'][-1] - self.mu) / self.sigma
        return float(np.clip(-z / 2.0, -1.0, 1.0))

Rules:
- bars dict keys: 'open', 'high', 'low', 'close', 'volume', 'vwap', 'funding', 'premium'
- Each value is a 1D numpy float64 array, ~240 elements.
- signal() must return a single float in [-1.0, +1.0].
- Guard against: short arrays, zero std, NaN, division by zero.

Return ONLY a python code fence containing your class. No explanation.
"""

USER = "Write a mean reversion strategy that uses z-score and volatility ratio."

MODEL = "meta-llama/llama-3.3-70b-instruct"


def make_call(router, call_id, model=MODEL):
    """Single LLM call, returns timing info."""
    t0 = time.time()
    try:
        resp = router.chat(model, [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER},
        ], temperature=0.7 + call_id * 0.01)
        elapsed = time.time() - t0
        return {"id": call_id, "ok": True, "time": elapsed, "chars": len(resp.text)}
    except Exception as e:
        elapsed = time.time() - t0
        return {"id": call_id, "ok": False, "time": elapsed, "error": str(e)}


def print_result(r):
    if r["ok"]:
        print(f"  #{r['id']}: OK {r['time']:.1f}s ({r['chars']} chars)")
    else:
        print(f"  #{r['id']}: FAIL {r['time']:.1f}s — {r['error'][:100]}")


def main():
    print("=== LLM Router Concurrency Test ===")
    print(f"Model: {MODEL}")
    print(f"Prompt: {len(SYSTEM)+len(USER)} chars")
    print()

    LLMRouter.reset()
    router = LLMRouter()

    if not router.has_model(MODEL):
        print(f"ERROR: {MODEL} not available")
        print(f"Available: {router.model_names()[:10]}")
        return

    # Test 1: Single call
    print("--- Test 1: Single call ---")
    r = make_call(router, 0)
    print_result(r)
    print()

    # Test 2: Two sequential calls
    print("--- Test 2: Two sequential calls ---")
    for i in range(2):
        r = make_call(router, i)
        print_result(r)
    print()

    # Test 3: Two concurrent calls
    print("--- Test 3: Two concurrent calls ---")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(make_call, router, i) for i in range(2)]
        for f in concurrent.futures.as_completed(futures):
            print_result(f.result())
    print(f"  Total: {time.time()-t0:.1f}s")
    print()

    # Test 4: Four concurrent calls
    print("--- Test 4: Four concurrent calls ---")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(make_call, router, i) for i in range(4)]
        for f in concurrent.futures.as_completed(futures):
            print_result(f.result())
    print(f"  Total: {time.time()-t0:.1f}s")
    print()

    # Test 5: Eight concurrent calls
    print("--- Test 5: Eight concurrent calls ---")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(make_call, router, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            print_result(f.result())
    print(f"  Total: {time.time()-t0:.1f}s")
    print()

    # Test 6: Eight concurrent, second round (check consistency)
    print("--- Test 6: Eight concurrent (round 2) ---")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(make_call, router, i) for i in range(8)]
        for f in concurrent.futures.as_completed(futures):
            print_result(f.result())
    print(f"  Total: {time.time()-t0:.1f}s")
    print()

    # Test 7: Sixteen concurrent (exceeds typical worker pool)
    print("--- Test 7: Sixteen concurrent ---")
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(make_call, router, i) for i in range(16)]
        for f in concurrent.futures.as_completed(futures):
            print_result(f.result())
    print(f"  Total: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
