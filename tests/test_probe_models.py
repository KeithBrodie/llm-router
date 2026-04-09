#!/usr/bin/env python3
"""Probe all backends and write the model catalog to CSV.

Initializes the router, probes every enabled backend, writes one row
per model to models.csv (model_name, backend_label), then shuts down.

Usage:
    python3 tests/test_probe_models.py
    python3 tests/test_probe_models.py --out /tmp/models.csv
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_router import LLMRouter


def main():
    p = argparse.ArgumentParser(description="Probe backends and dump model catalog")
    p.add_argument("--out", default="models.csv",
                   help="Output CSV path (default: models.csv)")
    args = p.parse_args()

    print("Initializing router...", flush=True)
    router = LLMRouter()

    models = router.models()
    print(f"Found {len(models)} models across {len(router._backends)} backend(s)")

    out_path = Path(args.out)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "backend"])
        for m in models:
            writer.writerow([m.name, m.backend_label])

    print(f"Wrote {out_path}")

    # Per-backend summary
    by_backend: dict[str, int] = {}
    for m in models:
        by_backend[m.backend_label] = by_backend.get(m.backend_label, 0) + 1
    for label, count in sorted(by_backend.items()):
        print(f"  {label}: {count} models")

    LLMRouter.reset()
    print("Router shut down.")


if __name__ == "__main__":
    main()
