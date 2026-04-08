# llm-router TODO

## Service Mode
- Stand up as a persistent HTTP service on the cluster (visible over Tailscale)
- All workers hit one endpoint instead of each instantiating their own router
- Central request queue with rate limiting and retry logic
- Single process manages all provider connections — no per-worker connection overhead
- Observability: log every request with timing, provider, model, success/fail
- Could run on Dell32FU alongside the dashboard

## Cleanup
- The 300s timeout mystery was actually SimBackend loading data, not the router.
  But the investigation revealed we need better error attribution — when a call
  fails, report which layer timed out (DNS, connect, first-byte, read).
