# prefix-router

A small async router (Rust, tokio + axum + reqwest) that sits in front of several OpenAI-compatible LLM servers (vLLM, SGLang) and sends
requests that share a prompt prefix to the same replica, so that replica's prefix cache gets reused. It was written for the agent-style
traffic in `../README.md`: every session starts with the same system prompt and few-shot examples, then adds a new screenshot per turn.

```
clients --> prefix-router --+--> replica 0  (vLLM / SGLang, OpenAI API)
              /metrics      +--> replica 1
              /router/health
```

```
cargo build --release
target/release/prefix-router --listen 127.0.0.1:9000 --backend http://127.0.0.1:8001 --backend http://127.0.0.1:8002
```

## How a request is routed

1. **Prefix key.** In order of preference: an `X-Prefix-Key` header; the first `--prefix-messages N` messages if configured; otherwise all
   `system` messages (what agents typically share); otherwise the first message; otherwise the model name. The key is a deterministic 64-bit
   hash (FNV-1a plus a splitmix64 finaliser; the standard library hasher makes no stability promise).
2. **Consistent hashing with bounded load.** Each replica owns 160 virtual nodes on a ring. A key goes to the first eligible replica clockwise
   from its position whose in-flight count is below `ceil(load_factor * (total + 1) / eligible)` (`--load-factor`, default 1.25). A busy owner
   therefore *spills* the key to the next replica instead of queueing it; if every candidate is at the cap the request goes to the least-loaded
   replica. An unhealthy replica is skipped, so only the keys it owned move (tested), and they come back when it recovers.
3. **Baselines** for benchmarking: `--policy round-robin` and `--policy least-loaded`.

## Failure handling

| situation | behaviour |
|---|---|
| connect error, timeout waiting for headers (`--header-timeout-ms`), a 5xx, or the connection dying / going silent (`--idle-timeout-ms`) **before the first body byte** | retry on a *different* replica, up to `--max-retries` (default 2), with exponential backoff (50 ms doubling, capped 1 s) and +-25% jitter. The router holds the response until the first chunk arrives, so nothing has reached the client yet. |
| replica fails after the first chunk was sent to the client | cannot be replayed: the client sees a broken stream, and the failure counts against the replica. Never silently retried. |
| replica silent for `--idle-timeout-ms` after the first chunk | stream is cut and counted as a failure |
| replica fails `--unhealthy-after` consecutive health checks (`GET /health` every 500 ms) | marked unhealthy and skipped; back after `--healthy-after` successes |
| 5 consecutive failed requests (`--breaker-threshold`) | circuit breaker opens for `--breaker-cooldown-ms` (3 s), then lets exactly **one** probe request through; success closes it, failure reopens it |
| no eligible replica | `503` with a JSON error body |
| client disconnects mid-response | counted as `aborted`, not as a replica failure; a half-open probe slot it held is released |

The breaker and the health checker are deliberately separate: the health check can pass while real requests fail (a replica that is up but
broken in its inference path), and only a real request can prove that recovered. A dead replica therefore still costs one failed attempt per
cooldown (the probe), and the tests assert that bound.

## Observability

`GET /metrics` (Prometheus text): requests, routing decisions (`affinity`, `spill`, `least_loaded`, `round_robin`), retries, requests rejected
for lack of a replica, per-replica in-flight / healthy / breaker state / ok / errors / aborted / retried-to / went-down / came-up, and
histograms of time to the replica's first response byte and of total request time. `GET /router/health` returns JSON per replica (503 if none is
healthy). Responses carry `x-router-backend: <index>`. Everything else (including a replica's own `/health`) is proxied.

## Tests

`cargo test`: **27 unit tests** (hash ring balance and stability, breaker state machine with injected time, selection policy including a
hot-key bound, prefix extraction, backoff, metrics rendering) and **15 integration tests** against mock replicas that stream SSE and can fail,
hang, die before or after their first chunk, be killed and be restarted on the same port: affinity, header override, round-robin
evenness, streaming fidelity and order, failover with zero client errors, retry before the health checker notices, retry of a replica
that dies before its first byte (and no retry after it), the breaker's full open/probe/close cycle, a hung replica, all replicas down,
bounded-load spill under concurrency, and metrics.

## Limits (read before relying on it)

* **The key is message content, not tokens.** The router cannot know where the shared token prefix ends, so the default is "the system
  message". Sessions that share a system prompt but differ in few-shot examples will be grouped together even though only part of their
  prefix is common; `--prefix-messages` or `X-Prefix-Key` can refine this.
* **It does not see the engines' caches.** Affinity raises the chance of a cache hit; it cannot promise one (evictions, different token
  boundaries). A cache-aware router that mirrors each engine's radix tree is a different and larger design.
* **Bounded load caps the maximum, not the minimum.** A very hot key fills replicas in ring order, so the last replica on that key's ring may
  get little. Load is counted in requests, not tokens: a 4k-token prompt weighs the same as a 400-token one.
* **Retries only happen before the first body byte reaches the client**, and replaying a request assumes it is safe to run twice (true for
  the greedy, read-only requests used here). A replica that dies after the first chunk still breaks that one response. Holding the response
  until the first chunk arrives also means the router's time-to-headers metric now includes the replica's time to first byte.
* The whole request body is buffered in memory (limit `--max-body-mb`, default 64) because the router reads the messages and may retry.
* One router instance: no high availability, static replica list, no service discovery, no TLS or authentication (plain HTTP to localhost).
* Tested with mock replicas and, in `../README.md`, with vLLM replicas on one GPU. Not tested at scale or across machines.
