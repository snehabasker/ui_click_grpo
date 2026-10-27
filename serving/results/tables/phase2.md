## Prefix-aware vs round-robin routing

### Run order: prefix then round-robin  (`router_bench_20261027-095936.json`)

Two FP8 replicas, 0.4 GiB KV cache each (14,976 tokens), max_num_seqs 8, max_model_len 3072. Median (min–max) over 3 repeats; caches flushed before every repeat.

**c=4**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 0.77 (0.74–0.77) | 2.15 (2.08–2.17) | 1.45 (1.35–1.47) | 2.95 (2.94–2.99) | 63% (63%–63%) | 0.0% (0.0%–0.0%) | 54 (54–55) | 31 (31–31) |
| round-robin | 0.64 (0.62–0.64) | 1.75 (1.72–1.75) | 0.97 (0.97–1.00) | 3.85 (3.68–3.86) | 57% (57%–58%) | 0.0% (0.0%–0.0%) | 76 (75–76) | 65 (65–65) |

**c=16**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 2.29 (2.18–2.31) | 6.06 (5.89–6.27) | 4.28 (4.06–4.44) | 3.62 (3.47–3.78) | 65% (65%–65%) | 0.0% (0.0%–0.0%) | 58 (57–59) | 29 (28–29) |
| round-robin | 1.59 (1.48–1.66) | 5.80 (5.30–6.52) | 3.13 (3.11–3.24) | 3.85 (3.78–3.96) | 50% (50%–50%) | 0.0% (0.0%–0.0%) | 78 (78–79) | 67 (66–67) |

**c=32**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 7.39 (4.69–7.75) | 11.86 (8.35–13.34) | 11.78 (7.03–12.07) | 2.83 (2.66–4.28) | 51% (51%–51%) | 0.0% (0.0%–0.0%) | 71 (57–78) | 28 (28–67) |
| round-robin | 5.00 (4.15–5.84) | 15.27 (11.36–17.17) | 7.73 (6.89–9.30) | 3.45 (3.26–3.52) | 48% (46%–49%) | 0.0% (0.0%–0.0%) | 78 (75–78) | 64 (63–64) |

Router decisions (all repeats of the cell):

| policy | c | affinity | spill | round-robin |
|---|---|---|---|---|
| prefix | 4 | 716 | 133 | 0 |
| prefix | 16 | 1441 | 299 | 0 |
| prefix | 32 | 1537 | 203 | 0 |
| round-robin | 4 | 0 | 0 | 849 |
| round-robin | 16 | 0 | 0 | 1740 |
| round-robin | 32 | 0 | 0 | 1740 |

### Run order: round-robin then prefix  (`router_bench_20261027-105136.json`)

Two FP8 replicas, 0.4 GiB KV cache each (14,976 tokens), max_num_seqs 8, max_model_len 3072. Median (min–max) over 3 repeats; caches flushed before every repeat.

**c=4**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 0.27 (0.26–0.27) | 1.59 (1.58–1.72) | 0.49 (0.49–0.50) | 4.64 (4.63–4.64) | 63% (63%–63%) | 0.0% (0.0%–0.0%) | 78 (76–78) | 69 (69–69) |
| round-robin | 0.34 (0.33–0.35) | 1.43 (1.27–1.53) | 0.75 (0.75–0.76) | 4.32 (4.27–4.35) | 58% (57%–58%) | 0.0% (0.0%–0.0%) | 77 (74–77) | 70 (69–70) |

**c=16**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 1.64 (1.64–1.67) | 3.88 (3.85–4.26) | 2.35 (2.32–2.36) | 5.87 (5.82–5.94) | 65% (65%–65%) | 0.0% (0.0%–0.0%) | 74 (74–75) | 69 (69–69) |
| round-robin | 1.49 (1.45–1.58) | 5.93 (4.78–6.02) | 3.20 (3.19–3.44) | 4.24 (4.13–4.56) | 51% (50%–53%) | 0.0% (0.0%–0.0%) | 77 (75–79) | 69 (69–69) |

**c=32**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 4.41 (4.30–4.43) | 7.65 (7.62–7.76) | 6.70 (6.63–6.74) | 4.58 (4.56–4.61) | 50% (50%–51%) | 0.0% (0.0%–0.0%) | 74 (73–74) | 69 (69–70) |
| round-robin | 3.70 (3.70–4.34) | 13.08 (12.40–14.11) | 5.98 (5.95–6.71) | 4.21 (4.07–4.27) | 49% (47%–50%) | 0.0% (0.0%–0.0%) | 77 (77–78) | 69 (69–69) |

Router decisions (all repeats of the cell):

| policy | c | affinity | spill | round-robin |
|---|---|---|---|---|
| prefix | 4 | 709 | 140 | 0 |
| prefix | 16 | 1448 | 292 | 0 |
| prefix | 32 | 1543 | 197 | 0 |
| round-robin | 4 | 0 | 0 | 849 |
| round-robin | 16 | 0 | 0 | 1740 |
| round-robin | 32 | 0 | 0 | 1740 |

### Run order: interleaved, prefix / round-robin alternating first  (`router_bench_20261027-114352.json`)

Two FP8 replicas, 0.4 GiB KV cache each (14,976 tokens), max_num_seqs 8, max_model_len 3072. Median (min–max) over 3 repeats; caches flushed before every repeat.

**c=4**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 0.45 (0.40–0.48) | 1.52 (1.39–1.57) | 1.03 (1.00–1.04) | 4.13 (4.11–4.22) | 63% (62%–63%) | 0.0% (0.0%–0.0%) | 88 (87–88) | 56 (55–60) |
| round-robin | 0.42 (0.40–0.43) | 1.49 (1.35–1.76) | 0.84 (0.83–0.85) | 3.89 (3.86–3.98) | 58% (57%–59%) | 0.0% (0.0%–0.0%) | 88 (87–89) | 55 (55–57) |

**c=16**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 1.96 (1.82–2.00) | 4.51 (4.25–4.53) | 2.66 (2.64–2.76) | 5.01 (4.99–5.07) | 65% (65%–65%) | 0.0% (0.0%–0.0%) | 90 (90–91) | 54 (53–54) |
| round-robin | 1.70 (1.66–1.80) | 5.55 (5.41–6.49) | 3.68 (3.61–3.89) | 3.77 (3.61–3.94) | 51% (51%–53%) | 0.0% (0.0%–0.0%) | 90 (90–91) | 55 (54–55) |

**c=32**

| policy | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| prefix | 4.98 (4.85–5.03) | 8.45 (8.35–8.45) | 8.33 (8.08–8.36) | 3.90 (3.89–3.97) | 51% (51%–51%) | 0.0% (0.0%–0.0%) | 90 (89–91) | 54 (53–54) |
| round-robin | 4.42 (3.70–5.31) | 15.64 (8.09–15.98) | 6.71 (6.12–8.64) | 3.47 (3.47–3.72) | 48% (46%–49%) | 0.0% (0.0%–0.0%) | 90 (89–90) | 55 (54–57) |

Router decisions (all repeats of the cell):

| policy | c | affinity | spill | round-robin |
|---|---|---|---|---|
| prefix | 4 | 722 | 127 | 0 |
| prefix | 16 | 1433 | 307 | 0 |
| prefix | 32 | 1548 | 192 | 0 |
| round-robin | 4 | 0 | 0 | 849 |
| round-robin | 16 | 0 | 0 | 1740 |
| round-robin | 32 | 0 | 0 | 1740 |

## Failure test

### `router_failure_20261027-112714.json`

```
{
 "kind": "router_failure",
 "router_binary_sha256": "b3b31b527ab327df8f79323e31050718df40769fb51265a78d7d23819124ee24",
 "router_version": "prefix-router 0.1.0",
 "replicas": [
  {
   "log": "router-failure-replica0-20261027-112714.log",
   "kv_cache_tokens": 14976,
   "model_gib": 2.38
  },
  {
   "log": "router-failure-restart-replica1-20261027-112714.log",
   "kv_cache_tokens": 14976,
   "model_gib": 2.38
  }
 ],
 "replica_settings": {
  "quantization": "fp8",
  "gpu_util_each": 0.4,
  "kv_cache_gib_each": 0.4,
  "max_seqs": 8,
  "max_len": 3072
 },
 "gpu_mem_loaded_mib": 6935,
 "events": {
  "kill_replica1_at": 60.035,
  "restart_launched_at": 90.164,
  "restarted_replica_ready_at": 146.205,
  "router_marked_unhealthy_at": 60.669,
  "router_marked_healthy_again_at": 146.857,
  "first_request_served_by_restarted_replica_at": null,
  "detect_seconds": 0.634,
  "healthy_again_seconds_after_ready": 0.652,
  "first_request_seconds_after_ready": null
 },
 "loadtest_stdout_last_line": "[router-failure c=4 r1] ok=557/560 ttft p50=0.5458242780005094 e2e p50=0.6527348100007657 req/s=5.12 acc=0.5583482944344704 -> loadtest_router-failure_c4_r1_20261027-113107.json"
}
```

### `router_failure_20261027-113934.json`

```
{
 "kind": "router_failure",
 "router_binary_sha256": "7efb9be5c46a766312b71f6ee2e90570d490bfdb82c4338b99d2b57e67670dfd",
 "router_version": "prefix-router 0.1.0",
 "replicas": [
  {
   "log": "router-failure-replica0-20261027-113934.log",
   "kv_cache_tokens": 14976,
   "model_gib": 2.38
  },
  {
   "log": "router-failure-restart-replica1-20261027-113934.log",
   "kv_cache_tokens": 14976,
   "model_gib": 2.38
  }
 ],
 "replica_settings": {
  "quantization": "fp8",
  "gpu_util_each": 0.4,
  "kv_cache_gib_each": 0.4,
  "max_seqs": 8,
  "max_len": 3072
 },
 "gpu_mem_loaded_mib": 7479,
 "events": {
  "kill_replica1_at": 25.024,
  "restart_launched_at": 27.095,
  "restarted_replica_ready_at": 85.129,
  "router_marked_unhealthy_at": 25.947,
  "router_marked_healthy_again_at": 85.169,
  "first_request_served_by_restarted_replica_at": 86.29,
  "detect_seconds": 0.923,
  "healthy_again_seconds_after_ready": 0.04,
  "first_request_seconds_after_ready": 1.161
 },
 "loadtest_stdout_last_line": "[router-failure c=4 r1] ok=559/560 ttft p50=0.5929540679999263 e2e p50=0.709089099999801 req/s=4.96 acc=0.5581395348837209 -> loadtest_router-failure_c4_r1_20261027-114332.json"
}
```
