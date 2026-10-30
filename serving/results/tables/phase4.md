### `disagg-bigbudget_bench_20261030-191355.json`

Prefill-instance extra flags: `--no-enable-chunked-prefill --max-num-batched-tokens=16384`.

One GPU. Setups: **colocated** = one vLLM instance, FP8 weights, 0.8 GiB KV cache; **split** = a prefill instance and a decode instance (FP8, 0.4 GiB KV each) connected by `ExampleConnector (KV through files on disk)`, behind the prefill/decode proxy. max_num_seqs 8, max_model_len 3072. Median (min–max) over 3 rounds; order of the setups per round: round 1: split; round 2: split; round 3: split.

**c=16**

| setup | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) | prefill step p50 (s) | KV store per request (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| split | 29.60 (21.48–30.32) | 40.30 (33.46–47.63) | 29.72 (21.59–30.45) | 0.61 (0.53–0.81) | 96% (96%–96%) | 0.2% (0.0%–0.4%) | 66 (66–67) | 27 (27–29) | 24.514 (17.606–24.664) | 49.4 (49.3–49.5) |

### `disagg_bench_20261030-173211.json`

Prefill-instance extra flags: `none (vLLM defaults, chunked prefill on)`.

One GPU. Setups: **colocated** = one vLLM instance, FP8 weights, 0.8 GiB KV cache; **split** = a prefill instance and a decode instance (FP8, 0.4 GiB KV each) connected by `ExampleConnector (KV through files on disk)`, behind the prefill/decode proxy. max_num_seqs 8, max_model_len 3072. Median (min–max) over 3 rounds; order of the setups per round: round 1: colocated then split; round 2: split then colocated; round 3: colocated then split.

**c=1**

| setup | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) | prefill step p50 (s) | KV store per request (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| colocated | 0.16 (0.15–0.25) | 0.42 (0.41–0.67) | 0.23 (0.22–0.35) | 3.88 (2.37–4.05) | 64% (64%–64%) | 0.0% (0.0%–0.0%) | 70 (51–72) | 65 (27–68) | – | – |
| split | 0.28 (0.27–0.39) | 1.08 (1.05–1.19) | 0.35 (0.34–0.48) | 1.77 (1.76–1.88) | 97% (97%–97%) | 0.0% (0.0%–0.0%) | 65 (55–67) | 46 (27–47) | 0.219 (0.214–0.302) | 49.6 (49.6–49.6) |

**c=4**

| setup | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) | prefill step p50 (s) | KV store per request (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| colocated | 0.56 (0.55–0.89) | 1.09 (1.08–1.69) | 0.65 (0.64–1.04) | 5.47 (3.39–5.58) | 65% (65%–65%) | 0.0% (0.0%–0.0%) | 77 (54–79) | 68 (28–69) | – | – |
| split | 1.74 (1.37–3.35) | 1.84 (1.45–3.44) | 2.01 (1.58–3.57) | 0.02 (0.02–0.02) | 100% (100%–100%) | 99.2% (99.2%–99.2%) | 70 (57–70) | 42 (26–44) | 0.886 (0.872–1.140) | 48.9 (48.8–49.1) |

**c=16**

| setup | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) | prefill step p50 (s) | KV store per request (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| colocated | 1.55 (1.54–2.44) | 3.14 (3.11–5.33) | 2.39 (2.37–3.86) | 6.29 (3.94–6.32) | 67% (67%–67%) | 0.0% (0.0%–0.0%) | 81 (59–81) | 69 (28–70) | – | – |
| split | – | – | – | 0.00 (0.00–0.00) | – | 100.0% (100.0%–100.0%) | 78 (77–78) | 61 (60–62) | 4.575 (4.496–4.592) | 38.4 (37.9–39.1) |

### `disagg_bench_20261030-182042.json`

Prefill-instance extra flags: `--no-enable-chunked-prefill --max-num-batched-tokens=4096`.

One GPU. Setups: **colocated** = one vLLM instance, FP8 weights, 0.8 GiB KV cache; **split** = a prefill instance and a decode instance (FP8, 0.4 GiB KV each) connected by `ExampleConnector (KV through files on disk)`, behind the prefill/decode proxy. max_num_seqs 8, max_model_len 3072. Median (min–max) over 3 rounds; order of the setups per round: round 1: colocated then split; round 2: split then colocated; round 3: colocated then split.

**c=1**

| setup | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) | prefill step p50 (s) | KV store per request (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| colocated | 0.15 (0.14–0.15) | 0.41 (0.40–0.41) | 0.22 (0.21–0.22) | 4.07 (4.05–4.11) | 64% (64%–64%) | 0.0% (0.0%–0.0%) | 69 (69–70) | 68 (68–68) | – | – |
| split | 0.26 (0.25–0.26) | 0.74 (0.73–0.76) | 0.33 (0.32–0.33) | 2.68 (2.67–2.69) | 97% (97%–97%) | 0.0% (0.0%–0.0%) | 68 (67–68) | 54 (53–54) | 0.201 (0.199–0.205) | 49.6 (49.6–49.6) |

**c=4**

| setup | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) | prefill step p50 (s) | KV store per request (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| colocated | 0.54 (0.54–0.54) | 1.08 (1.07–1.08) | 0.63 (0.62–0.63) | 5.63 (5.58–5.65) | 65% (65%–65%) | 0.0% (0.0%–0.0%) | 76 (75–76) | 69 (69–69) | – | – |
| split | 0.95 (0.95–1.00) | 1.97 (1.94–1.99) | 1.04 (1.03–1.09) | 3.55 (3.45–3.57) | 98% (98%–98%) | 0.0% (0.0%–0.0%) | 74 (74–74) | 54 (54–55) | 0.771 (0.759–0.803) | 49.5 (49.5–49.5) |

**c=16**

| setup | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) | prefill step p50 (s) | KV store per request (MiB) |
|---|---|---|---|---|---|---|---|---|---|---|
| colocated | 1.53 (1.52–1.53) | 3.07 (3.01–3.09) | 2.36 (2.36–2.36) | 6.38 (6.37–6.39) | 67% (67%–67%) | 0.0% (0.0%–0.0%) | 79 (78–79) | 70 (70–70) | – | – |
| split | 3.63 (3.49–3.69) | 4.02 (3.70–4.17) | 3.75 (3.61–3.86) | 0.03 (0.03–0.03) | 80% (80%–80%) | 99.1% (99.1%–99.1%) | 78 (78–79) | 62 (60–63) | 4.557 (4.360–4.702) | 46.7 (46.5–46.7) |

## Click accuracy through the split path vs one colocated instance

| set | n | colocated acc | split acc | delta (split - colocated) | only colocated right / only split right | sign-test p |
|---|---|---|---|---|---|---|
| synth300 | 300 | 64.67% | 64.67% | +0.00 pts | 1 / 1 | 1.000 |
| screenspot400 | 400 | 49.75% | 49.50% | -0.25 pts | 5 / 4 | 1.000 |
