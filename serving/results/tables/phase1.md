## Click accuracy through each engine

| run | synthetic held-out (n=300) | ScreenSpot subset (n=400) |
|---|---|---|
| HF reference (merged checkpoint) | 63.33% [57.7, 68.6], valid 100.0% | 53.75% [48.9, 58.6], valid 98.2% |
| sglang-flashinfer-bs32-prefix-on | 64.00% [58.4, 69.2], valid 100.0% | 53.50% [48.6, 58.3], valid 98.8% |
| sglang-prefix-off | 63.67% [58.1, 68.9], valid 100.0% | – |
| sglang-prefix-on | 64.00% [58.4, 69.2], valid 100.0% | 53.75% [48.9, 58.6], valid 98.5% |
| sglang-triton-bs32-prefix-on | 64.00% [58.4, 69.2], valid 100.0% | – |
| vllm-prefix-off | 63.67% [58.1, 68.9], valid 100.0% | – |
| vllm-prefix-on | 64.33% [58.8, 69.5], valid 100.0% | 53.00% [48.1, 57.8], valid 97.5% |

## Engine settings recorded at launch

| run | engine | PyTorch | attention backend (log) | startup | KV cache (tokens) | GPU memory after load |
|---|---|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | sglang | – | flashinfer (set by flag) | **FAILED to start** after 18 s (cause in NOTES.md) | – | – |
| sglang-flashinfer-bs32-prefix-on | sglang | – | flashinfer (set by flag) | **FAILED to start** after 63 s (cause in NOTES.md) | – | – |
| sglang-flashinfer-bs32-prefix-on | sglang | – | flashinfer (set by flag) | **FAILED to start** after 84 s (cause in NOTES.md) | – | – |
| sglang-flashinfer-bs32-prefix-on | sglang 0.5.22 | torch 2.14.1+cu132 | flashinfer (set by flag) | 54 s | 50728 | 6885 MiB |
| sglang-prefix-off | sglang 0.5.22 | torch 2.14.1+cu132 | triton (set by flag) | 66 s | 50728 | 6359 MiB |
| sglang-prefix-on | sglang 0.5.22 | torch 2.14.1+cu132 | triton (set by flag) | 39 s | 50728 | 6359 MiB |
| sglang-triton-bs32-prefix-on | sglang | – | triton (set by flag) | **FAILED to start** after 9 s (cause in NOTES.md) | – | – |
| sglang-triton-bs32-prefix-on | sglang 0.5.22 | torch 2.14.1+cu132 | triton (set by flag) | 48 s | 50728 | 6381 MiB |
| vllm-prefix-off | vllm 0.32.0 | torch 2.13.0+cu132 | FLASH_ATTN | 90 s | 57168 | 6438 MiB |
| vllm-prefix-on-rerun | vllm 0.32.0 | torch 2.13.0+cu132 | FLASH_ATTN | 75 s | 57168 | 6438 MiB |
| vllm-prefix-on | vllm 0.32.0 | torch 2.13.0+cu132 | FLASH_ATTN | 60 s | 57168 | 6438 MiB |

## Load tests

Median (min–max) over repeats; repeats per cell: 3. Columns are client concurrency.

### Time to first token, p50 (s)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 0.39 (0.38–0.51) | 1.80 (1.79–1.85) | 10.15 (10.07–10.85) | 18.63 (17.91–18.72) |
| sglang-prefix-off | 0.62 (0.61–0.63) | 2.00 (1.95–2.00) | 5.57 (5.44–5.69) | 13.86 (13.10–13.93) |
| sglang-prefix-on | 0.38 (0.38–0.38) | 1.82 (1.82–1.83) | 8.63 (7.74–8.70) | 14.35 (14.20–15.89) |
| sglang-triton-bs32-prefix-on |  |  | 9.91 (9.42–10.14) | 12.03 (11.57–16.06) |
| vllm-prefix-off | 0.71 (0.68–0.74) | 1.50 (1.45–1.93) | 3.21 (3.18–3.21) | 13.88 (13.49–13.90) |
| vllm-prefix-on | 0.27 (0.26–0.27) | 1.00 (1.00–1.00) | 2.21 (2.11–2.22) | 2.03 (2.01–2.04) |
| vllm-prefix-on-rerun |  |  | 3.03 (2.89–3.11) |  |

### Time to first token, p99 (s)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 1.52 (1.45–1.71) | 4.10 (3.99–4.12) | 20.78 (17.94–20.84) | 47.18 (46.22–54.11) |
| sglang-prefix-off | 1.60 (1.59–1.65) | 4.04 (4.01–4.07) | 11.49 (11.40–11.96) | 27.33 (26.18–29.09) |
| sglang-prefix-on | 1.55 (1.53–1.56) | 4.06 (4.04–4.17) | 16.15 (13.55–16.59) | 33.90 (31.03–35.10) |
| sglang-triton-bs32-prefix-on |  |  | 18.03 (17.78–18.47) | 36.80 (27.29–36.80) |
| vllm-prefix-off | 1.26 (1.18–1.38) | 3.13 (2.91–4.11) | 7.54 (7.50–8.06) | 17.47 (17.23–18.46) |
| vllm-prefix-on | 0.78 (0.77–0.78) | 2.03 (2.03–2.05) | 6.18 (6.17–6.20) | 11.52 (11.24–11.55) |
| vllm-prefix-on-rerun |  |  | 6.96 (6.89–6.99) |  |

### Inter-token latency, p50 (ms)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 12.8 (12.2–12.8) | 13.5 (13.4–14.5) | 14.4 (14.3–14.5) | 16.6 (16.4–16.6) |
| sglang-prefix-off | 13.6 (13.3–14.0) | 14.5 (14.3–14.7) | 32.4 (32.3–33.0) | 33.3 (32.6–34.0) |
| sglang-prefix-on | 12.5 (12.4–12.5) | 13.0 (12.9–13.0) | 51.4 (30.9–52.8) | 57.3 (55.8–58.0) |
| sglang-triton-bs32-prefix-on |  |  | 17.7 (17.6–17.9) | 18.7 (18.1–23.7) |
| vllm-prefix-off | 14.9 (14.5–15.1) | 14.1 (13.9–15.8) | 768.0 (749.8–778.2) | 782.5 (757.2–786.5) |
| vllm-prefix-on | 13.0 (12.0–13.1) | 13.9 (13.9–13.9) | 17.6 (17.4–17.7) | 577.0 (536.5–676.8) |
| vllm-prefix-on-rerun |  |  | 20.6 (20.1–20.8) |  |

### End-to-end latency, p50 (s)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 0.52 (0.52–0.63) | 1.96 (1.96–2.02) | 12.61 (11.74–13.24) | 29.75 (23.80–29.93) |
| sglang-prefix-off | 0.75 (0.75–0.77) | 2.45 (2.39–2.45) | 9.46 (9.41–9.60) | 21.15 (20.41–21.25) |
| sglang-prefix-on | 0.51 (0.51–0.52) | 1.99 (1.98–2.00) | 10.71 (9.80–10.76) | 23.11 (21.83–24.31) |
| sglang-triton-bs32-prefix-on |  |  | 11.62 (11.34–12.10) | 18.29 (17.77–25.82) |
| vllm-prefix-off | 0.85 (0.82–0.89) | 2.31 (2.24–3.06) | 10.65 (10.65–10.66) | 21.37 (20.79–21.44) |
| vllm-prefix-on | 0.40 (0.38–0.40) | 1.17 (1.15–1.17) | 4.26 (4.20–4.27) | 8.46 (8.44–8.47) |
| vllm-prefix-on-rerun |  |  | 4.87 (4.83–4.87) |  |

### End-to-end latency, p99 (s)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 1.67 (1.61–1.84) | 4.38 (4.17–4.94) | 22.56 (18.47–22.78) | 59.60 (52.82–61.89) |
| sglang-prefix-off | 1.75 (1.74–1.80) | 4.51 (4.37–4.59) | 13.08 (13.04–13.58) | 46.12 (42.83–46.20) |
| sglang-prefix-on | 1.69 (1.66–1.70) | 4.26 (4.23–5.08) | 18.59 (15.71–19.15) | 46.87 (44.38–49.44) |
| sglang-triton-bs32-prefix-on |  |  | 20.13 (18.98–20.21) | 39.10 (37.67–52.40) |
| vllm-prefix-off | 1.41 (1.33–1.55) | 3.34 (3.19–4.36) | 14.54 (14.41–15.27) | 25.16 (24.95–26.60) |
| vllm-prefix-on | 0.91 (0.90–0.91) | 2.62 (2.61–2.65) | 8.85 (8.83–8.86) | 18.29 (17.91–18.34) |
| vllm-prefix-on-rerun |  |  | 8.35 (8.22–8.39) |  |

### Throughput (requests/s)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 1.67 (1.38–1.69) | 1.76 (1.74–1.77) | 1.15 (1.10–1.17) | 1.09 (1.06–1.13) |
| sglang-prefix-off | 1.24 (1.22–1.26) | 1.50 (1.50–1.54) | 1.60 (1.56–1.60) | 1.28 (1.28–1.35) |
| sglang-prefix-on | 1.69 (1.68–1.70) | 1.75 (1.75–1.77) | 1.40 (1.40–1.58) | 1.29 (1.22–1.37) |
| sglang-triton-bs32-prefix-on |  |  | 1.28 (1.24–1.33) | 1.57 (1.18–1.64) |
| vllm-prefix-off | 1.07 (1.04–1.14) | 1.58 (1.27–1.68) | 1.49 (1.49–1.50) | 1.49 (1.48–1.53) |
| vllm-prefix-on | 2.32 (2.31–2.35) | 3.10 (3.07–3.10) | 3.52 (3.48–3.55) | 3.59 (3.58–3.60) |
| vllm-prefix-on-rerun |  |  | 3.07 (3.06–3.09) |  |

### Prompt tokens served from the prefix cache (engine-reported)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 66% (66%–66%) | 66% (66%–66%) | 67% (67%–67%) | 67% (67%–67%) |
| sglang-prefix-off | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) |
| sglang-prefix-on | 66% (66%–66%) | 66% (66%–66%) | 67% (67%–67%) | 67% (67%–67%) |
| sglang-triton-bs32-prefix-on |  |  | 67% (67%–67%) | 67% (67%–67%) |
| vllm-prefix-off | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) | 0% (0%–0%) |
| vllm-prefix-on | 66% (66%–66%) | 66% (66%–66%) | 67% (67%–67%) | 67% (67%–67%) |
| vllm-prefix-on-rerun |  |  | 67% (67%–67%) |  |

### Click accuracy on the trace requests

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 57.4% (57.4%–57.4%) | 57.4% (57.0%–57.4%) | 57.1% (56.6%–57.3%) | 56.6% (56.4%–57.0%) |
| sglang-prefix-off | 58.9% (58.9%–58.9%) | 56.3% (55.9%–57.0%) | 57.0% (56.8%–57.1%) | 57.5% (57.0%–57.7%) |
| sglang-prefix-on | 58.9% (58.9%–58.9%) | 55.9% (55.5%–55.9%) | 56.4% (56.2%–56.6%) | 56.6% (56.4%–57.5%) |
| sglang-triton-bs32-prefix-on |  |  | 56.6% (56.4%–57.0%) | 56.8% (56.6%–56.8%) |
| vllm-prefix-off | 57.4% (57.4%–57.4%) | 55.9% (55.9%–55.9%) | 55.9% (55.7%–55.9%) | 56.1% (55.7%–56.6%) |
| vllm-prefix-on | 57.4% (57.4%–57.4%) | 55.9% (55.9%–56.3%) | 55.9% (55.7%–55.9%) | 56.4% (56.4%–56.6%) |
| vllm-prefix-on-rerun |  |  | 55.9% (55.9%–56.1%) |  |

### Request error rate

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) |
| sglang-prefix-off | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) |
| sglang-prefix-on | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) |
| sglang-triton-bs32-prefix-on |  |  | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) |
| vllm-prefix-off | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) |
| vllm-prefix-on | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) | 0.0% (0.0%–0.0%) |
| vllm-prefix-on-rerun |  |  | 0.0% (0.0%–0.0%) |  |

### Peak GPU memory (MiB)

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 7493 (7435–7493) | 7699 (7695–7703) | 7855 (7853–7881) | 7843 (7835–7855) |
| sglang-prefix-off | 7171 (7171–7171) | 7313 (7241–7313) | 7313 (7313–7313) | 7253 (7253–7432) |
| sglang-prefix-on | 6971 (6913–6973) | 7167 (7167–7179) | 7449 (7427–7465) | 7437 (7437–7449) |
| sglang-triton-bs32-prefix-on |  |  | 7465 (7461–7471) | 7457 (7357–7469) |
| vllm-prefix-off | 7090 (7090–7090) | 7162 (7160–7162) | 7162 (7162–7162) | 7162 (7162–7162) |
| vllm-prefix-on | 7366 (7366–7366) | 7366 (7366–7366) | 7368 (7368–7368) | 7368 (7368–7368) |
| vllm-prefix-on-rerun |  |  | 7248 (7248–7250) |  |

### Peak GPU temperature (C), runs with the extended monitor

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 90 (87–91) | 91 (90–92) | 90 (90–91) | 87 (69–89) |
| sglang-prefix-off | – | – | – | – |
| sglang-prefix-on | – | – | – | – |
| sglang-triton-bs32-prefix-on |  |  | 79 (75–79) | 90 (73–90) |
| vllm-prefix-off | – | – | – | – |
| vllm-prefix-on | – | – | – | – |
| vllm-prefix-on-rerun |  |  | 79 (76–79) |  |

### Median GPU power draw (W), runs with the extended monitor

| run | c=1 | c=4 | c=16 | c=32 |
|---|---|---|---|---|
| sglang-flashinfer-bs32-prefix-on | 54.7 (53.0–55.6) | 51.9 (50.3–52.3) | 51.6 (49.4–52.2) | 49.9 (49.9–54.0) |
| sglang-prefix-off | – | – | – | – |
| sglang-prefix-on | – | – | – | – |
| sglang-triton-bs32-prefix-on |  |  | 31.6 (30.4–34.0) | 51.6 (28.4–55.0) |
| vllm-prefix-off | – | – | – | – |
| vllm-prefix-on | – | – | – | – |
| vllm-prefix-on-rerun |  |  | 39.8 (39.7–39.8) |  |
