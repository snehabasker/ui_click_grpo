### `trtllm_bench_20261030-211848.json`

One GPU, bf16, serving/traces/single (one image per request, 120 requests, no shared prefix), prefix caching/reuse off on both engines, max_num_seqs 32, max_model_len 4096. TensorRT-LLM 1.2.1 (pytorch backend) with torch 2.9.1+cu130, transformers 4.57.3; vLLM 0.32.0 (from the Phase 1 engine record). Median (min–max) over 3 rounds; engine order per round: round 1: vllm then trtllm; round 2: trtllm then vllm; round 3: vllm then trtllm.

TensorRT-LLM local patch to the installed package (vision-tower attention):

```
388c388
<             self.config._attn_implementation = 'flash_attention_2'
---
>             self.config._attn_implementation = 'sdpa'
```

**c=1**

| engine | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| vllm | 0.23 (0.23–0.23) | 0.24 (0.24–0.25) | 0.36 (0.36–0.36) | 2.77 (2.77–2.78) | 0% (0%–0%) | 0.0% (0.0%–0.0%) | 71 (70–71) | 69 (69–70) |
| trtllm | 0.42 (0.40–0.43) | 0.48 (0.48–0.58) | 0.53 (0.51–0.55) | 1.88 (1.80–1.91) | 0% (0%–0%) | 0.0% (0.0%–0.0%) | 71 (69–71) | 59 (57–60) |

**c=4**

| engine | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| vllm | 0.88 (0.87–0.88) | 0.92 (0.91–0.92) | 1.04 (1.03–1.04) | 3.86 (3.86–3.87) | 0% (0%–0%) | 0.0% (0.0%–0.0%) | 75 (75–75) | 69 (69–69) |
| trtllm | 1.28 (1.26–1.29) | 1.54 (1.42–1.56) | 1.48 (1.47–1.50) | 2.70 (2.66–2.70) | 0% (0%–0%) | 0.0% (0.0%–0.0%) | 73 (72–73) | 63 (63–63) |

**c=16**

| engine | TTFT p50 (s) | TTFT p99 (s) | e2e p50 (s) | req/s | prompt tokens from cache | errors | peak temp (C) | median power (W) |
|---|---|---|---|---|---|---|---|---|
| vllm | 1.79 (1.75–1.80) | 3.58 (3.57–3.65) | 3.72 (3.72–3.73) | 4.28 (4.27–4.29) | 0% (0%–0%) | 0.0% (0.0%–0.0%) | 75 (75–75) | 70 (70–70) |
| trtllm | 4.68 (4.67–4.76) | 5.56 (5.56–5.59) | 5.49 (5.48–5.55) | 2.89 (2.87–2.90) | 0% (0%–0%) | 0.0% (0.0%–0.0%) | 73 (72–73) | 63 (62–63) |

## Click accuracy: TensorRT-LLM vs vLLM bf16

| set | n | vLLM bf16 (Phase 1) | TensorRT-LLM | delta (TensorRT-LLM - vLLM) | only vLLM right / only TensorRT-LLM right | sign-test p |
|---|---|---|---|---|---|---|
| synth300 | 300 | 64.33% | 62.67% | -1.67 pts | 11 / 6 | 0.3323 |
| screenspot400 | 400 | 53.00% | 57.75% | +4.75 pts | 12 / 31 | 0.0054 |

## Prompt tokens each engine reports vs the Hugging Face reference

| engine | set | items | prompt tokens equal to the Hugging Face reference | smallest difference | largest difference |
|---|---|---|---|---|---|
| trtllm | screenspot400 | 20 | 0 | +1 | +1 |
| trtllm | synth300 | 20 | 0 | +1 | +1 |
| vllm | screenspot400 | 20 | 20 | +0 | +0 |
| vllm | synth300 | 20 | 20 | +0 | +0 |
