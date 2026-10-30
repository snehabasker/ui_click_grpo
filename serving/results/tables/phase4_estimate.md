**ESTIMATE, not a measurement.** Lower bounds at each link's theoretical line rate.

KV cache per token: 2 (K, V) x 28 layers x 2 KV heads x 128 dims x dtype bytes = 28,672 bytes (bf16), 14,336 bytes (fp8).

Reference for scale: the measured median time to first token of vLLM with prefix caching off at 1 client in Phase 1 was 711 ms (all turns of the trace, so it mixes prompt sizes).

| prompt | tokens | KV dtype | KV size (MiB) | link | line rate (GB/s) | time at line rate (ms) | as a share of the reference TTFT |
|---|---|---|---|---|---|---|---|
| shared prefix (system + 2 few-shot screenshots) | 1254 | bf16 | 34.3 | PCIe 3.0 x16 | 15.75 | 2.3 | 0.3% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | bf16 | 34.3 | PCIe 4.0 x16 | 31.51 | 1.1 | 0.2% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | bf16 | 34.3 | PCIe 5.0 x16 | 63.02 | 0.6 | 0.1% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | bf16 | 34.3 | 10 GbE | 1.25 | 28.8 | 4.0% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | bf16 | 34.3 | 25 GbE | 3.12 | 11.5 | 1.6% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | bf16 | 34.3 | 100 GbE | 12.50 | 2.9 | 0.4% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | fp8 | 17.1 | PCIe 3.0 x16 | 15.75 | 1.1 | 0.2% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | fp8 | 17.1 | PCIe 4.0 x16 | 31.51 | 0.6 | 0.1% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | fp8 | 17.1 | PCIe 5.0 x16 | 63.02 | 0.3 | 0.0% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | fp8 | 17.1 | 10 GbE | 1.25 | 14.4 | 2.0% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | fp8 | 17.1 | 25 GbE | 3.12 | 5.8 | 0.8% |
| shared prefix (system + 2 few-shot screenshots) | 1254 | fp8 | 17.1 | 100 GbE | 12.50 | 1.4 | 0.2% |
| first turn of a session | 1666 | bf16 | 45.6 | PCIe 3.0 x16 | 15.75 | 3.0 | 0.4% |
| first turn of a session | 1666 | bf16 | 45.6 | PCIe 4.0 x16 | 31.51 | 1.5 | 0.2% |
| first turn of a session | 1666 | bf16 | 45.6 | PCIe 5.0 x16 | 63.02 | 0.8 | 0.1% |
| first turn of a session | 1666 | bf16 | 45.6 | 10 GbE | 1.25 | 38.2 | 5.4% |
| first turn of a session | 1666 | bf16 | 45.6 | 25 GbE | 3.12 | 15.3 | 2.1% |
| first turn of a session | 1666 | bf16 | 45.6 | 100 GbE | 12.50 | 3.8 | 0.5% |
| first turn of a session | 1666 | fp8 | 22.8 | PCIe 3.0 x16 | 15.75 | 1.5 | 0.2% |
| first turn of a session | 1666 | fp8 | 22.8 | PCIe 4.0 x16 | 31.51 | 0.8 | 0.1% |
| first turn of a session | 1666 | fp8 | 22.8 | PCIe 5.0 x16 | 63.02 | 0.4 | 0.1% |
| first turn of a session | 1666 | fp8 | 22.8 | 10 GbE | 1.25 | 19.1 | 2.7% |
| first turn of a session | 1666 | fp8 | 22.8 | 25 GbE | 3.12 | 7.6 | 1.1% |
| first turn of a session | 1666 | fp8 | 22.8 | 100 GbE | 12.50 | 1.9 | 0.3% |
| last measured turn | 1861 | bf16 | 50.9 | PCIe 3.0 x16 | 15.75 | 3.4 | 0.5% |
| last measured turn | 1861 | bf16 | 50.9 | PCIe 4.0 x16 | 31.51 | 1.7 | 0.2% |
| last measured turn | 1861 | bf16 | 50.9 | PCIe 5.0 x16 | 63.02 | 0.8 | 0.1% |
| last measured turn | 1861 | bf16 | 50.9 | 10 GbE | 1.25 | 42.7 | 6.0% |
| last measured turn | 1861 | bf16 | 50.9 | 25 GbE | 3.12 | 17.1 | 2.4% |
| last measured turn | 1861 | bf16 | 50.9 | 100 GbE | 12.50 | 4.3 | 0.6% |
| last measured turn | 1861 | fp8 | 25.4 | PCIe 3.0 x16 | 15.75 | 1.7 | 0.2% |
| last measured turn | 1861 | fp8 | 25.4 | PCIe 4.0 x16 | 31.51 | 0.8 | 0.1% |
| last measured turn | 1861 | fp8 | 25.4 | PCIe 5.0 x16 | 63.02 | 0.4 | 0.1% |
| last measured turn | 1861 | fp8 | 25.4 | 10 GbE | 1.25 | 21.3 | 3.0% |
| last measured turn | 1861 | fp8 | 25.4 | 25 GbE | 3.12 | 8.5 | 1.2% |
| last measured turn | 1861 | fp8 | 25.4 | 100 GbE | 12.50 | 2.1 | 0.3% |
