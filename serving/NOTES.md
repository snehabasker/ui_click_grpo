# Serving notes: dead ends, fixes, and things to remember

Chronological. Numbers here are only ones that come from a committed raw file under `results/raw/`, or are flagged as a
design fact rather than a measurement.

## Phase 0

### Images must be rendered on the machine that has the training fonts
`uiclick/synth.py` draws text with whatever fonts are installed (Arial, Segoe UI, Calibri, Verdana, Tahoma on this Windows
machine; DejaVu on a stock Ubuntu). Rendering inside WSL2 would give *different pixels*, so the policy would see a different
distribution and accuracy would move for reasons unrelated to the serving engine. So every image used by the benchmarks
(trace screenshots, few-shot examples, the two accuracy sets) is rendered once on Windows and shipped as PNG with a sha256 in
a manifest. Engines on any OS then receive byte-identical inputs. The PNGs are git-ignored (traces ~10 MB, ScreenSpot subset
~260 MB of third-party screenshots); manifests and `trace.jsonl` are committed and carry the checksums.

### The re-rendered synthetic set reproduces the published number
`accuracy_check`/`verify_merge` on `serving/data/synth300` with the LoRA adapter gives 63.33% (190/300), identical to
`results/eval_clean.json`. That is the check that the export step did not change the inputs.
(raw: `results/raw/merge_verify_*.json`)

### `base_revision` is null in `merge_info.json`
`huggingface_hub.snapshot_download(local_files_only=True)` did not resolve the cached snapshot offline, so the base model's
commit hash was not recorded. The merged checkpoint is identified by its own sha256 (`model_hash`), which is what every later
run records, so results are still traceable; only the upstream revision is unknown.

### Inter-token latency: validated the client; the mock server cannot be used for small intervals
A unit test against the mock server reported a 0.27 ms inter-token gap when the mock slept 3 ms between tokens (seen in the
failing test output; not a logged result). Neither cause was in the client:
1. On Windows `asyncio.sleep` has coarse granularity: in the logged run `sleep(3 ms)` took 15.95 ms, `sleep(10 ms)` 16.05 ms,
   `sleep(30 ms)` 34.63 ms (`results/raw/client_timing_20261026-004655.json`; the figure moves between runs, e.g. ~7 ms for the
   3 ms sleep in an earlier scratch run).
2. At 10-20 ms the aiohttp *server* inside the mock bunched its writes, so tokens arrived back to back.
So the client was checked against a raw-socket server with exact busy-wait spacing, read both by a raw-socket reader and by
`servebench/client.py`: they agree to within 0.04 ms at 10, 20 and 40 ms (table in README). `scripts/validate_client_timing.py`
reproduces it. The mock test now uses a 30 ms interval. All engine ITL numbers will come from real servers, never the mock.

How the validation itself went wrong first (kept because it is the kind of thing that silently corrupts a benchmark):
* In the first logged runs only half of the client's requests succeeded (4 of 8, every other one, `ServerDisconnectedError`
  after 0.000 s), so only 36 of 72 gaps were collected. The timings that *were* collected agreed with the reference, but I did
  not want to report a validation that dropped half its samples.
* The cause was my crude raw test server, not the client: it did not read the request body before closing the socket and did not
  send `Connection: close`, so aiohttp reused a closed pooled connection on every second request.
* Two of my first fix attempts did not take effect at all (an escaping mistake in a scripted edit that did not assert the
  replacement matched). I found this by grepping the file instead of trusting the "fixed" run. After applying them properly:
  0 errors, 72 of 72 gaps.
* Separately, `servebench/client.py` now reads each stream to EOF instead of stopping at `[DONE]`. This is the correct way to
  leave a pooled connection in a known state; it was not the cause of the errors above.

### The mock server is for tests only
`servebench/mock_server.py` answers `(500, 500)` with configured sleeps and a crude prefix-cache simulation (it only calls a
hit when the *entire* prefix including growing session history repeats, unlike a real longest-common-prefix cache).
Output files from it are prefixed `MOCK_` and are never reported. A smoke run of the full `loadtest.py` path against it
succeeded (48/48 requests, environment record complete) and the file was deleted.

### Things I must not forget in Phase 1
* ScreenSpot images are up to 1920x1080. The HF evaluation used a processor with `max_pixels = 1280*28*28`; vLLM / SGLang
  must be launched with the equivalent multimodal-processor setting, otherwise they silently resize differently and accuracy
  changes for a reason that is not the engine.
* The policy was trained zero-shot with one screenshot. The agent trace adds a system prompt, two few-shot screenshots and
  text history. Click accuracy *on the trace* is reported separately from the zero-shot accuracy checks.
* `usage.prompt_tokens_details.cached_tokens` and engine `/metrics` need engine-specific flags; the client records whichever
  is available and says which.
* Prefix cache must be flushed (or the server restarted) between repeats, otherwise repeat 2 and 3 start warm.

### Environment status
* WSL2 is **not installed** on this machine (checked: `wsl --status`). Linux phases wait for the user to install it.
* Rust and Docker are not installed. Rust will be installed inside WSL2.
* GPU: RTX 5050 Laptop, 8 GB, compute capability 12.0, driver 592.82, CUDA 13.1 (driver-reported).

## Phase 1: getting two engines to run on a Blackwell laptop GPU

Neither engine started out of the box. Each failure had one specific cause, found by reading the server log (not guessing),
and each fix was applied once and verified by the next launch. Order matters: the bottleneck moved down the stack.

**Environment (WSL2).** Ubuntu came with Python 3.14 only, no pip/venv, no compiler, and `sudo` needs a password I cannot type.
Everything below was done without `sudo`: `uv` (user-level) for Python 3.12 venvs, `micromamba` for compilers. The minimal image
has no `bzip2` binary, so micromamba's archive was unpacked with Python's `tarfile`. WSL gets half the host RAM (11.7 GB), which
was enough for one engine. The Ubuntu disk was moved to another partition of the same SSD with `wsl --manage Ubuntu --move`
(a manual cut-and-paste of `ext4.vhdx` would break WSL's registry record of where it lives).

**vLLM 0.32.0 (needed three launches).**
1. `RuntimeError: Failed to find C compiler` inside Triton's driver setup (it compiles a small C helper on first use). Fix: a
   conda-forge GCC in `~/toolchain`, selected via `CC`. The weights had already loaded (4.26 GiB), so GPU support was not the issue.
2. `Could not find nvcc` from FlashInfer's top-k/top-p sampler JIT. The attention backend (FlashAttention 2) was fine. Fix:
   `VLLM_USE_FLASHINFER_SAMPLER=0`. All benchmark requests are greedy, so this cannot change any output.
3. Ready after 210 s (compile + 13 CUDA graphs); later starts reuse the compile cache (60-90 s).

**SGLang 0.5.22 (needed three launches).**
1. Failed in 21 s, before loading anything: on sm_120 it imports `deep_gemm`, whose `find_cuda_home()` asserts a CUDA toolkit
   exists. Fix: CUDA 13.2 (nvcc, headers) from conda-forge in `~/cuda132`, matching the CUDA of its torch wheel. A test kernel
   compiled for sm_120 and ran on the GPU. nvcc needed its bundled host compiler (GCC 15.3) via `NVCC_PREPEND_FLAGS=-ccbin`;
   the GCC 16 in the other prefix was not used for nvcc.
2. A JIT kernel compiled but failed to link: `cannot find -lcudart`. The conda layout keeps libraries in `lib/`, SGLang links
   against `lib64/`. Fix: a `lib64 -> lib` symlink inside that private prefix. The failed build killed the server during CUDA-graph
   capture, which looked like a crash but was this one link error.
3. Ready after 80 s.
The CUDA toolkit is only put on PATH by `serve_sglang.sh`, so the already-validated vLLM launch is untouched.

**Settings that are NOT identical between the engines (read before comparing them).**
* Attention kernels: vLLM uses FlashAttention 2 (its own choice); SGLang uses Triton attention, which I set explicitly as the
  nvcc-free option *before* I knew nvcc would work. I never tested SGLang's FlashInfer or FA3/FA4 backends, so SGLang's numbers
  are for the Triton backend, not necessarily its best configuration on this GPU.
* Sampler: PyTorch's on both (vLLM's is a deviation from its default, recorded in the server command).
* KV-cache capacity differs because the memory-fraction semantics differ: 57,168 tokens (vLLM) vs 50,728 (SGLang).
* Different PyTorch versions (2.13.0 vs 2.14.1), as each engine pins its own.
* Scheduler token budget: this one *matches* by default. I first suspected vLLM's per-step budget was larger than SGLang's
  2,048-token prefill chunk, and checked: in this vLLM version's source, a GPU under 70 GB gets `max_num_batched_tokens = 2048`
  for the OpenAI server. (Derived from the source, not logged at runtime.) So that hypothesis was wrong.
* SGLang captured decode CUDA graphs only up to batch size 8 (`bs=[1, 2, 4, 8]` in its startup log); decode batches of 16
  requests report `cuda graph: False`. I did not try a larger `--cuda-graph-max-bs`.

**Exploratory runs (not reported as results).** Before the formal runs I launched each engine once and ran the accuracy checks and
a short load test to find problems. Those files are tagged `explore` and are excluded from every table by `summarize.py`.
One useful cross-check came out of them: the cached-token fraction from the per-request `usage` field and the one computed from
the engine's own `/metrics` counters were identical (0.6322 vs 55,344 hits / 87,545 queries) on the first vLLM run.

**Run-to-run variation I could not explain.** SGLang with the prefix cache on at concurrency 16 has a visibly wider spread than
the other cells (throughput 1.40-1.58 req/s, ITL p50 30.9-52.8 ms). A laptop GPU under a 35 W cap may be thermally limited, but
Phase 1 did not record GPU temperature or clock speed, so this is a hypothesis, not a finding. Later phases record both.

**Things I did not do.** No test of whether the two engines' *outputs* differ token by token (only click accuracy); no
profile of where the time goes (vision encoder vs prefill vs decode); no repeat of the whole experiment on a second day.

## Phase 1 follow-up: what went wrong, in order

Goal: test whether SGLang's low throughput in Phase 1 was my configuration (Triton attention, CUDA graphs only up to batch 8).
It took several attempts and four distinct problems; each cause came from a log, not a guess. The rows below are grouped by problem,
not strictly in time order (step A actually ran before the last two FlashInfer retries).

| # | what I ran | what happened | cause | fix |
|---|---|---|---|---|
| 1 | first follow-up script | SGLang steps failed instantly; vLLM rerun ran | **The Linux clone had not updated.** `git pull` aborted because the clone held untracked copies of result files the new commits also contain, my sync script piped it through `tail` which hid the failure, and I read a truncated "Updating a..b" line as success. The runner was the old one and rejected the new arguments. | Sync script now fails loudly unless the clone ends at exactly the Windows HEAD, prints the hash, removes only byte-identical duplicates and generated tables. I also check the *content* (arguments exist), not only the hash. The vLLM rerun from that attempt used old code without the new monitor and was discarded. |
| 2 | B and A (SGLang) | refused to start in 9-18 s | SGLang 0.5.22 renamed `--cuda-graph-max-bs`: plain `--cuda-graph-max-bs` is now ambiguous between `-decode` and `-prefill`. My earlier flag check was a substring `grep` count, which wrongly looked like the flag existed; SGLang's own earlier error message even named the new flag. | `--cuda-graph-max-bs-decode`, with a test that guards against the old spelling. |
| 3 | B | FlashInfer JIT failed | FlashInfer calls nvcc with its own `-ccbin` (plain `c++`, my GCC 16), overriding my `NVCC_PREPEND_FLAGS`; CUDA 13.2 rejects GCC > 15 ("unsupported GNU version"). | `CC`, `CXX`, `CUDAHOSTCXX` point at the GCC 15.3 that ships with the toolkit. |
| 4 | B | kernels compiled, then link failed | `cannot find -lcuda`: under WSL2 the driver library is in `/usr/lib/wsl/lib`, which the conda linker does not search. | `LIBRARY_PATH` includes it. |
| 5 | B | ran | | |
| 6 | A | ran | | |
| 7 | C (vLLM rerun) | ran | | |

Between attempts I once followed the wrong log: my "wait for ready" helper picked the newest log by name and read the previous
attempt's failure before the new run had created its own file. `wait_new_log.sh` now waits for a log newer than itself.

**The SM clock is not reported.** Under WSL2 `nvidia-smi` shows 180 MHz for the SM clock at 100% utilisation and ~40 W, i.e. the
idle value, so the field is useless there. The monitor still records it in the raw files but no table uses it. Temperature and
power are real readings.

**The GPU's power state is a confound I did not control.** Median power per run ranged from 28 W to 55 W and peak temperature from
69 to 92 C. In SGLang's Triton c=32 cell throughput went 1.18, 1.57, 1.64 req/s across repeats while median power went 28.4, 55.0,
51.6 W (`scripts/gpu_state_by_run.py` prints every run in time order). I did not compute a correlation: three repeats per group
are too few to mean anything. The FlashInfer runs sat at 50-55 W and 87-92 C throughout, so a low-power state cannot explain why they
did not beat Phase 1.

**Phase 1 did not reproduce exactly.** vLLM prefix-on at c=16: 3.07 req/s (3.06-3.09) in the rerun vs 3.52 (3.48-3.55) in Phase 1,
and p50 time to first token 3.03 s vs 2.21 s. Phase 1 recorded no temperature or power, so I cannot say what differed (the rerun ran
at 39.8 W and 76-79 C). Within-session repeats were within about 2%; across sessions the difference was 13%. The README's
min-max ranges therefore understate real uncertainty.

**Hypotheses I rejected or could not test.**
* Rejected: that SGLang's low throughput came from small decode CUDA graphs (graphs up to 32 fixed inter-token latency, 51 to 14-18 ms
  at c=16, but not throughput or time to first token) or from Triton attention (FlashInfer was no better).
* Not tested: SGLang's vision-encoder attention backend (`sdpa` here; vLLM's log says FlashAttention for its vision encoder), its
  scheduling policy and multimodal preprocessing. These are the next things I would try.

## Phase 2: the Rust router

### Two replicas on one 8 GB GPU: the second one computed a negative KV budget
The first attempt launched two FP8 replicas one after the other, each with `--gpu-memory-utilization 0.46`. Replica 0 logged
`Available KV cache memory: 0.68 GiB`; replica 1 logged `-2.24 GiB` and exited with `No available memory for the cache blocks`.
In vLLM 0.32.0 (`v1/worker/gpu_worker.py`, `utils/mem_utils.py`) the budget is `util x total memory` minus a "non-KV" figure, and that
figure is the drop in *device-wide* free memory while the replica starts up. For replica 1 the figure came out near 5.9 GiB
(3.66 requested + 2.24), more than the 2.38 GiB of weights plus activations the replica can account for. I did not find out what else
the measurement counted; I suspect the first replica's memory, but I did not verify that, so it is only a guess.
Fix: `--kv-cache-memory` (bytes), which makes vLLM skip that profiling step. Both replicas now get an explicit 0.40 GiB, which is
14,976 tokens each. Every router number in the README is for that configuration; the cache is small, and 8 concurrent sequences of
the trace's mean prompt length (about 1.8k tokens) come to roughly the whole cache. That bears on how much prefix caching can help
here and is untested.

### The first router comparison measured the run order, not the policy
Prefix-aware ran first and round-robin second; round-robin looked faster (c=16 time to first token 1.59 s vs 2.29 s). Median GPU
power was ~30 W for the prefix half and ~65 W for the round-robin half. I reran with the order reversed, and the prefix policy
(now second) had the higher throughput at every concurrency and the lower time to first token at c=4 (0.27 s vs 0.34 s), though not at
c=16 or c=32. Round-robin's own time to first token at c=4
went 0.64 s, 0.34 s, 0.42 s across the three sessions with identical settings. So the GPU's state moved latency by up to 2x, more
than any policy effect I could see. Why the first half ran at ~30 W I do not know. The final comparison alternates which policy
goes first on every repeat (`--interleave`); the first two runs stay in `results/raw/` and in `results/tables/phase2.md`, labelled
by run order, because deleting the inconvenient one would hide the confound.

### I first wrote "load stayed balanced" from the wrong counter
I read per-replica totals (280 / 303) from the router's cumulative counter after the first run, which included the 300 accuracy
requests and one load run. Recomputed per cell from the interleaved record, prefix-aware routing is uneven (replica 0 got 40% of
requests at c=32: 699 vs 1041), while round-robin is 50/50. The README states the recomputed figures.

### Four of my test expectations were wrong, not the router
* the lower bound on how many requests a hot prefix spills, and the breaker's failure bound: an open breaker admits one probe per cooldown, so the number of failed
  attempts grows with elapsed time. I had bounded it by a constant. This happened twice (a second time in
  `server_errors_are_retried_and_the_breaker_opens_then_recovers`, found when the first-byte change made the loop slower than the
  250 ms cooldown).
* the retry test raced the health checker, which marked the dead replica down before the request reached it; split into a test with
  health checks effectively off.
In each case the router behaved as designed and the assertion encoded a timing assumption.

### The first kill-a-replica test hid two problems
Run 1 (earlier build): 557/560 requests succeeded; the 3 failures were in flight on the killed replica. Two had received no token yet
but the response headers had gone out, so the router had already committed to that replica and could not retry. Also the load
finished (118 s) before the restarted replica was ready (146 s), so rejoining under traffic was never exercised
(`first_request_served_by_restarted_replica_at: null`). Fixes: the router now waits for the first body chunk before committing
(tests for both sides of that line), and the final run kills earlier (25 s) and restarts after 2 s so traffic outlasts the restart.
A replica that dies after the first chunk is still not retried, on purpose: the client already holds part of the answer.
Also noted: an open breaker closes only when a real request probes the replica; the health check recovering does not close it.
With no traffic it stays open.

### FP8 replicas cost accuracy
Two bf16 copies do not fit in 8 GB, so the replicas are FP8 (`--quantization fp8`, 2.38 GiB of weights each). Paired against the bf16
vLLM run: ScreenSpot 53.00% -> 49.75% (24 clicks lost, 11 gained, sign test p=0.041); synthetic 64.33% -> 64.33% (10 and 10). The
router results therefore describe the FP8 policy, which is a slightly worse clicker on ScreenSpot than the one in Phase 1.

### Smaller things
* The `.sh` files edited on Windows had CRLF endings, which broke `set -x` in a replica launch (`set -x\r`). `.gitattributes` pins
  `*.sh` to LF and the runs use the Linux clone's copy.
* `collect_results.sh` did not copy `router_bench_*.json` / `router_failure_*.json`, so the first summary came out empty; fixed.
* Two background "wait for the bench to finish" loops matched their own command line with `pgrep -f` and never exited; killed by PID.

## Phase 3: the Triton kernel

### What went wrong, in order

* **Windows Python has no Triton**, so the kernel, its tests and the benchmark run in WSL, in the vLLM venv (torch 2.13 + Triton 3.7.1,
  both work on the 5050, compute capability 12.0). Other venvs also have Triton (`flowgen`, `sgym`: 3.6.0; `sglang`: 3.8.0); only 3.7.1 was used.
* **That venv has no pip** (it was made by `uv`), so `pytest` went in with `uv pip install --python ... pytest`. Nothing else was changed in it.
* **First benchmark attempt died with "Failed to find C compiler"** (Triton and inductor build small C launchers). The kernel tests had passed
  only because the launcher was already cached. The serve scripts already put `~/toolchain/bin` on PATH when there is no system gcc; the kernel
  scripts did not. Fixed, committed, then the benchmark was re-run from the committed code. The log directory I used (`~/logs`) did not exist
  either (`~/work/logs` is the real one), so the first attempt's log was lost; nothing from it is in the results.
* **The first smoke run showed cache effects.** `do_bench_cudagraph` reached 490 GB/s on a 5 MB operation, above the 384 GB/s peak, because
  it does not flush L2 and the tensors fit in the 32 MiB cache. Kept as a labelled column, but the roofline uses only cells of at least 64 MiB.
* **My first GPU-state sampler was useless**: it read `nvidia-smi` between repeats, when the GPU was idle (180 MHz). Replaced by a 1 Hz sampler
  during each repeat. Under WSL the SM clock still reads 180 MHz in every sample at 67 W, so the clock column is reported as unusable and not cited.
* **The GPU throttled during the benchmark** (power cap, then thermal slowdown at 83-89 C), so repeats are bimodal and the measured copy ceiling moved
  from 262 to 321 GB/s. I did not re-run with a per-cell ceiling measurement; instead the roofline table compares each cell's best repeat with the
  best measured ceiling, and the README says why. The ratios between implementations are measured back to back within a cell and are not affected.
* **The tests passed on the first run**, which made me want proof they could fail. I broke the kernel on purpose (126 instead of 127 in the
  scale) and confirmed a test failed, then restored the file.
* **README prose I had to correct before publishing**: I first wrote that the kernel and `torch.compile` are about equal "at 1 or 16 rows"; at
  hidden size 8192 with 1 row the kernel is 1.36-1.46x faster, so only 5 cells are tied. I also wrote that eager launches "about ten kernels"
  without having counted them; removed. The checker now guards the ratio ranges.
* **The accuracy check needed its own loader**: `uiclick.model.load_model` imports `peft` even without an adapter, and the vLLM venv has none, so
  `kernel_accuracy.py` loads the merged checkpoint with `Qwen2VLForConditionalGeneration` directly (same dtype, attention and device).
* **Text identity is a poor metric here.** Only 40-43% of outputs were identical between the baseline and the kernel, which looked alarming until the
  adapter-vs-merged check from Phase 0 (same function up to rounding) showed only 54.5-56.3%, and the kernel vs the fp32 reference quantiser (nearly
  the same function) showed 38.5-43.3%. Differences of a point or two in accuracy are within what bf16 noise alone produces on these sets.

### Not established

* Why `do_bench` and `do_bench_cudagraph` disagree by about 20% at the largest cells.
* Why the fused kernel beats `torch.compile` by 1.4-2.6x at 1024+ rows (inductor's kernels were not inspected).
* Whether a tuned launch configuration (`num_warps`, several rows per program) would change the small-cell results.

## Phase 4: disaggregation (one GPU)

### What went wrong, in order

* **The first split run died under concurrency.** The decode instance's engine crashed with `shape mismatch: value tensor of shape [1248, 2, 256] cannot
  be broadcast to indexing result of shape [1664, 2, 256]` in vLLM's `ExampleConnector.inject_kv_into_layer`; every in-flight request then failed. At 1
  client everything worked (91 of 91). I kept that run as a result: it is what the default configuration does.
* **My first diagnosis was incomplete.** Reading `build_connector_meta` showed the connector stores a request's KV at the first step it is scheduled, with the
  blocks allocated then, so a prompt scheduled in pieces is stored truncated. Chunked prefill was the obvious cause, and switching it off on the prefill
  instance fixed 4 clients (263 of 263 in a diagnostic run). I first wrote that up as "chunked prefill", then 16 clients still failed with the *same* error. The gap
  (1,664 - 1,248 = 416 tokens) is about one screenshot, so something else also splits a prompt across steps (I suspect the multimodal encoder's per-step budget;
  not confirmed). A 16,384-token step budget stopped the errors at 16 clients, which is what the README reports, as an inference.
* **The 16,384-token configuration is very slow** (median time to first token 29.6 s, 0.61 requests/s at 16 clients, against about 1.5 s and 6.4 for colocated, in a
  different session) and drifts slower across rounds (21.5, 29.6, 30.3 s). I did not measure where the time goes. 1 to 2 of 560 requests still failed in two
  rounds; not investigated.
* **Log files are overwritten per setup** (`disagg-<role>-<timestamp>.log` reuses the run's timestamp), so after a multi-round run only the last split instance's
  log survives; the diagnosis used the logs of the run that had just finished.
* **A diagnostic run wrote its result under the real name** (`disagg_bench_...`), because the tag prefix option did not exist yet. I renamed it
  `explore_disagg_bench_20261030-181519.json` before committing so the summariser does not treat it as a result.
* **The proxy counts the warm-up requests** in its prefill-step timings (the load client sends its discarded warm-up through it), so the "prefill step" column covers them too.

### Not established

* Why the 16,384-token configuration is so slow, and why 1 to 2 requests per 560 still fail in it.
* Whether the multimodal encoder budget is what splits the prompts at 16 clients.
* Anything about a real two-GPU deployment, or about connectors other than `ExampleConnector`.

## Phase 5: TensorRT-LLM

TensorRT-LLM 1.2.1 (PyTorch backend) runs on this GPU (RTX 5050 Laptop, compute capability 12.0) under WSL2 without sudo, after a series of fixes. Every
one was a missing library, a version skew or a missing header; none was the GPU being unsupported. In order:

1. `import tensorrt_llm` failed with `cannot load MPI library`: no `libmpi` without apt. Fix: OpenMPI from conda-forge into `~/mpi` (`09_trtllm_mpi.sh`).
2. `libcublasLt.so.13` not found: the default PyPI `torch==2.9.1` is built for CUDA 12.8, TensorRT-LLM 1.2.1 links CUDA 13. uv treated `torch==2.9.1` as already
   satisfied by the `+cu128` build, so the first reinstall did nothing; the explicit `torch==2.9.1+cu130` from the PyTorch index was needed (`10_trtllm_torch_cu130.sh`).
3. `libnvrtc.so.13`: the CUDA 13 runtime libraries had to be on `LD_LIBRARY_PATH` (`trtllm_env.sh`). Then `libnuma.so.1` (conda-forge `libnuma` into `~/mpi`).
4. torchvision was still the CUDA 12.8 build: "PyTorch and torchvision were compiled with different CUDA major versions" (`11_trtllm_torchvision.sh`).
5. `pyarrow` 26 needs NumPy 2 while TensorRT-LLM pins NumPy 1.26: older `pyarrow<21` (`12_trtllm_pyarrow.sh`). Then `deep_gemm` asserts `CUDA_HOME` is set (the user-space toolkit in `~/cuda132`).
6. **The merged checkpoint could not be read by TensorRT-LLM's transformers (4.57.3)**: it was written by transformers 5.x. `config.json` is nested (`text_config`), so
   `architectures` came back `None` and the server died with `'NoneType' object is not subscriptable`; then `tokenizer_config.json` had `extra_special_tokens` as a
   list where 4.57 wants a mapping; then the image-processor settings are in a different file. `make_trtllm_view.py` builds a directory whose weights are a
   symlink to the same `model.safetensors` (same sha256) and whose small config files are rewritten for 4.x, with the image pixel budget set to the 1,003,520 used elsewhere.
7. MPI worker spawn failed (`OPAL ERROR ... dpm.c`): OpenMPI 5 needs its `prte` daemon on `PATH`.
8. The Qwen2-VL vision tower hard-codes `attn_implementation='flash_attention_2'` in `modeling_qwen2vl.py` (comment: "to avoid OOM issue") and `flash_attn` is not installable
   here. **A local patch to the installed package** switches that line to `sdpa` (`13_trtllm_sdpa_patch.sh`; original kept as `.orig`; the diff is recorded in the result).
   After that, FlashInfer JIT-compiles a normalisation kernel for `sm_120a`: nvcc 13.2 rejects the user-space gcc 16 (use gcc 14 from conda-forge, `14_trtllm_gcc14.sh`), `cublasLt.h`
   comes from the cu13 pip package via `CPATH`, and the link step needs a `libcuda.so` name (a symlink to WSL's `/usr/lib/wsl/lib/libcuda.so.1`, `15_trtllm_libcuda_link.sh`).

### A mistake of mine that touched the checkpoint

The first version of `make_trtllm_view.py` symlinked `tokenizer_config.json` and `processor_config.json` into the view directory. The second version rewrote those two files
in the view directory, which wrote *through the stale symlinks* and overwrote the real files in the Linux clone's `serving/models/merged-clean/`
(`tokenizer_config.json`: the token list became `{}`; `processor_config.json`: reduced to the class name). The weights, `config.json`, `tokenizer.json` and the chat template
were not changed (checked by hash). I found it because the next run failed with `KeyError: 'image_processor'`, restored both files from the untouched Windows copy,
verified every file of the checkpoint (including the 4.4 GB weights) is byte-identical to the Windows copy and to `MODEL_HASH`, made the script remove any link or file before
writing, and added a regression test that reproduces the situation (it needs symlinks, so it runs in WSL: `wsl/run_view_tests.sh`). The overwrite happened after the Phase 3 and
Phase 4 runs had finished, so no reported number used the altered files.

### Other things that went wrong

* The agent trace (three images per request) fails inside the Hugging Face Qwen2-VL processor under TensorRT-LLM: `IndexError: index 3 is out of bounds for dimension 0 with size 3`
  (4 image placeholders, 3 images). I did not find why. The comparison therefore uses a new one-image-per-request trace (`traces/single`), not the Phase 1 workload.
* The first run of the benchmark driver died in round 2 with `Address already in use`: the TensorRT-LLM server killed at the end of round 1 still held port 8003. The driver now waits for the
  port to be free. The round-1 files of that crashed invocation are not results (no benchmark record lists them); they are kept in the clone as `loadtest_explore-crashed-*`.

### Not established

* Why TensorRT-LLM scores higher on ScreenSpot than vLLM (see the README): preprocessing differences are one candidate.
* Whether TensorRT-LLM would be faster with its default flash-attention vision tower, with CUDA graphs tuned, or with its TensorRT (engine-building) backend, which was not tried.
* Why the multi-image prompts fail.
