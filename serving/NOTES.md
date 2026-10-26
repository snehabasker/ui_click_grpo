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
