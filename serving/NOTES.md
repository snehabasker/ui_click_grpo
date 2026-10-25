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
