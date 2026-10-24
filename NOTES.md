# Lab notes: what I tried, including what didn't work

Chronological, honest. Feeds the "what didn't work" section of the README.

## Setup problems (Windows + RTX 5050 Laptop, 8 GB)
- System torch was CPU-only and system NumPy 1.x conflicts with pyarrow, so everything runs in a project venv with
  torch 2.11+cu128 (Blackwell needs cu128+). First model download took ~9 min (4.4 GB).
- transformers is 5.x here: Qwen2-VL needs `mm_token_type_ids` in every forward that has an image, or M-RoPE raises.
- A scratch script named `profile.py` shadowed the stdlib `profile` module and broke `import torch`. Renamed.

## Dead end 1: HF `generate` + full-sequence forward for GRPO
Naive loop = `generate(num_return_sequences=8)` then re-forward 8 x (449 prompt + 12 completion) tokens with grad.
Profile: generate 4.2 s, forward+backward 12 s per prompt-group, i.e. 20-60 s per optimiser step (3 hours per run).
Almost all of it re-processes the *same* 449-token prompt (384 image tokens) 8 times to score ~10-token answers.
**Fix:** run the prompt once, share its KV cache across the 8 completions (`grpo.py: sample_group, completion_logps`).
Gradients still flow through the shared prompt, so this is the exact policy gradient, not an approximation.
Verified against a plain full forward: mean |dlogp| = 0.02 (bf16 rounding; logits ~20 have a bf16 step of ~0.1).
Result: ~7-10 s per step.

## Dead end 2: fp32 LoRA weights blew VRAM
LoRA params are fp32 (stable updates) while the base is bf16, so every LoRA matmul saved fp32 activations:
peak 8.7 GB on an 8.15 GB card, Windows silently spilled to shared memory and everything slowed down.
**Fix:** run forward under `torch.autocast(bf16)`. Peak is still 8.2 GB (1e9 units), right at the ceiling.
Gradient checkpointing is not an option here: HF disables the KV cache under checkpointing, and the cache sharing above needs it.

## Dead end 3: in-training eval was ~100x slower than standalone eval
Same ceiling problem: 96 greedy generations took ~12 min *inside* the training process vs. ~12 min for 1,800 standalone.
The first run (clean) lost ~70 of its 109 minutes to this. **Fix (applied to later runs):** `gc.collect(); empty_cache()`
before each periodic eval. The clean run's numbers are unaffected, only its wall-clock (reported honestly in results).

## Reward-hacking hole found *by reading model samples*, not by theory
Base Qwen2-VL often answers with a *box* `(x0, y0, x1, y1)`. With a 24-token cap a truncated box like
`(733, 414, 886,` contained exactly one valid pair and my first parser accepted it as the click (733, 414).
**Fix:** exactly two numbers may appear in the whole answer (`reward.py`, covered by tests). Boxes, truncated boxes and
multi-point spam all score 0.

## Data-generator iterations
- First layout packed all rows at the top of the window, which makes "click near the top" a free positional shortcut
  I did not plant. Re-did the layout to spread rows over the window. Natural skew remains (targets are 55% left-half
  in the unbiased split, mean y = 0.40) and is why the baselines include "click mean train target".
- Held-out *style* split (one colour theme + one font never seen in training) added later, as the synthetic stand-in
  for "evaluate on unseen apps". Existing seeds produce byte-identical images (hash-checked), so earlier runs stay valid.

## Sanity checks of the coordinate convention
- ScreenSpot bbox is `[x0,y0,x1,y1]` in 0-1. Clicking the exact centre of the converted box scores 100.0% on all 1,272
  samples, so the conversion is right. The "always click centre" baseline scores 2.7% on real UIs.
- Predicted points drawn on screenshots: `scripts/check_convention.py` (synthetic) and `eval_screenspot.py --draw` (real).

## Checklist items and where they landed
| item | status |
|---|---|
| draw predictions to verify convention | done, see results/convention_*.png |
| clean split by app/site, no ScreenSpot training | ScreenSpot is eval-only. Synthetic held-out style split (run `holdout`). |
| log seed/model/data/lr | `run_config.json` per run, `meta` in every eval json |
| reward = hit + small format reward; unparseable = 0 | implemented (`--format_bonus`); main runs used 0 because valid-format hit 100% by step ~25 without it; `holdout` run uses 0.1 |
| centre baseline + large-element check | in every results table (by_size, baseline_center) |
| short test before long run | 3- and 5-step dry runs caught 3 bugs (mm_token_type_ids, memory spill, parser hole) |
| report failures | below / README |

## Late additions
- Control for "is the ScreenSpot gain just prompting?": base model asked for a native bounding box, centre clicked: 9.8% overall,
  60.8% parseable (24-token cap truncated verbose answers), 16% accuracy among parseable vs 54% for the RL model. Not exhaustive:
  no prompt tuning, no longer token cap.
- The laptop went into Modern Standby mid-run (Kernel-Power 506/507) and froze one ScreenSpot eval for an hour; it had to be killed and
  rerun. A temporary SetThreadExecutionState keep-awake process fixed it.
- Two patches of mine silently turned `
` inside string literals into real newlines (report.py, eval_screenspot.py) and
  broke the scripts. Lesson: run the script, don't only `ast.parse` it.
