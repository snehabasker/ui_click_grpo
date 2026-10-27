"""Rule 3 for the kernel: does quantising every RMSNorm output to per-row INT8 change what the model clicks?

    python serving/scripts/kernel_accuracy.py --model serving/models/merged-clean

The kernel is not wired into any serving engine (that would need INT8 GEMMs behind it, which this repo does not have), so there is no
speed claim here. What this measures is the accuracy cost of the *numerics* the kernel implements: in the merged model's 57 RMSNorms
(28 layers x input/post-attention norm, plus the final norm) the output is replaced by dequantise(quantise(RMSNorm(x))), "fake
quantisation", and the same greedy HF decoding is run on both fixed sets. Three conditions in one process, so only the norm differs:

    baseline   the model's own RMSNorm
    kernel     serving::rmsnorm_int8 (the Triton kernel), then q * scale
    reference  the fp32 PyTorch reference of the same maths, then q * scale   (the kernel's output should match this one)

Pass criteria, fixed BEFORE looking at results (the same as the merge check): |acc_kernel - acc_baseline| <= 2.0 points AND an
exact two-sided sign test on the paired discordant items has p > 0.05. A failure is reported as a failure.
"""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

import torch
from transformers import Qwen2VLForConditionalGeneration

from servebench import accuracy, envinfo
from servebench.data import load_manifest
from servebench.stats import sign_test_p
from servekernels import rmsnorm_int8 as K
from uiclick.model import load_processor

MAX_DELTA, ALPHA = 0.02, 0.05
SETS = {"synth300": 8, "screenspot400": 2}   # name -> batch size (same as the merge check)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=str(ROOT / "serving" / "models" / "merged-clean"))
ap.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
ap.add_argument("--conditions", nargs="+", default=["baseline", "kernel", "reference"], choices=["baseline", "kernel", "reference"])
ap.add_argument("--error-items", type=int, default=16, help="items of synth300 used to measure the activation error per norm")
a = ap.parse_args()


def load(path):
    # the loader in uiclick.model imports peft even without an adapter; this environment has none, so load the weights directly
    m = Qwen2VLForConditionalGeneration.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda")
    m.generation_config.pad_token_id = m.generation_config.pad_token_id or 0
    return m


def norms(model):
    return [(n, m) for n, m in model.named_modules() if type(m).__name__.endswith("RMSNorm")]


class Patch:
    """Swap each RMSNorm.forward for a fake-quantising one; restore on exit. ``fn(x, weight, eps) -> (q, scale)``."""

    def __init__(self, model, fn, err=None):
        self.mods, self.fn, self.err, self.orig = norms(model), fn, err, {}

    def __enter__(self):
        for name, m in self.mods:
            self.orig[name] = m.forward
            eps = getattr(m, "variance_epsilon", getattr(m, "eps", 1e-6))

            def fwd(x, _m=m, _name=name, _eps=eps):
                q, s = self.fn(x, _m.weight, _eps)
                deq = (q.float() * s.unsqueeze(-1)).to(x.dtype)
                if self.err is not None:                         # error against the model's own RMSNorm output, same input
                    ref = self.orig[_name](x).float()
                    e = self.err.setdefault(_name, [0.0, 0.0, 0.0])
                    e[0] += (deq.float() - ref).pow(2).sum().item()
                    e[1] += ref.pow(2).sum().item()
                    e[2] = max(e[2], (deq.float() - ref).abs().max().item())
                return deq
            m.forward = fwd
        return self

    def __exit__(self, *exc):
        for name, m in self.mods:
            m.forward = self.orig[name]


FNS = {"kernel": lambda x, w, e: torch.ops.serving.rmsnorm_int8(x, w, e), "reference": K.reference}

model = load(a.model)
proc = load_processor(a.model, max_pixels=1280 * 28 * 28)
n_norms = len(norms(model))
print(f"patching {n_norms} RMSNorm modules", flush=True)
assert n_norms == 57, f"expected 28 x 2 + 1 RMSNorms in the language model, found {n_norms}"

preds, timing = {}, {}
for cond in a.conditions:
    for s in a.sets:
        root = ROOT / "serving" / "data" / s
        _, items = load_manifest(root)
        t0 = time.time()
        if cond == "baseline":
            preds[(cond, s)] = accuracy.predict_hf(model, proc, root, items, SETS[s])
        else:
            with Patch(model, FNS[cond]):
                preds[(cond, s)] = accuracy.predict_hf(model, proc, root, items, SETS[s])
        sc = accuracy.score(items, preds[(cond, s)])
        timing[f"{cond}/{s}"] = time.time() - t0
        print(f"{cond:9s} {s:14s} acc={sc['acc']:.4f} valid={sc['valid_format']:.3f} ({time.time() - t0:.0f}s)", flush=True)

# activation error of the kernel's output against the model's own RMSNorm, on real inputs
err = {}
root = ROOT / "serving" / "data" / "synth300"
_, items = load_manifest(root)
with Patch(model, FNS["kernel"], err):
    accuracy.predict_hf(model, proc, root, items[:a.error_items], 8)
rel = {n: (e[0] / e[1]) ** 0.5 for n, e in err.items()}
act_error = {"items": a.error_items, "n_norms": len(rel), "relative_l2_error_mean": sum(rel.values()) / len(rel),
             "relative_l2_error_max": max(rel.values()), "worst_norm": max(rel, key=rel.get),
             "max_abs_error_any_norm": max(e[2] for e in err.values()), "per_norm_relative_l2": rel}
print(f"activation error vs the model's own RMSNorm: mean rel-L2 {act_error['relative_l2_error_mean']:.4f}, "
      f"max {act_error['relative_l2_error_max']:.4f} ({act_error['worst_norm']})", flush=True)

report = {}
for s in a.sets:
    _, items = load_manifest(ROOT / "serving" / "data" / s)
    sc = {c: accuracy.score(items, preds[(c, s)]) for c in a.conditions}
    rep = {"n": len(items), **{c: {k: sc[c][k] for k in ("acc", "ci", "valid_format")} for c in a.conditions}}
    if {"baseline", "kernel"} <= set(a.conditions):
        hb, hk = sc["baseline"]["per_item_hit"], sc["kernel"]["per_item_hit"]
        b, c = sum(x and not y for x, y in zip(hb, hk)), sum(y and not x for x, y in zip(hb, hk))
        p, delta = sign_test_p(b, c), sc["kernel"]["acc"] - sc["baseline"]["acc"]
        rep["kernel_vs_baseline"] = {"delta_acc": delta, "only_baseline_correct": b, "only_kernel_correct": c, "sign_test_p": p,
                                     "identical_output_text": sum(x.strip() == y.strip() for x, y in zip(preds[("baseline", s)], preds[("kernel", s)])) / len(items),
                                     "pass": abs(delta) <= MAX_DELTA and p > ALPHA}
        print(f"{s}: baseline {sc['baseline']['acc']:.4f} vs kernel {sc['kernel']['acc']:.4f} (delta {100 * delta:+.1f} pts), "
              f"discordant {b}/{c}, sign-test p={p:.3f}, identical text {rep['kernel_vs_baseline']['identical_output_text']:.1%} "
              f"-> {'PASS' if rep['kernel_vs_baseline']['pass'] else 'FAIL'}")
    if {"kernel", "reference"} <= set(a.conditions):
        rep["kernel_vs_reference_identical_output_text"] = sum(x.strip() == y.strip() for x, y in zip(preds[("kernel", s)], preds[("reference", s)])) / len(items)
        print(f"{s}: kernel vs fp32-reference quantiser, identical output text {rep['kernel_vs_reference_identical_output_text']:.1%}")
    report[s] = rep

out = ROOT / "serving" / "results" / "raw" / f"kernel_accuracy_{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(envinfo.dumps_scrubbed({"kind": "kernel_accuracy", "criteria": {"max_abs_delta_acc": MAX_DELTA, "sign_test_alpha": ALPHA},
                                       "what": "fake quantisation of every RMSNorm output to per-row INT8 (activations only; weights stay bf16; no INT8 GEMM)",
                                       "env": envinfo.capture(a.model, {"args": vars(a), "seconds": timing}), "results": report,
                                       "activation_error": act_error,
                                       "predictions": {f"{c}/{s}": t for (c, s), t in preds.items()}}, indent=1))
print("wrote", out)
