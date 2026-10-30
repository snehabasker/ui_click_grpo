"""Does a served engine see the same prompt as the Hugging Face reference? Compares the engine's reported prompt_tokens with the number the
Hugging Face processor produces for the same request at the repo's image pixel budget (max_pixels = 1003520).

    python serving/scripts/check_prompt_tokens.py --base-url http://127.0.0.1:8003 --model merged-clean --set screenspot400 --n 20 --tag trtllm

Why: a click-accuracy difference between two engines could come from different image preprocessing (a larger pixel budget shows small UI
elements in more detail) rather than from the engine's numerics. Equal token counts do not prove identical pixels, but a different count proves
a different input. Writes raw JSON to serving/results/raw/ (kind prompt_tokens_check). Needs the Hugging Face processor (the vllm venv has it).
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from PIL import Image

from servebench import accuracy, envinfo
from servebench.data import load_manifest
from uiclick.model import build_inputs, load_processor

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--set", dest="set_name", default="screenspot400", choices=["synth300", "screenspot400"])
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--tag", required=True)
ap.add_argument("--processor-dir", default=str(ROOT / "serving" / "models" / "merged-clean"))
ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
a = ap.parse_args()

root = ROOT / "serving" / "data" / a.set_name
_, items = load_manifest(root)
items = items[: a.n]
proc = load_processor(a.processor_dir, max_pixels=a.max_pixels)

rows = []
for it in items:
    msgs = accuracy.zero_shot_messages(root, it)
    body = {"model": a.model, "messages": msgs, "max_tokens": 1, "temperature": 0}
    req = urllib.request.Request(f"{a.base_url}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        served = json.loads(r.read())["usage"]["prompt_tokens"]
    img = Image.open(root / it.image).convert("RGB")
    ref = build_inputs(proc, [img], [it.instruction], device="cpu")["input_ids"].shape[1]
    rows.append({"id": it.id, "image_size": list(img.size), "served_prompt_tokens": served, "reference_prompt_tokens": int(ref), "diff": served - int(ref)})
    print(f"{it.id}: size {img.size} served {served} reference {ref} diff {served - ref:+d}", flush=True)

diffs = [r["diff"] for r in rows]
summary = {"n": len(rows), "equal": sum(d == 0 for d in diffs), "min_diff": min(diffs), "max_diff": max(diffs)}
print(summary)
out = ROOT / "serving" / "results" / "raw" / f"prompt_tokens_check_{a.tag}_{a.set_name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(envinfo.dumps_scrubbed({"kind": "prompt_tokens_check", "tag": a.tag, "set": a.set_name, "max_pixels": a.max_pixels,
                                       "env": envinfo.capture(a.processor_dir, {"args": vars(a)}), "summary": summary, "rows": rows}, indent=1))
print("wrote", out.name)
