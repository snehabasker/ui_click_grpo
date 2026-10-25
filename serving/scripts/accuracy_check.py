"""Click accuracy of a model on a fixed set (rule 3).  Writes raw JSON to serving/results/raw/.

    # HF reference (local weights; add --adapter to test an un-merged LoRA)
    python serving/scripts/accuracy_check.py --backend hf --model serving/models/merged-clean --set synth300 --tag merged-hf
    # any OpenAI-compatible server (vLLM / SGLang / router)
    python serving/scripts/accuracy_check.py --backend openai --base-url http://localhost:8000 --model <name> --set screenspot400 --tag vllm
"""
import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import accuracy, envinfo
from servebench.data import load_manifest

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["hf", "openai"], required=True)
ap.add_argument("--set", dest="set_name", choices=["synth300", "screenspot400"], required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--model", required=True, help="hf: weights dir or hub id; openai: served model name")
ap.add_argument("--adapter", default=None)
ap.add_argument("--base-url", default="http://localhost:8000")
ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28)
ap.add_argument("--bs", type=int, default=8)
ap.add_argument("--concurrency", type=int, default=4)
ap.add_argument("--model-dir", default=None, help="path whose hash identifies the weights (openai backend)")
a = ap.parse_args()

root = ROOT / "serving" / "data" / a.set_name
meta, items = load_manifest(root)
t0 = time.time()
if a.backend == "hf":
    from uiclick.model import load_model, load_processor
    model = load_model(model_id=a.model, adapter=a.adapter)
    texts = accuracy.predict_hf(model, load_processor(a.model, max_pixels=a.max_pixels), root, items, a.bs)
    lat = None
    model_dir = a.model if Path(a.model).exists() else None
else:
    texts, lat = asyncio.run(accuracy.predict_openai(root, items, a.base_url.rstrip("/"), a.model, a.concurrency))
    model_dir = a.model_dir
sc = accuracy.score(items, texts)
rec = {"kind": "accuracy", "tag": a.tag, "set": a.set_name, "backend": a.backend, "model": a.model, "adapter": a.adapter,
       "env": envinfo.capture(model_dir, {"args": vars(a), "duration_s": time.time() - t0}), "score": sc,
       "predictions": [{"id": it.id, "text": t} for it, t in zip(items, texts)]}
out = ROOT / "serving" / "results" / "raw" / f"accuracy_{a.tag}_{a.set_name}_{time.strftime('%Y%m%d-%H%M%S')}.json"
out.write_text(json.dumps(rec, indent=1))
print(f"{a.tag} {a.set_name}: acc={sc['acc']:.4f} ci=({sc['ci'][0]:.3f},{sc['ci'][1]:.3f}) valid={sc['valid_format']:.3f} -> {out.name}")
