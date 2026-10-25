"""Merge the trained LoRA into the base weights so every serving engine loads one plain checkpoint.

    python serving/scripts/merge_adapter.py --adapter runs/clean/adapter --out serving/models/merged-clean

Merging happens in bf16 (the base dtype), so the merged weights differ from "base + fp32 LoRA path" by rounding.
That is exactly what scripts/verify_merge.py measures; this script makes no claim about equivalence.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

from servebench import envinfo
from uiclick.model import MODEL_ID

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)

base = Qwen2VLForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)  # CPU: keeps the GPU free
merged = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
merged.save_pretrained(out, safe_serialization=True)
AutoProcessor.from_pretrained(MODEL_ID).save_pretrained(out)

try:
    from huggingface_hub import snapshot_download
    base_revision = Path(snapshot_download(MODEL_ID, local_files_only=True)).name
except Exception:
    base_revision = None
adapter_file = Path(a.adapter) / "adapter_model.safetensors"
info = {"base_model": MODEL_ID, "base_revision": base_revision, "adapter": str(a.adapter),
        "adapter_sha256": envinfo.file_sha256(adapter_file), "merge_dtype": "bfloat16",
        "model_hash": envinfo.model_hash(out), "env": envinfo.capture(out)}
(out / "merge_info.json").write_text(json.dumps(info, indent=2))
print("merged ->", out, "\nmodel_hash", info["model_hash"], "\nbase revision", base_revision)
