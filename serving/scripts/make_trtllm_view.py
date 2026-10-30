"""Make a directory TensorRT-LLM 1.2.1 can load from the merged checkpoint, WITHOUT touching or copying the weights.

    python serving/scripts/make_trtllm_view.py --src serving/models/merged-clean --dst ~/work/merged-clean-trt

Why: the merged checkpoint was written by transformers 5.x, whose Qwen2-VL config nests the language-model fields under ``text_config``
and uses ``dtype``. TensorRT-LLM's environment pins transformers 4.57.3, which parses that file but returns ``architectures = None``, and
TensorRT-LLM then fails with "'NoneType' object is not subscriptable". The weight names already use the original Qwen2-VL layout
(``model.layers.*``, ``visual.blocks.*``), so only ``config.json`` needs rewriting into the flat 4.x form. Every other file is a symlink,
so ``model.safetensors`` is byte-identical (same sha256) to the one every other phase served. The script verifies that and refuses to
write a config whose hidden size, layer count or head counts differ from the source's.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def flatten(c: dict) -> dict:
    """transformers-5 nested Qwen2-VL config -> the flat transformers-4 form."""
    t = c["text_config"]
    flat = {k: v for k, v in c.items() if k not in ("text_config", "dtype")}
    flat.update({k: v for k, v in t.items() if k not in ("model_type", "dtype", "rope_parameters")})
    flat["torch_dtype"] = c.get("dtype") or t.get("dtype") or "bfloat16"
    rp = t.get("rope_parameters")
    if rp:
        flat["rope_scaling"] = {"type": "mrope", "mrope_section": rp["mrope_section"], "rope_type": "default"}
        flat["rope_theta"] = rp["rope_theta"]
    return flat


def check_same_shape(src: dict, flat: dict) -> None:
    t = src["text_config"]
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "vocab_size", "rms_norm_eps"):
        if flat[k] != t[k]:
            raise SystemExit(f"flattened config differs from the source for {k}: {flat[k]} vs {t[k]}")
    if flat["vision_config"] != src["vision_config"]:
        raise SystemExit("vision_config changed")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    a = ap.parse_args()
    src, dst = Path(a.src).resolve(), Path(a.dst).expanduser()
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    flat = flatten(cfg)
    check_same_shape(cfg, flat)
    for f in src.iterdir():
        if f.name == "config.json" or f.is_dir():
            continue
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(f, link)
    (dst / "config.json").write_text(json.dumps(flat, indent=1))
    (dst / "VIEW_OF").write_text(f"{src}\nconfig.json rewritten from the transformers-5 nested form to the flat 4.x form; all other files are symlinks\n")
    same = sha256(dst / "model.safetensors") == sha256(src / "model.safetensors")
    print(f"wrote {dst}; model.safetensors identical to the source: {same}")
    sys.exit(0 if same else 1)
