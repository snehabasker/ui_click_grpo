"""Make a directory TensorRT-LLM 1.2.1 can load from the merged checkpoint, WITHOUT touching or copying the weights.

    python serving/scripts/make_trtllm_view.py --src serving/models/merged-clean --dst ~/work/merged-clean-trt

Why: the merged checkpoint was written by transformers 5.x, whose Qwen2-VL config nests the language-model fields under ``text_config``
and uses ``dtype``. TensorRT-LLM's environment pins transformers 4.57.3, which parses that file but returns ``architectures = None``, and
TensorRT-LLM then fails with "'NoneType' object is not subscriptable". The weight names already use the original Qwen2-VL layout
(``model.layers.*``, ``visual.blocks.*``), so only small text files need rewriting into the 4.x form: ``config.json`` (flat), ``tokenizer_config.json``
(``extra_special_tokens`` was written as a list, 4.57 wants a mapping; the special tokens themselves live in the symlinked ``tokenizer.json``),
and the image-processor settings (5.x nests them in ``processor_config.json``, 4.x reads ``preprocessor_config.json``). The image pixel budget is
set to the same ``max_pixels`` the vLLM runs used (``--mm-processor-kwargs max_pixels=1003520``), otherwise TensorRT-LLM would see different
prompts. Everything else is a symlink, so ``model.safetensors`` is byte-identical (same sha256) to the one every other phase served. The script
verifies that and refuses to write a config whose hidden size, layer count or head counts differ from the source's.
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


def tokenizer_config(src: dict) -> dict:
    out = dict(src)
    if not isinstance(out.get("extra_special_tokens"), dict):
        out["extra_special_tokens"] = {}
    return out


def preprocessor_config(processor_cfg: dict, max_pixels: int) -> dict:
    ip = dict(processor_cfg["image_processor"])
    ip["min_pixels"] = ip.get("size", {}).get("shortest_edge", 3136)
    ip["max_pixels"] = max_pixels
    ip["size"] = {"shortest_edge": ip["min_pixels"], "longest_edge": max_pixels}
    ip["image_processor_type"] = "Qwen2VLImageProcessor"
    ip["processor_class"] = "Qwen2VLProcessor"
    return ip


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, text: str) -> None:
    """Write a file in the view directory. A stale symlink (left by an earlier run) must be removed first: writing through it would
    overwrite the SOURCE checkpoint's file, which is exactly what an earlier version of this script did."""
    if path.is_symlink() or path.exists():
        path.unlink()
    path.write_text(text)


REWRITTEN = ("config.json", "tokenizer_config.json", "processor_config.json", "preprocessor_config.json")


def build_view(src: Path, dst: Path, max_pixels: int) -> bool:
    src = src.resolve()
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((src / "config.json").read_text())
    flat = flatten(cfg)
    check_same_shape(cfg, flat)
    tok = tokenizer_config(json.loads((src / "tokenizer_config.json").read_text()))
    pre = preprocessor_config(json.loads((src / "processor_config.json").read_text()), max_pixels)
    for name in REWRITTEN:                                  # drop any stale link BEFORE anything is written
        q = dst / name
        if q.is_symlink():
            q.unlink()
    for f in src.iterdir():
        if f.name in REWRITTEN or f.is_dir():
            continue
        link = dst / f.name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(f, link)
    _write(dst / "config.json", json.dumps(flat, indent=1))
    _write(dst / "tokenizer_config.json", json.dumps(tok, indent=1))
    _write(dst / "preprocessor_config.json", json.dumps(pre, indent=1))
    _write(dst / "processor_config.json", json.dumps({"processor_class": "Qwen2VLProcessor"}))
    _write(dst / "VIEW_OF", f"{src}" + chr(10) + "config.json, tokenizer_config.json, preprocessor_config.json, processor_config.json rewritten for transformers 4.x; all other files are symlinks" + chr(10))
    return sha256(dst / "model.safetensors") == sha256(src / "model.safetensors")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--max-pixels", type=int, default=1280 * 28 * 28, help="the image pixel budget used by every other engine in this repo (1003520)")
    a = ap.parse_args()
    same = build_view(Path(a.src), Path(a.dst).expanduser(), a.max_pixels)
    print(f"wrote {a.dst}; model.safetensors identical to the source: {same}")
    sys.exit(0 if same else 1)
