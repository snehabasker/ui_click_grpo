"""Generate an agent-style trace (see servebench/traces.py).

    python serving/scripts/gen_traces.py --name default --n-sessions 96 --shared-fraction 0.75 --stats

--stats tokenises one representative request per kind with the real Qwen2-VL processor, so prefix and per-turn token
counts in the README are measured, not assumed. Run this on the machine whose fonts the policy was trained on (Windows):
the PNGs it writes are what every engine is fed.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "serving"))

from servebench import envinfo
from servebench.traces import TraceConfig, generate, load_trace

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="default")
ap.add_argument("--n-sessions", type=int, default=96)
ap.add_argument("--min-turns", type=int, default=3)
ap.add_argument("--max-turns", type=int, default=8)
ap.add_argument("--shared-fraction", type=float, default=0.75)
ap.add_argument("--n-shared-groups", type=int, default=4)
ap.add_argument("--n-fewshot", type=int, default=2)
ap.add_argument("--history", choices=["none", "text", "full"], default="text")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--stats", action="store_true", help="measure token counts with the real processor")
a = ap.parse_args()

cfg = TraceConfig(name=a.name, n_sessions=a.n_sessions, min_turns=a.min_turns, max_turns=a.max_turns,
                  shared_fraction=a.shared_fraction, n_shared_groups=a.n_shared_groups, n_fewshot=a.n_fewshot,
                  history=a.history, seed=a.seed)
out = ROOT / "serving" / "traces" / a.name
meta = generate(cfg, out)
print(json.dumps({k: v for k, v in meta.items() if k != "fonts"}, indent=1))

if a.stats:
    from PIL import Image

    from servebench.prompts import build_messages
    from uiclick.model import load_processor

    proc = load_processor()
    sessions = load_trace(out)

    def n_tokens(msgs):
        # swap data URIs for the real images, render the chat template, count tokens
        images, conv = [], []
        for m in msgs:
            if isinstance(m["content"], str):
                conv.append(m)
                continue
            parts = []
            for p in m["content"]:
                if p["type"] == "image_url":
                    parts.append({"type": "image"})
                    images.append(p["_path"])
                else:
                    parts.append(p)
            conv.append({"role": m["role"], "content": parts})
        text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        imgs = [Image.open(p).convert("RGB") for p in images]
        return proc(text=[text], images=imgs or None, return_tensors="pt")["input_ids"].shape[1]

    def with_paths(session, t):
        msgs = build_messages(session, t, out)
        # recover image paths in the same order build_messages used them
        paths = [out / ex["image"] for ex in session["fewshot"]]
        if session["history"] == "full":
            paths += [out / p["image"] for p in session["turns"][:t]]
        paths.append(out / session["turns"][t]["image"])
        it = iter(paths)
        for m in msgs:
            if not isinstance(m["content"], str):
                for p in m["content"]:
                    if p["type"] == "image_url":
                        p["_path"] = str(next(it))
        return msgs

    shared = next(s for s in sessions if s["shared"])
    first = n_tokens(with_paths(shared, 0))
    last_t = len(shared["turns"]) - 1
    last = n_tokens(with_paths(shared, last_t))
    zero_shot = n_tokens([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}, "_path": str(out / shared["turns"][0]["image"]),
                                                       }, {"type": "text", "text": shared["turns"][0]["instruction"]}]}])
    stats = {"trace": a.name, "history": a.history, "tokens_first_turn_with_prefix": first, "tokens_last_turn": last,
             "last_turn_index": last_t, "tokens_single_image_no_prefix": zero_shot,
             "shared_prefix_tokens_approx": first - zero_shot,
             "note": "token counts from the real Qwen2-VL processor; 'approx' = first turn minus a bare one-image request"}
    (out / "token_stats.json").write_text(json.dumps({**stats, "env": envinfo.capture()}, indent=1))
    print(json.dumps(stats, indent=1))
